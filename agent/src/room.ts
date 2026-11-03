// The war room: turns actions into specialist steps and streams them to the
// editor as ACP plan entries, tool calls and messages.

import { relabel, setPeriodUnit } from "./units";
import { isAbsolute, join, resolve } from "node:path";
import { readFileSync } from "node:fs";
import { Engine, EngineError } from "./engine";
import { parse, COMMANDS, type Action, type Vocabulary } from "./intent";
import { llmEnabled, planWithClaude, summariseWithClaude } from "./llm";
import * as R from "./render";

// What the room needs from the ACP connection. Kept minimal so tests can fake it.
export interface Channel {
  update(update: any): Promise<void>;
  askPermission(toolCallId: string, title: string, rawInput: unknown): Promise<"allow" | "always" | "reject">;
}

type Step = {
  id: string;
  who: string;
  title: string;
  kind: "read" | "edit" | "execute" | "think" | "search";
  run: (log: (msg: string) => void) => Promise<string>; // returns markdown
  chart?: "prices" | "frontier" | "risk" | "backtest"; // engine chart attached to the finished tool call
};

const SPECIALIST: Record<string, { who: string; kind: Step["kind"] }> = {
  load: { who: "Data Steward", kind: "read" },
  export: { who: "Data Steward", kind: "edit" },
  demand: { who: "Demand Economist", kind: "execute" },
  backtest: { who: "Demand Economist", kind: "execute" },
  equilibrium: { who: "Game Theorist", kind: "execute" },
  recommend: { who: "Strategist", kind: "execute" },
  conduct: { who: "Game Theorist", kind: "think" },
  frontier: { who: "Strategist", kind: "execute" },
  rule: { who: "Strategist", kind: "execute" },
  whatif: { who: "Risk Analyst", kind: "execute" },
  stress: { who: "Risk Analyst", kind: "execute" },
  redteam: { who: "Red Team", kind: "think" },
  memo: { who: "Memo Writer", kind: "edit" },
  status: { who: "Chief of Staff", kind: "think" },
};

// Inline SVG charts go only to clients that render them (our web console, or DECISION_ROOM_SVG=1).
// Editors such as Zed show an SVG resource as raw code, so they get the tables and the memo instead.
export const display = { svgCharts: process.env.DECISION_ROOM_SVG === "1" };

export class Room {
  readonly engine = new Engine();
  mode: "advisor" | "autopilot" = "advisor";
  private vocab: Vocabulary = { firms: [], regions: [] };
  private cancelled = false;
  private counter = 0;
  private facts: Record<string, any> = {};

  constructor(readonly cwd: string, private ch: Channel) {}

  cancel() {
    this.cancelled = true;
  }

  close() {
    this.engine.close();
  }

  private async refreshVocab() {
    const c = await this.engine.call("context");
    if (c.market_loaded) this.vocab = { firms: c.firms, regions: c.regions, leader: c.leader };
    setPeriodUnit(c.period_unit);
  }

  async handle(promptBlocks: any[]): Promise<"end_turn" | "cancelled"> {
    this.cancelled = false;
    this.facts = {};
    const text = promptBlocks.filter((b) => b.type === "text").map((b) => b.text).join("\n").trim();
    const actions: Action[] = [];
    // Attached files or folders become the market to analyse.
    for (const b of promptBlocks) {
      const uri: string | undefined = b.type === "resource_link" ? b.uri : b.type === "resource" ? b.resource?.uri : undefined;
      if (uri?.startsWith("file://")) {
        let path = decodeURIComponent(uri.replace(/^file:\/\//, "")).replace(/^\/([A-Za-z]:)/, "$1");
        if (/\.csv$/i.test(path)) path = path.replace(/[\\/][^\\/]+$/, "");
        actions.push({ kind: "load", source: path });
      }
    }

    await this.refreshVocab();
    // The parser needs firm and region names, so the first question loads a market
    // (the attached one, or the demo) before it is read.
    if (!this.vocab.firms.length && !/^\/(market|load|export|help)\b/i.test(text)) {
      const attached = actions.find((a): a is Extract<Action, { kind: "load" }> => a.kind === "load");
      await this.engine.call("load", { source: attached?.source ?? "demo" });
      await this.refreshVocab();
      if (!attached) actions.push({ kind: "load", source: "demo" });
    }
    let planned: Action[] | null = null;
    if (text && !text.startsWith("/") && llmEnabled()) {
      await this.say("_Routing your question with Claude…_\n\n");
      planned = await planWithClaude(text, this.vocab);
    }
    actions.push(...(planned ?? (text ? parse(text, this.vocab) : [])));
    if (!actions.length) actions.push({ kind: "help" });

    const steps = this.expand(actions);
    if (!steps.length) return "end_turn";
    await this.ch.update({
      sessionUpdate: "plan",
      entries: steps.map((s) => ({ content: `${s.who}: ${s.title}`, priority: "high", status: "pending" })),
    });

    for (let i = 0; i < steps.length; i++) {
      if (this.cancelled) return "cancelled";
      await this.ch.update({
        sessionUpdate: "plan",
        entries: steps.map((s, j) => ({ content: `${s.who}: ${s.title}`, priority: "high", status: j < i ? "completed" : j === i ? "in_progress" : "pending" })),
      });
      const ok = await this.runStep(steps[i]);
      if (!ok) return this.cancelled ? "cancelled" : "end_turn";
    }
    await this.ch.update({
      sessionUpdate: "plan",
      entries: steps.map((s) => ({ content: `${s.who}: ${s.title}`, priority: "high", status: "completed" })),
    });
    if (this.cancelled) return "cancelled";
    await this.closingSummary(text || "full brief");
    return "end_turn";
  }

  private async say(text: string) {
    await this.ch.update({ sessionUpdate: "agent_message_chunk", content: { type: "text", text: relabel(text) } });
  }

  private async runStep(step: Step): Promise<boolean> {
    const toolCallId = `dr_${++this.counter}`;
    const log: string[] = [];
    await this.ch.update({
      sessionUpdate: "tool_call", toolCallId, title: `${step.who}: ${step.title}`, kind: step.kind, status: "in_progress",
      rawInput: { specialist: step.who, task: step.title },
    });
    const progress = (msg: string) => {
      log.push(relabel(`• ${msg}`));
      void this.ch.update({
        sessionUpdate: "tool_call_update", toolCallId,
        content: [{ type: "content", content: { type: "text", text: log.join("\n") } }],
      });
    };
    const t0 = Date.now();
    try {
      const md = await step.run(progress);
      const secs = ((Date.now() - t0) / 1000).toFixed(1);
      const content: any[] = [{ type: "content", content: { type: "text", text: [...log, `✓ done in ${secs}s`].join("\n") } }];
      if (step.chart && display.svgCharts) {
        const c = await this.engine.call("chart", { kind: step.chart }).catch(() => null);
        if (c?.svg) {
          content.push({
            type: "content",
            content: { type: "resource", resource: { uri: `decision-room://chart/${step.chart}.svg`, mimeType: "image/svg+xml", text: c.svg } },
          });
        }
      }
      await this.ch.update({ sessionUpdate: "tool_call_update", toolCallId, status: "completed", content });
      if (md) await this.say(`### ${step.who}\n\n${md}\n\n`);
      return true;
    } catch (err) {
      const msg = err instanceof EngineError || err instanceof Error ? err.message : String(err);
      await this.ch.update({
        sessionUpdate: "tool_call_update", toolCallId, status: "failed",
        content: [{ type: "content", content: { type: "text", text: [...log, `✗ ${msg.split("\n")[0]}`].join("\n") } }],
      });
      await this.say(`**${step.who} stopped:** ${msg.split("\n")[0]}\n`);
      return false;
    }
  }

  private step(kind: string, title: string, run: Step["run"], chart?: Step["chart"]): Step {
    const s = SPECIALIST[kind] ?? { who: "Chief of Staff", kind: "think" as const };
    return { id: kind, who: s.who, kind: s.kind, title, run, chart };
  }

  private call(method: string, params: Record<string, unknown> = {}) {
    return (log: (m: string) => void) => this.engine.call(method, params, log);
  }

  private expand(actions: Action[]): Step[] {
    const steps: Step[] = [];
    for (const a of actions) {
      switch (a.kind) {
        case "help":
          steps.push(this.step("status", "Explaining what Decision Room can do", async () => help(a.reason)));
          break;
        case "status":
          steps.push(this.step("status", "Reviewing what is computed", async (log) => R.status(await this.call("context")(log))));
          break;
        case "load":
          steps.push(this.step("load", a.source === "demo" ? "Loading the synthetic demo market" : `Loading ${a.source}`, async (log) => {
            const source = a.source === "demo" || isAbsolute(a.source) ? a.source : resolve(this.cwd, a.source);
            const m = await this.call("load", { source })(log);
            this.facts.market = m;
            await this.refreshVocab();
            return R.market(m);
          }));
          break;
        case "export":
          steps.push(this.step("export", `Writing the demo market to ${a.folder}`, async (log) =>
            R.exported(await this.call("export", { folder: resolve(this.cwd, a.folder) })(log))));
          break;
        case "demand":
          steps.push(this.step("demand", "Estimating demand: Berry inversion, OLS vs 2SLS", async (log) => {
            const d = await this.call("fit_demand")(log);
            this.facts.demand = d;
            return R.demand(d);
          }));
          break;
        case "backtest":
          steps.push(this.backtestStep());
          break;
        case "equilibrium":
          steps.push(this.step("equilibrium", "Solving the capacity-constrained Bertrand-Nash game", async (log) =>
            R.equilibrium(await this.call("equilibrium")(log))));
          break;
        case "recommend":
          steps.push(this.recommendStep(a.maxRegionShare, a.minMarketShare, a.robust));
          break;
        case "conduct":
          steps.push(this.step("conduct", "Testing market conduct: competition, collusion or price = cost", async (log) =>
            R.conduct(await this.call("conduct")(log))));
          break;
        case "frontier":
          steps.push(this.step("frontier", "Tracing profit vs concentration", async (log) => {
            const f = await this.call("tradeoff_frontier")(log);
            this.facts.frontier = f;
            return R.frontier(f);
          }, "frontier"));
          break;
        case "rule":
          steps.push(this.step("rule", "Pricing the commercial-security rule", async (log) => {
            const r = await this.call("price_rule", { max_region_share: a.maxRegionShare ?? null, min_market_share: a.minMarketShare ?? null })(log);
            this.facts.rule = r;
            return R.rule(r);
          }));
          break;
        case "whatif":
          steps.push(this.step("whatif", `Scenario: ${a.scenario.name}`, async (log) => {
            const w = await this.call("whatif", { spec: a.scenario })(log);
            this.facts.whatif = w;
            return R.whatif(w);
          }));
          break;
        case "stress":
          steps.push(this.stressStep(a.draws));
          break;
        case "redteam":
          steps.push(this.redteamStep());
          break;
        case "memo":
          steps.push(this.memoStep());
          break;
        case "brief": {
          steps.push(this.backtestStep(true));
          steps.push(this.step("equilibrium", "Solving the capacity-constrained Bertrand-Nash game", async (log) =>
            R.equilibrium(await this.call("equilibrium")(log))));
          steps.push(this.recommendStep(a.maxRegionShare));
          steps.push(this.step("frontier", "Tracing profit vs concentration", async (log) => {
            const f = await this.call("tradeoff_frontier")(log);
            this.facts.frontier = f;
            return R.frontier(f);
          }, "frontier"));
          steps.push(this.stressStep(200));
          steps.push(this.redteamStep());
          steps.push(this.memoStep());
          break;
        }
      }
    }
    return steps;
  }

  private backtestStep(withDemand = false): Step {
    return this.step("backtest", withDemand ? "Validating out of sample, then estimating demand on the best window" : "Backtesting across calibration windows", async (log) => {
      const b = await this.call("backtest")(log);
      this.facts.backtest = b.best;
      if (!withDemand) return R.backtest(b);
      const d = await this.call("fit_demand", { window: b.retained_window })(log);
      this.facts.demand = d;
      return `${R.backtest(b)}\n\n${R.demand(d)}`;
    }, "backtest");
  }

  private recommendStep(maxRegionShare?: number, minMarketShare?: number, robust = false): Step {
    const title = robust ? "Optimising our prices on average over sampled worlds (robust)" : "Optimising our prices, anticipating rivals (Stackelberg)";
    return this.step("recommend", title, async (log) => {
      const r = await this.call("recommend", { max_region_share: maxRegionShare ?? null, min_market_share: minMarketShare ?? null, robust })(log);
      this.facts.recommendation = {
        uplift_musd: r.uplift_vs_status_quo_musd, uplift_pct: r.uplift_vs_status_quo_pct,
        price_moves: r.price_moves_vs_status_quo, profit: r.recommended.profit_musd, status_quo_profit: r.status_quo.profit_musd,
      };
      return R.recommendation(r);
    }, "prices");
  }

  private stressStep(draws: number): Step {
    return this.step("stress", `Stress-testing the decision (${draws} simulated futures)`, async (log) => {
      const s = await this.call("stress", { draws })(log);
      this.facts.stress = { win_pct: s.prob_recommended_wins_pct, uplift: s.uplift, expected: s.expected_uplift_musd };
      return R.stress(s);
    }, "risk");
  }

  private redteamStep(): Step {
    return this.step("redteam", "Attacking the recommendation before anyone sees it", async (log) => {
      const r = await this.call("red_team")(log);
      this.facts.red_team = { grade: r.grade, verdict: r.verdict, warnings: r.checks.filter((c: any) => c.status !== "pass").map((c: any) => c.evidence) };
      return R.redteam(r);
    });
  }

  private memoStep(): Step {
    return this.step("memo", "Writing the board memo", async (log) => {
      const outDir = join(this.cwd, "decision-room-out");
      const id = `dr_${this.counter}`;
      if (this.mode === "advisor") {
        const choice = await this.ch.askPermission(id, `Write the decision memo to ${outDir}`, { dir: outDir });
        if (choice === "reject") return "Memo not written (you declined). Everything above stays available in this session.";
        if (choice === "always") this.mode = "autopilot";
      }
      const m = await this.call("write_memo", { out_dir: outDir })(log);
      this.facts.memo = { headline: m.headline, grade: m.grade };
      const md = readFileSync(join(m.dir, "memo.md"), "utf8");
      await this.ch.update({
        sessionUpdate: "tool_call_update", toolCallId: id,
        content: [{ type: "diff", path: join(m.dir, "memo.md"), oldText: null, newText: md }],
        locations: [{ path: join(m.dir, "memo.md") }],
      });
      return R.memo(m);
    });
  }

  private async closingSummary(question: string) {
    const f = this.facts;
    if (!f.recommendation && !f.whatif && !f.rule && !f.red_team) return;
    if (llmEnabled()) {
      const s = await summariseWithClaude(question, f);
      if (s) {
        await this.say(`---\n**Chief of staff** _(written by Claude from the numbers above)_\n\n${s}\n`);
        return;
      }
    }
    const lines: string[] = [];
    if (f.red_team && ["C", "D"].includes(f.red_team.grade)) lines.push(`Red team grade ${f.red_team.grade}: ${f.red_team.verdict}`);
    if (f.recommendation) {
      const r = f.recommendation;
      lines.push(`Reprice for ${r.uplift_musd >= 0 ? "+" : ""}$${r.uplift_musd.toFixed(2)}M per quarter (${r.uplift_pct >= 0 ? "+" : ""}${r.uplift_pct.toFixed(2)}%) versus last quarter's prices.`);
    }
    if (f.stress) lines.push(`It beats the status quo in ${f.stress.win_pct.toFixed(1)}% of simulated futures (P10 uplift ${f.stress.uplift.p10 >= 0 ? "+" : ""}${f.stress.uplift.p10.toFixed(2)} $M).`);
    if (f.rule) lines.push(`The tested rule costs $${f.rule.cost_musd_per_period.toFixed(2)}M per quarter (${f.rule.cost_pct.toFixed(2)}% of profit).`);
    if (f.whatif) lines.push(`Under the scenario, reacting is worth $${f.whatif.value_of_reacting_musd.toFixed(2)}M per quarter versus standing still.`);
    if (f.red_team && !["C", "D"].includes(f.red_team.grade)) lines.push(`Red team grade ${f.red_team.grade}${f.red_team.warnings.length ? `; caveat: ${f.red_team.warnings[0]}` : "."}`);
    if (lines.length) await this.say(`---\n**Bottom line.** ${lines.join(" ")}\n`);
  }
}

function help(reason?: string): string {
  return [
    reason ? `${reason}\n` : "",
    "**Decision Room** is a pricing war room for producers in concentrated markets. Ask in plain English, or use a command:",
    "",
    ...COMMANDS.map((c) => `- \`/${c.name}${c.hint ? ` ${c.hint}` : ""}\`: ${c.description}`),
    "",
    "Try: `What should we charge next quarter?` · `What if Dune adds 0.4 Mt?` · `What does capping any region at 22% cost us?` · `/brief`",
    llmEnabled() ? "" : "\n_Running offline: deterministic parser, no API key needed. Set `ANTHROPIC_API_KEY` to let Claude read free-form questions and write the executive summary._",
  ].join("\n");
}
