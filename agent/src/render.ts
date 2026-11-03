// Engine results -> Markdown for the editor's agent panel.
// Every number shown here comes straight from the engine's JSON.

import { periodUnit } from "./units";

const f1 = (v: number) => v.toFixed(1);
const f2 = (v: number) => v.toFixed(2);
const money = (v: number) => `$${v.toFixed(2)}M`;
const signed = (v: number, d = 2) => `${v >= 0 ? "+" : ""}${v.toFixed(d)}`;
const pct = (v: number, d = 1) => `${(100 * v).toFixed(d)}%`;

function table(head: string[], rows: (string | number)[][], align?: ("l" | "r")[]): string {
  const a = align ?? head.map((_, i) => (i === 0 ? "l" : "r"));
  return [
    `| ${head.join(" | ")} |`,
    `|${a.map((x) => (x === "r" ? "---:" : "---")).join("|")}|`,
    ...rows.map((r) => `| ${r.join(" | ")} |`),
  ].join("\n");
}

export function market(m: any): string {
  const rows = m.firms.map((f: string) => [
    f === m.leader ? `**${f}** (us)` : f,
    f2(m.firm_volume_mt[f]),
    f2(m.capacity_mt[f]),
    `${f1((100 * m.firm_volume_mt[f]) / m.capacity_mt[f])}%`,
    `${f1(m.firm_share_pct[f])}%`,
  ]);
  return [
    `**Market loaded:** ${m.source}`,
    `${m.firms.length} producers, ${m.grades.length} grades, ${m.regions.length} regions (${m.regions.join(", ")}), ${m.periods} quarters, ${m.observations.toLocaleString("en-US")} observations.`,
    "",
    table(["Producer", "Volume Mt/q", "Capacity", "Utilisation", "Market share"], rows),
    "",
    ...(m.checks ?? []).map((c: string) => `- ${c}`),
  ].join("\n");
}

export function demand(d: any): string {
  const el = Object.entries(d.own_price_elasticity_by_firm ?? {})
    .map(([f, v]) => `${f} ${v}`)
    .join(", ");
  const truth = d.ground_truth_alpha !== undefined
    ? `\n- **Ground truth** (synthetic data only): ${d.ground_truth_alpha}. ${d.alpha_ci95[0] <= d.ground_truth_alpha && d.ground_truth_alpha <= d.alpha_ci95[1] ? "Inside the 95% CI." : "Outside the 95% CI."}`
    : "";
  const nest = d.kind === "nested"
    ? `\n- **Within-grade nesting** sigma = ${d.sigma} (95% CI ${d.sigma_ci95[0]} to ${d.sigma_ci95[1]}${d.ground_truth_sigma != null ? `; truth ${d.ground_truth_sigma}` : ""}): buyers switch mostly between producers of the same grade, so each grade has its own competitive pressure.`
    : "";
  const alt = d.alternative
    ? `\n- The ${d.alternative.kind === "nested" ? "nested" : "plain"} logit alternative would put price sensitivity at ${d.alternative.alpha}${d.alternative.kind === "logit" ? ` (${signed((100 * (d.alternative.alpha - d.alpha)) / d.alpha, 0)}%): ignoring within-grade substitution misprices every grade` : ""}.`
    : "";
  return [
    `**Demand model:** ${d.method}, quarters ${d.window[0]}-${d.window[1]}, ${d.observations.toLocaleString("en-US")} observations.`,
    "",
    `- **Price sensitivity** ${d.alpha} per $/t (95% CI ${d.alpha_ci95[0]} to ${d.alpha_ci95[1]}), first-stage F = ${d.first_stage_f}${d.first_stage_f_nest ? ` (within-nest share F = ${d.first_stage_f_nest})` : ""}.${nest}${alt}`,
    `- **Plain OLS** would say ${d.ols_alpha}: ${signed(d.endogeneity_bias_pct, 1)}% off. Sellers see demand shocks when they set prices, so OLS is biased; cost shifters and freight fix it.${truth}`,
    `- **Own-price elasticity** (average by producer): ${el}.`,
    `- Demand shocks persist quarter to quarter (rho = ${d.xi_persistence}); the forecast carries them forward.`,
  ].join("\n");
}

export function backtest(b: any): string {
  const hasPrice = b.windows[0]?.mae_price_usd_t !== undefined;
  const rows = b.windows.map((w: any) => [
    `${w.window}q${w.window === b.retained_window ? " **(kept)**" : w.window === b.best?.window ? " (best)" : ""}`,
    `${f2(w.mae_share_pts)} vs ${f2(w.naive_mae_share_pts)} (${signed(w.skill_vs_naive_pct, 1)}%)`,
    ...(hasPrice ? [`${f2(w.mae_price_usd_t)} vs ${f2(w.naive_mae_price_usd_t)} (${signed(w.price_skill_vs_naive_pct, 1)}%)`, `${f1(w.price_direction_pct)}%`] : []),
    `${f1(w.directional_accuracy_pct)}%`,
  ]);
  const cmp = b.comparison
    ? `\n\n**Model choice: ${b.selected_model === "nested" ? "nested logit" : "plain logit"}**, because ${b.selection_reason}.`
    : "";
  return [
    `**Out-of-sample test:** for each quarter ${b.test_periods[0]}-${b.test_periods.at(-1)}, fit on the window before it and forecast the whole equilibrium (prices and shares) without seeing that quarter's demand shock. Benchmark: "same as last quarter".${cmp}`,
    "",
    table(["Calibration window", "Share error (pts) vs naive", ...(hasPrice ? ["Price error ($/t) vs naive", "Price direction right"] : []), "Share direction right"], rows),
    "",
    b.retained_window
      ? `Demand is re-estimated on the last ${b.retained_window} quarters: it forecasts materially better than the full sample.`
      : "**Full sample kept.** No shorter window cuts the share error by more than 2%, and the full sample pins down price sensitivity far more tightly.",
  ].join("\n");
}

function leaderRows(views: any[]): string {
  const keys: [string, string, (v: any) => string][] = [
    ["profit_musd", "Profit per quarter", (v) => money(v)],
    ["volume_mt", "Volume", (v) => `${v.toFixed(3)} Mt`],
    ["capacity_use_pct", "Capacity used", (v) => `${f1(v)}%`],
    ["market_share_pct", "Market share", (v) => `${f2(v)}%`],
    ["max_region_share_pct", "Largest region (% of our volume)", (v) => `${f1(v)}%`],
  ];
  return table(["", ...views.map((v) => v.label.split(" (")[0])], keys.map(([k, label, fmt]) => [label, ...views.map((v) => fmt(v[k]))]));
}

export function equilibrium(e: any): string {
  const firms = Object.keys(e.profit_musd);
  const rows = firms.map((f) => [f, money(e.profit_musd[f]), `${e.volume_mt[f].toFixed(3)} Mt`, e.shadow_price[f] > 0.01 ? `**$${f2(e.shadow_price[f])}/t** (at capacity)` : "0 (slack)"]);
  const k = e.kkt;
  return [
    "**Capacity-constrained Bertrand-Nash equilibrium** (every producer best-responds):",
    "",
    table(["Producer", "Profit", "Volume", "Capacity shadow price"], rows),
    "",
    `KKT certificate: stationarity ${k.stationarity_max_abs_usd_t.toExponential(1)} $/t, complementarity ${k.complementarity.toExponential(1)}, ${k.certified ? "**certified**" : "**NOT certified**"} (${e.iterations.outer} outer / ${e.iterations.inner_total} inner iterations).`,
    "",
    leaderRows([e.leader_status_quo, e.leader_at_nash]),
  ].join("\n");
}

export function recommendation(r: any): string {
  // Regions where we sell nothing in either plan (no route in the data) are left out.
  const regions = Object.keys(r.recommended.by_region).filter(
    (reg) => r.recommended.by_region[reg].volume_mt > 0 || r.status_quo.by_region[reg].volume_mt > 0);
  const rows = regions.map((reg) => {
    const a = r.status_quo.by_region[reg];
    const b = r.recommended.by_region[reg];
    return [reg, f1(a.price), `**${f1(b.price)}**`, signed(b.price - a.price, 1), b.volume_mt.toFixed(3), `${f1(b.share_of_our_volume_pct)}%`];
  });
  const rules = [
    r.rules?.max_region_share != null ? `no region above ${pct(r.rules.max_region_share, 0)} of our volume` : null,
    r.rules?.min_market_share != null ? `market share at least ${pct(r.rules.min_market_share, 0)}` : null,
  ].filter(Boolean);
  return [
    `**Recommended prices** (we commit first; rivals re-optimise)${rules.length ? `, under rule: ${rules.join(" and ")}` : ""}:`,
    "",
    table(["Region", "Status quo $/t", "Recommended $/t", "Move", "Volume Mt", "Share of our volume"], rows),
    "",
    `**Uplift: ${money(r.uplift_vs_status_quo_musd)} per quarter (${signed(r.uplift_vs_status_quo_pct)}%)** versus keeping last quarter's prices.${r.optimiser.feasible ? "" : " **Warning: the rule could not be met exactly.**"}`,
    r.optimiser.worlds
      ? `\n_Robust plan: chosen to do best on average over ${r.optimiser.worlds} sampled worlds (mean ${money(r.optimiser.mean_profit_musd)}, sd ${money(r.optimiser.sd_profit_musd)}, worst ${money(r.optimiser.worst_profit_musd)})._`
      : "",
    "",
    gradeTable(r.recommended),
    "",
    leaderRows([r.status_quo, r.nash, r.recommended]),
    "",
    `Rivals at capacity after our move: ${Object.entries(r.recommended.rivals_shadow_price).filter(([, v]) => (v as number) > 0.01).map(([k, v]) => `${k} ($${v}/t)`).join(", ") || "none"}.`,
  ].join("\n");
}

function gradeTable(view: any): string {
  const regions = Object.keys(view.by_region);
  const grades = [...new Set(regions.flatMap((r) => Object.keys(view.by_region[r].by_grade ?? {})))];
  if (grades.length < 2) return "";
  return [
    "Recommended price by grade ($/t FOB):",
    "",
    table(["Region", ...grades], regions.map((r) => [r, ...grades.map((g) => (view.by_region[r].by_grade?.[g] != null ? f1(view.by_region[r].by_grade[g]) : "–"))])),
  ].join("\n");
}

export function conduct(c: any): string {
  const rows = Object.entries(c.median_abs_gap_pct).map(([k, v]) => [k === c.best ? `**${c.labels[k]}**` : c.labels[k], `${f1(v as number)}%`]);
  return [
    "**Conduct test:** invert last quarter's prices into the marginal costs that would make them optimal under each behaviour, and compare with stated costs (firms with spare capacity only: " + c.firms_used.join(", ") + ").",
    "",
    table(["Market behaviour", "Median gap to stated costs"], rows),
    "",
    `Prices are most consistent with **${c.labels[c.best]}**.${c.best === "bertrand" ? " The competitive model the room uses is supported by the data." : " **The room's Bertrand model is not the best fit: treat recommendations with caution.**"}`,
  ].join("\n");
}

export function frontier(fr: any): string {
  const rows = fr.points.filter((p: any) => p.feasible).map((p: any) => [
    `${f1(p.cap_pct)}%`, f2(p.profit_musd), `${f2(p.give_up_pct)}%`, p.marginal_cost_musd_per_pt == null ? "" : p.marginal_cost_musd_per_pt.toFixed(3),
  ]);
  const last = fr.points.filter((p: any) => p.feasible).at(-1);
  return [
    `**Concentration frontier.** Unconstrained, our top region takes ${f1(fr.unconstrained_top_share_pct)}% of our volume. Tightening the cap:`,
    "",
    table(["Cap on any one region", "Profit $M/q", "Given up", "Marginal cost $M per point"], rows),
    "",
    `Reading: the first points of diversification are almost free; at a ${f1(last.cap_pct)}% cap the rule costs ${f2(last.give_up_pct)}% of profit. The marginal column is the price of one more point of safety.`,
  ].join("\n");
}

export function rule(r: any): string {
  const what = r.rule.max_region_share != null
    ? `no region above ${pct(r.rule.max_region_share, 0)} of our volume`
    : `market share at least ${pct(r.rule.min_market_share, 0)}`;
  return [
    `**Rule priced: ${what}.**`,
    "",
    `- Profit without the rule: ${money(r.profit_free_musd)} per quarter; with it: ${money(r.profit_with_rule_musd)}.`,
    `- **The rule costs ${money(r.cost_musd_per_period)} per quarter (${f2(r.cost_pct)}% of profit)**${periodUnit() === "year" ? "" : `, about ${money(4 * r.cost_musd_per_period)} a year`}.`,
    r.marginal_cost_next_point_musd != null ? `- One more point of tightening would cost a further ${money(r.marginal_cost_next_point_musd)} per quarter.` : "",
    r.feasible ? "" : "- **Warning: the rule cannot be met at any price.**",
  ].filter(Boolean).join("\n");
}

export function whatif(w: any): string {
  const s = w.scenario;
  const desc = [
    ...Object.entries(s.capacity_change ?? {}).map(([k, v]) => `${k} capacity ${signed(v as number)} Mt/q`),
    ...Object.entries(s.demand_change ?? {}).map(([k, v]) => `${k} demand ${signed(100 * (v as number), 0)}%`),
    ...Object.entries(s.cost_change ?? {}).map(([k, v]) => `${k} costs ${signed(100 * (v as number), 0)}%`),
    ...Object.entries(s.freight_change ?? {}).map(([k, v]) => `freight to ${k} ${signed(v as number, 0)} $/t`),
    ...(s.alpha_scale ? [`buyers' price sensitivity x${s.alpha_scale}`] : []),
  ].join("; ");
  const moves = Object.entries(w.price_moves).map(([k, v]) => [k, signed(v as number, 1)]);
  const rivals = Object.entries(w.rival_profit_change_musd).map(([k, v]) => [k, signed(v as number, 2), `${w.rival_shadow_prices_before[k]} -> ${w.rival_shadow_prices_after[k]}`]);
  return [
    `**What if: ${desc || s.name}**`,
    "",
    `- Our profit moves **${signed(w.impact_musd)} $M per quarter** once we react (${money(w.before.profit_musd)} -> ${money(w.after_reacting.profit_musd)}).`,
    `- If we keep today's recommended prices instead: ${money(w.after_not_reacting_profit_musd)}. **Reacting is worth ${money(w.value_of_reacting_musd)} per quarter.**`,
    "",
    table(["Region", "Our price move $/t"], moves),
    "",
    table(["Rival", "Profit change $M", "Capacity shadow price $/t"], rivals),
  ].join("\n");
}

export function stress(mc: any): string {
  return [
    `**Stress test:** ${mc.draws} joint draws of demand parameters (from the estimator's covariance), next-quarter demand shocks and market sizes. Rivals re-optimise in every draw; recommendation and status quo face the same draw.`,
    "",
    `- **Recommendation wins in ${f1(mc.prob_recommended_wins_pct)}% of futures.** Expected uplift ${money(mc.expected_uplift_musd)} per quarter.`,
    `- Uplift P10 / P50 / P90: ${signed(mc.uplift.p10)} / ${signed(mc.uplift.p50)} / ${signed(mc.uplift.p90)} $M. Worst draw ${signed(mc.worst_case_uplift_musd)}.`,
    `- Profit if we act, P10 / P50 / P90: ${f1(mc.recommended_profit.p10)} / ${f1(mc.recommended_profit.p50)} / ${f1(mc.recommended_profit.p90)} $M.`,
    "",
    "```",
    sparkHist(mc.histogram.counts, mc.histogram.edges),
    "```",
  ].join("\n");
}

function sparkHist(counts: number[], edges: number[]): string {
  const bars = "▁▂▃▄▅▆▇█";
  const max = Math.max(...counts, 1);
  const line = counts.map((c) => (c === 0 ? " " : bars[Math.min(7, Math.floor((c / max) * 7.999))])).join("");
  return `uplift $M  ${edges[0].toFixed(1)} ${line} ${edges.at(-1)!.toFixed(1)}`;
}

const MARK: Record<string, string> = { pass: "PASS", warn: "WARN", fail: "FAIL" };

export function redteam(r: any): string {
  return [
    `**Red team grade: ${r.grade}.** ${r.verdict}`,
    "",
    table(["Check", "Result", "Evidence"], r.checks.map((c: any) => [c.check, `**${MARK[c.status]}**`, c.evidence.replace(/\|/g, "/")]), ["l", "l", "l"]),
    "",
    "**What this model cannot tell you:**",
    ...r.limits.map((l: string) => `- ${l}`),
  ].join("\n");
}

export function memo(m: any): string {
  return [
    `**Decision memo written** (red-team grade ${m.grade}): ${m.headline}`,
    "",
    `Files in \`${m.dir}\`: ${m.files.map((f: string) => `\`${f}\``).join(", ")}. Open \`memo.html\` for the board version with charts.`,
  ].join("\n");
}

export function status(c: any): string {
  if (!c.market_loaded) return "Nothing computed yet. Try `/brief` for the full war room on the demo market, or `/market path/to/folder` for your data.";
  const done = [
    ["Market", c.market_loaded], ["Demand model", c.demand_fitted], ["Recommendation", c.recommendation],
    ["Stress test", c.stress_tested], ["Red team", c.red_team_grade ? `grade ${c.red_team_grade}` : false],
  ].map(([k, v]) => `- ${k}: ${v === true ? "done" : v || "not yet"}`);
  return [`**Session status** (advising ${c.leader}):`, ...done, c.whatifs.length ? `- Scenarios run: ${c.whatifs.join("; ")}` : ""].filter(Boolean).join("\n");
}

export function exported(e: any): string {
  return `**Demo market written** to \`${e.saved_to}\` (${e.files.join(", ")}). Replace the rows with your own data, keep the headers, then run \`/market ${e.saved_to}\`.`;
}
