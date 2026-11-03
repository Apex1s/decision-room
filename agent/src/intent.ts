// Turns a prompt into a list of engine actions.
// Slash commands are exact; plain English goes through a deterministic parser
// (always available, no API key), optionally upgraded by Claude in llm.ts.

export type Scenario = {
  name: string;
  capacity_change?: Record<string, number>;
  demand_change?: Record<string, number>;
  cost_change?: Record<string, number>;
  freight_change?: Record<string, number>;
  alpha_scale?: number;
};

export type Action =
  | { kind: "load"; source: string }
  | { kind: "export"; folder: string }
  | { kind: "demand" }
  | { kind: "backtest" }
  | { kind: "equilibrium" }
  | { kind: "recommend"; maxRegionShare?: number; minMarketShare?: number; robust?: boolean }
  | { kind: "conduct" }
  | { kind: "frontier" }
  | { kind: "rule"; maxRegionShare?: number; minMarketShare?: number }
  | { kind: "whatif"; scenario: Scenario }
  | { kind: "stress"; draws: number }
  | { kind: "redteam" }
  | { kind: "memo" }
  | { kind: "brief"; maxRegionShare?: number }
  | { kind: "status" }
  | { kind: "help"; reason?: string };

export type Vocabulary = { firms: string[]; regions: string[]; leader?: string };

export const COMMANDS: { name: string; description: string; hint?: string }[] = [
  { name: "brief", description: "Full war room: data, demand, competition, decision, stress test, red team, memo", hint: "optional: cap 22%" },
  { name: "market", description: "Load a market folder (CSV) or the synthetic demo", hint: "demo | path/to/folder" },
  { name: "demand", description: "Estimate logit demand (OLS vs instrumented) and elasticities" },
  { name: "backtest", description: "Out-of-sample validation across calibration windows" },
  { name: "equilibrium", description: "Capacity-constrained Bertrand-Nash among all producers, KKT-certified" },
  { name: "recommend", description: "Our optimal prices by region and grade, anticipating rivals (Stackelberg)", hint: "optional: robust | cap 22% | share 25%" },
  { name: "conduct", description: "Test whether prices fit competition, collusion or price = cost" },
  { name: "frontier", description: "Profit vs how much of our volume any one region may take" },
  { name: "rule", description: "Price a commercial-security rule", hint: "cap 22% | share 25%" },
  { name: "whatif", description: "Scenario, with and without our reaction", hint: "Dune +0.4 Mt | demand Europe -10% | cost Boreal +8%" },
  { name: "stress", description: "Monte Carlo: how often the recommendation beats the status quo", hint: "draws, default 200" },
  { name: "redteam", description: "Adversarial checks on data, model, solver and decision; A-D grade" },
  { name: "memo", description: "Write the board-ready decision memo (Markdown, HTML, charts)" },
  { name: "export", description: "Write the demo market as CSV files to copy your own data into", hint: "folder" },
  { name: "status", description: "What has been computed so far in this session" },
  { name: "help", description: "What Decision Room can do" },
];

const pct = (s: string) => {
  const m = s.match(/(-?\d+(?:\.\d+)?)\s*%/);
  return m ? Number(m[1]) / 100 : undefined;
};

function find(list: string[], text: string): string | undefined {
  const t = text.toLowerCase();
  // Longest names first so "East Asia" wins over "Asia".
  return [...list].sort((a, b) => b.length - a.length).find((x) => t.includes(x.toLowerCase()));
}

function rules(text: string) {
  const t = text.toLowerCase();
  const out: { maxRegionShare?: number; minMarketShare?: number } = {};
  const cap = t.match(/(?:cap|no (?:single )?region|concentrat\w*|max(?:imum)?|above|over|exceed\w*)[^%]{0,40}?(\d+(?:\.\d+)?)\s*%/);
  if (cap) out.maxRegionShare = Number(cap[1]) / 100;
  const share = t.match(/(?:market share|share)\s*(?:of\s*)?(?:at least|above|>=|≥|min(?:imum)?|floor)?\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*%/);
  if (share && /(at least|>=|≥|min|floor|keep|defend|hold)/.test(t)) out.minMarketShare = Number(share[1]) / 100;
  return out;
}

export function parseScenario(text: string, vocab: Vocabulary): Scenario | undefined {
  const t = text.toLowerCase();
  const sc: Scenario = { name: text.trim().replace(/^\/?whatif\s*/i, "").slice(0, 80) || "scenario" };
  let hit = false;
  const firm = find(vocab.firms, text);
  const region = find(vocab.regions, text);
  const p = pct(text);

  const mt = t.match(/([+-]?\d+(?:\.\d+)?)\s*(?:mt|million t)/);
  if (firm && (mt || /(outage|shut|strike|closes|offline|expan|capacity|adds?)/.test(t))) {
    const losing = /(outage|shut|strike|closes|offline|loses|cut)/.test(t);
    let dk = mt ? Number(mt[1]) : undefined;
    if (dk !== undefined && losing) dk = -Math.abs(dk);
    if (dk === undefined && losing) dk = -0.3; // unspecified outage: a third of a Mt per quarter
    if (dk !== undefined) {
      sc.capacity_change = { [firm]: dk };
      hit = true;
    }
  }
  if (/cost|energy|sulfur|sulphur|input/.test(t) && p !== undefined) {
    const who = firm ?? vocab.leader;
    if (who) {
      const sign = /(fall|drop|down|lower|cut|decreas)/.test(t) ? -1 : 1;
      sc.cost_change = { [who]: sign * Math.abs(p) };
      hit = true;
    }
  }
  if (/demand|market size|consumption|imports?/.test(t) && region && p !== undefined) {
    const sign = /(fall|drop|down|lower|shrink|collapse|decreas|slump|-\s*\d)/.test(t) ? -1 : 1;
    sc.demand_change = { [region]: sign * Math.abs(p) };
    hit = true;
  }
  const usd = t.match(/\$\s*(\d+(?:\.\d+)?)|(\d+(?:\.\d+)?)\s*(?:\$|usd|dollars?)/);
  if (/freight|shipping|canal|route|tariff/.test(t) && region && usd) {
    const v = Number(usd[1] ?? usd[2]);
    sc.freight_change = { [region]: /(fall|drop|down|lower|cheaper)/.test(t) ? -v : v };
    hit = true;
  }
  if (/(price[- ]sensitiv|elastic|buyers? (?:get|become) (?:more|less))/.test(t) && p !== undefined) {
    sc.alpha_scale = 1 + (/(less)/.test(t) ? -1 : 1) * Math.abs(p);
    hit = true;
  }
  return hit ? sc : undefined;
}

export function parseSlash(cmd: string, arg: string, vocab: Vocabulary): Action[] {
  switch (cmd) {
    case "brief":
      return [{ kind: "brief", ...pick(rules(arg), "maxRegionShare") }];
    case "market":
    case "load":
      return [{ kind: "load", source: arg.trim() || "demo" }];
    case "export":
      return [{ kind: "export", folder: arg.trim() || "decision-room-data" }];
    case "demand":
      return [{ kind: "demand" }];
    case "backtest":
      return [{ kind: "backtest" }];
    case "equilibrium":
    case "nash":
      return [{ kind: "equilibrium" }];
    case "recommend":
    case "price":
      return [{ kind: "recommend", ...rules(arg), ...(/robust|uncertain|safe/i.test(arg) ? { robust: true } : {}) }];
    case "conduct":
    case "collusion":
      return [{ kind: "conduct" }];
    case "frontier":
      return [{ kind: "frontier" }];
    case "rule": {
      const r = rules(arg);
      if (r.maxRegionShare === undefined && r.minMarketShare === undefined) {
        const p = pct(arg);
        if (p !== undefined) r.maxRegionShare = p;
      }
      if (r.maxRegionShare === undefined && r.minMarketShare === undefined)
        return [{ kind: "help", reason: "Which rule? Try `/rule cap 22%` or `/rule share 25%`." }];
      return [{ kind: "rule", ...r }];
    }
    case "whatif": {
      const sc = parseScenario(arg, vocab);
      return sc ? [{ kind: "whatif", scenario: sc }] : [{ kind: "help", reason: `I could not read a scenario in "${arg}". Try \`/whatif Dune +0.4 Mt\`, \`/whatif demand Europe -10%\`, \`/whatif cost Boreal +8%\` or \`/whatif freight South Asia +$12\`.` }];
    }
    case "stress": {
      const n = Number(arg.match(/\d+/)?.[0] ?? 200);
      return [{ kind: "stress", draws: Math.min(Math.max(n, 20), 2000) }];
    }
    case "redteam":
    case "validate":
      return [{ kind: "redteam" }];
    case "memo":
    case "report":
      return [{ kind: "memo" }];
    case "status":
      return [{ kind: "status" }];
    default:
      return [{ kind: "help", reason: cmd === "help" ? undefined : `Unknown command /${cmd}.` }];
  }
}

function pick<T extends object, K extends keyof T>(o: T, k: K): Partial<T> {
  return o[k] === undefined ? {} : ({ [k]: o[k] } as Partial<T>);
}

export function parsePlain(text: string, vocab: Vocabulary): Action[] {
  const t = text.toLowerCase();
  const r = rules(text);
  const out: Action[] = [];
  const looksScenario = /(what if|what happens|suppose|scenario|if \w+ (?:adds|expands|shuts|cuts|raises|drops)|outage|strike|expands?)/.test(t);
  if (looksScenario) {
    const sc = parseScenario(text, vocab);
    if (sc) out.push({ kind: "whatif", scenario: sc });
  }
  if (/(cost of|what does .* cost|price (?:of )?(?:the|this|our) rule|how much .* rule|commercial[- ]security)/.test(t) &&
      (r.maxRegionShare !== undefined || r.minMarketShare !== undefined)) {
    out.push({ kind: "rule", ...r });
  }
  if (/(full|everything|war ?room|brief|board|end[- ]to[- ]end|from scratch|what should we (?:do|charge))/.test(t)) out.push({ kind: "brief", ...pick(r, "maxRegionShare") });
  if (/(frontier|trade-?off|concentration curve)/.test(t)) out.push({ kind: "frontier" });
  if (/(collu|cartel|conduct|really compet|honestly compet|(?:are|is) (?:they|the rivals|rivals) (?:\w+ )?competing)/.test(t)) out.push({ kind: "conduct" });
  if (/(nash|equilibri|competitors?|rivals?)/.test(t) && !out.length) out.push({ kind: "equilibrium" });
  if (/(elastic|demand curve|estimate demand|price sensitiv)/.test(t) && !looksScenario) out.push({ kind: "demand" });
  if (/(backtest|out[- ]of[- ]sample|accura)/.test(t)) out.push({ kind: "backtest" });
  if (/(recommend|optimi[sz]e|best price|what (?:price|should)|how should we price|reprice|pricing)/.test(t) &&
      !out.some((a) => a.kind === "brief" || a.kind === "whatif")) out.push({ kind: "recommend", ...r });
  if (/(robust (?:price|plan|recommend)|price (?:for|under) uncertainty|safest price)/.test(t)) out.push({ kind: "recommend", ...r, robust: true });
  else if (/(stress|monte carlo|risk|robust|how sure|confiden)/.test(t)) out.push({ kind: "stress", draws: 200 });
  if (/(red ?team|validate|audit|trust|can we believe|sanity)/.test(t)) out.push({ kind: "redteam" });
  if (/(memo|report|write (?:it )?up|deck|board pack)/.test(t) && !out.some((a) => a.kind === "brief")) out.push({ kind: "memo" });
  if (/(load|use) .*(data|folder|csv)|^demo$|synthetic/.test(t)) out.push({ kind: "load", source: "demo" });
  if (/(status|where are we|what have you (?:done|computed))/.test(t)) out.push({ kind: "status" });
  return out.length ? dedupe(out) : [{ kind: "help", reason: "I did not catch a decision question in that." }];
}

function dedupe(actions: Action[]): Action[] {
  const seen = new Set<string>();
  return actions.filter((a) => {
    const k = JSON.stringify(a);
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

export function parse(text: string, vocab: Vocabulary): Action[] {
  const trimmed = text.trim();
  const slash = trimmed.match(/^\/(\w+)\s*([\s\S]*)$/);
  if (slash) return parseSlash(slash[1].toLowerCase(), slash[2], vocab);
  return parsePlain(trimmed, vocab);
}
