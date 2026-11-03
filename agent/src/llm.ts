// Optional Claude layer. With ANTHROPIC_API_KEY set, Claude (1) reads free-form
// questions into engine actions and (2) writes a short executive summary from
// the engine's numbers. It never computes anything: every figure it may use is
// passed in, and the summary is labelled as written by the model.
// Without a key, the deterministic parser and templates are used instead.

import type { Action, Vocabulary } from "./intent";

const API = "https://api.anthropic.com/v1/messages";
export const MODEL = process.env.DECISION_ROOM_MODEL ?? "claude-opus-5-5";

export function llmEnabled(): boolean {
  return Boolean(process.env.ANTHROPIC_API_KEY) && process.env.DECISION_ROOM_OFFLINE !== "1";
}

async function complete(system: string, user: string, maxTokens = 800, signal?: AbortSignal): Promise<string> {
  const res = await fetch(API, {
    method: "POST",
    signal,
    headers: {
      "content-type": "application/json",
      "x-api-key": process.env.ANTHROPIC_API_KEY!,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({ model: MODEL, max_tokens: maxTokens, system, messages: [{ role: "user", content: user }] }),
  });
  if (!res.ok) throw new Error(`Claude API ${res.status}: ${(await res.text()).slice(0, 200)}`);
  const body: any = await res.json();
  return body.content.filter((b: any) => b.type === "text").map((b: any) => b.text).join("");
}

const ACTION_SCHEMA = `Reply with ONLY a JSON array of actions, no prose. Allowed actions:
{"kind":"brief","maxRegionShare"?:number}            full analysis end to end
{"kind":"demand"} {"kind":"backtest"} {"kind":"equilibrium"} {"kind":"frontier"} {"kind":"redteam"} {"kind":"memo"} {"kind":"status"}
{"kind":"recommend","maxRegionShare"?:number,"minMarketShare"?:number,"robust"?:boolean}   robust: best on average over uncertainty
{"kind":"conduct"}                                   do prices fit competition, collusion or price = cost?
{"kind":"rule","maxRegionShare"?:number,"minMarketShare"?:number}   price a commercial rule
{"kind":"whatif","scenario":{"name":string,"capacity_change"?:{firm:Mt},"demand_change"?:{region:fraction},"cost_change"?:{firm:fraction},"freight_change"?:{region:usd_per_t},"alpha_scale"?:number}}
{"kind":"stress","draws":number}
{"kind":"help","reason":string}                    when the request is outside pricing/market decisions
Shares and changes are fractions (22% -> 0.22, -10% -> -0.1). Capacity changes are in Mt per quarter.`;

export async function planWithClaude(prompt: string, vocab: Vocabulary, signal?: AbortSignal): Promise<Action[] | null> {
  const system = `You route requests for Decision Room, a pricing war room for a commodity producer.
The firm we advise: ${vocab.leader ?? "unknown"}. Firms: ${vocab.firms.join(", ") || "unknown"}. Regions: ${vocab.regions.join(", ") || "unknown"}.
Use exact firm and region names from these lists. ${ACTION_SCHEMA}`;
  try {
    const text = await complete(system, prompt, 600, signal);
    const json = text.slice(text.indexOf("["), text.lastIndexOf("]") + 1);
    const actions = JSON.parse(json) as Action[];
    return Array.isArray(actions) && actions.length ? actions : null;
  } catch {
    return null; // fall back to the deterministic parser
  }
}

export async function summariseWithClaude(question: string, facts: unknown, signal?: AbortSignal): Promise<string | null> {
  const system = `You are the chief of staff in a pricing war room. Write 3 to 5 sentences for an executive:
the decision, the money, the main risk, and how much to trust it. Use ONLY numbers that appear in the JSON facts;
never compute new ones or round them differently. If the red-team grade is C or D, say so plainly first.
Plain words, no hype, no bullet points, no em dashes.`;
  try {
    return (await complete(system, `Question: ${question}\n\nFacts (JSON):\n${JSON.stringify(facts).slice(0, 12000)}`, 400, signal)).trim();
  } catch {
    return null;
  }
}
