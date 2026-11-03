#!/usr/bin/env bun
// Decision Room as an MCP server: the same engine and specialists, exposed as
// tools to any MCP client (Claude Code, Claude Desktop, other agents).
//
// ACP puts Decision Room inside an editor as the agent you talk to; MCP lets
// another agent call it as a tool. Same numbers either way.
//
//   claude mcp add decision-room -- bun /path/to/decision-room/agent/src/mcp.ts

import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { readFileSync } from "node:fs";
import { join, resolve } from "node:path";
import { z } from "zod";
import { Engine, ENGINE_DIR } from "./engine";
import * as R from "./render";

const VERSION = "0.2.0";
const engine = new Engine();
const server = new McpServer({ name: "decision-room", version: VERSION });

type Out = { content: { type: "text"; text: string }[]; isError?: boolean };

async function run(method: string, params: Record<string, unknown>, render: (r: any) => string): Promise<Out> {
  try {
    const result = await engine.call(method, params);
    return { content: [{ type: "text", text: render(result) }] };
  } catch (err) {
    return { content: [{ type: "text", text: `Engine error: ${(err as Error).message.split("\n")[0]}` }], isError: true };
  }
}

const share = z.number().min(0).max(1).optional();
const readOnly = { readOnlyHint: true, openWorldHint: false } as const;

server.registerTool("load_market", {
  title: "Load market data",
  description: "Load a market folder (five CSV files: firms, regions, products, freight, observations; optional capacity) or 'demo' for the synthetic market. Every other tool loads the demo if nothing is loaded.",
  inputSchema: { source: z.string().default("demo").describe("Absolute folder path, or 'demo'") },
  annotations: { openWorldHint: false },
}, async ({ source }) => run("load", { source: source === "demo" ? source : resolve(source) }, R.market));

server.registerTool("validate_demand", {
  title: "Select and validate the demand model",
  description: "Forecast each of the last 8 quarters' equilibrium (prices and shares) out of sample with plain and nested logit, pick the better model, and report skill against a naive forecast.",
  annotations: readOnly,
}, async () => run("backtest", {}, R.backtest));

server.registerTool("estimate_demand", {
  title: "Estimate demand",
  description: "Berry-inversion demand with instruments (2SLS): price sensitivity, within-grade nesting, elasticities, OLS bias.",
  inputSchema: { model: z.enum(["auto", "logit", "nested"]).default("auto") },
  annotations: readOnly,
}, async ({ model }) => run("fit_demand", { model: model === "auto" ? null : model }, R.demand));

server.registerTool("solve_equilibrium", {
  title: "Competitive equilibrium",
  description: "Capacity-constrained Bertrand-Nash equilibrium among all producers, with capacity shadow prices and a KKT certificate.",
  annotations: readOnly,
}, async () => run("equilibrium", {}, R.equilibrium));

server.registerTool("test_conduct", {
  title: "Market conduct test",
  description: "Do observed prices fit Bertrand competition, collusion, or price = cost? Inverts prices into implied costs under each.",
  annotations: readOnly,
}, async () => run("conduct", {}, R.conduct));

server.registerTool("recommend_prices", {
  title: "Recommend prices",
  description: "Our profit-maximising prices by region and grade as a Stackelberg leader (rivals re-optimise), optionally under commercial rules or robust to uncertainty.",
  inputSchema: {
    max_region_share: share.describe("No region may take more than this fraction of our volume, e.g. 0.22"),
    min_market_share: share.describe("Keep our share of the whole market at least this fraction"),
    robust: z.boolean().default(false).describe("Optimise the average over sampled worlds instead of the expected world"),
  },
  annotations: readOnly,
}, async (a) => run("recommend", { max_region_share: a.max_region_share ?? null, min_market_share: a.min_market_share ?? null, robust: a.robust }, R.recommendation));

server.registerTool("price_rule", {
  title: "Price a commercial rule",
  description: "What a commercial-security rule costs per quarter, and the cost of tightening it one more point.",
  inputSchema: { max_region_share: share, min_market_share: share },
  annotations: readOnly,
}, async (a) => run("price_rule", { max_region_share: a.max_region_share ?? null, min_market_share: a.min_market_share ?? null }, R.rule));

server.registerTool("concentration_frontier", {
  title: "Concentration frontier",
  description: "Profit as the cap on any one region's share of our volume tightens, with the marginal cost of each point.",
  annotations: readOnly,
}, async () => run("tradeoff_frontier", {}, R.frontier));

server.registerTool("what_if", {
  title: "What-if scenario",
  description: "Re-solve the market under a scenario and compare reacting with standing still.",
  inputSchema: {
    name: z.string().default("scenario"),
    capacity_change: z.record(z.string(), z.number()).optional().describe("firm -> Mt per quarter added (negative for outages)"),
    demand_change: z.record(z.string(), z.number()).optional().describe("region -> fractional change in market size, e.g. -0.1"),
    cost_change: z.record(z.string(), z.number()).optional().describe("firm -> fractional change in marginal cost"),
    freight_change: z.record(z.string(), z.number()).optional().describe("region -> $/t added to every seller's freight"),
  },
  annotations: readOnly,
}, async (spec) => run("whatif", { spec }, R.whatif));

server.registerTool("stress_test", {
  title: "Stress test",
  description: "Monte Carlo over parameters, demand shocks and market sizes: how often the recommendation beats the status quo.",
  inputSchema: { draws: z.number().int().min(20).max(2000).default(200) },
  annotations: readOnly,
}, async ({ draws }) => run("stress", { draws }, R.stress));

server.registerTool("red_team", {
  title: "Red team",
  description: "Adversarial checks on data, demand model, conduct, solver and decision, with a mechanical A-D grade and stated limits.",
  annotations: readOnly,
}, async () => run("red_team", {}, R.redteam));

server.registerTool("write_memo", {
  title: "Write the decision memo",
  description: "Write the board memo (Markdown, HTML with charts, SVGs) to a folder.",
  inputSchema: { out_dir: z.string().describe("Absolute folder to write into") },
  annotations: { readOnlyHint: false, destructiveHint: false, openWorldHint: false },
}, async ({ out_dir }) => run("write_memo", { out_dir: resolve(out_dir) }, R.memo));

server.registerTool("status", {
  title: "Session status",
  description: "What has been computed so far.",
  annotations: readOnly,
}, async () => run("context", {}, R.status));

server.registerResource("methodology", "decision-room://methodology", {
  title: "Decision Room methodology",
  description: "Every assumption the engine makes: demand, conduct, equilibrium, optimisation, risk, red team.",
  mimeType: "text/markdown",
}, async (uri) => ({
  contents: [{ uri: uri.href, mimeType: "text/markdown", text: readFileSync(join(ENGINE_DIR, "..", "docs", "METHODOLOGY.md"), "utf8") }],
}));

await server.connect(new StdioServerTransport());
const stop = () => {
  engine.close();
  process.exit(0);
};
process.on("SIGINT", stop);
process.stdin.on("end", stop);
