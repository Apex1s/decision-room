import { describe, expect, test } from "bun:test";
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";
import { join } from "node:path";

const text = (r: any) => r.content.map((c: any) => c.text).join("\n");

describe("MCP server", () => {
  test("lists the specialists as tools and runs them on the shared engine", async () => {
    const transport = new StdioClientTransport({ command: process.execPath, args: [join(import.meta.dir, "..", "src", "mcp.ts")] });
    const client = new Client({ name: "test", version: "0.0.0" });
    await client.connect(transport);
    try {
      const tools = (await client.listTools()).tools.map((t) => t.name);
      for (const t of ["load_market", "validate_demand", "recommend_prices", "price_rule", "what_if", "test_conduct", "red_team", "write_memo"]) {
        expect(tools).toContain(t);
      }

      const rec = await client.callTool({ name: "recommend_prices", arguments: { max_region_share: 0.22 } });
      expect(rec.isError).toBeFalsy();
      expect(text(rec)).toContain("Recommended prices");
      expect(text(rec)).toContain("no region above 22% of our volume");

      const conduct = await client.callTool({ name: "test_conduct", arguments: {} });
      expect(text(conduct)).toContain("Bertrand competition");

      const scenario = await client.callTool({ name: "what_if", arguments: { name: "Dune expands", capacity_change: { Dune: 0.4 } } });
      expect(text(scenario)).toContain("Reacting is worth");

      const bad = await client.callTool({ name: "what_if", arguments: { capacity_change: { Nobody: 1 } } });
      expect(bad.isError).toBe(true);
      expect(text(bad)).toContain("unknown firm");

      const doc = await client.readResource({ uri: "decision-room://methodology" });
      expect(String((doc.contents[0] as any).text)).toContain("Berry");
    } finally {
      await client.close();
    }
  }, 180_000);
});
