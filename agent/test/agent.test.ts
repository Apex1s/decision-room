import { describe, expect, test } from "bun:test";
import * as acp from "@agentclientprotocol/sdk";
import { spawn } from "node:child_process";
import { Readable, Writable } from "node:stream";
import { join } from "node:path";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { parse, parseScenario } from "../src/intent";

const vocab = {
  firms: ["Atlas", "Boreal", "Cerro", "Dune"],
  regions: ["South Asia", "Latin America", "Europe", "East Asia", "Africa", "North America"],
  leader: "Atlas",
};

describe("intent parser", () => {
  test("slash commands", () => {
    expect(parse("/brief", vocab)).toEqual([{ kind: "brief" }]);
    expect(parse("/rule cap 22%", vocab)).toEqual([{ kind: "rule", maxRegionShare: 0.22 }]);
    expect(parse("/stress 500", vocab)).toEqual([{ kind: "stress", draws: 500 }]);
    expect(parse("/market data/q3", vocab)).toEqual([{ kind: "load", source: "data/q3" }]);
  });

  test("capacity scenarios", () => {
    expect(parseScenario("What if Dune adds 0.4 Mt of capacity?", vocab)?.capacity_change).toEqual({ Dune: 0.4 });
    expect(parseScenario("Boreal outage of 0.5 Mt", vocab)?.capacity_change).toEqual({ Boreal: -0.5 });
  });

  test("demand, cost and freight scenarios", () => {
    expect(parseScenario("what if demand in Europe falls 15%", vocab)?.demand_change).toEqual({ Europe: -0.15 });
    expect(parseScenario("what if demand in East Asia grows 10%", vocab)?.demand_change).toEqual({ "East Asia": 0.1 });
    expect(parseScenario("what if Boreal's energy cost rises 8%", vocab)?.cost_change).toEqual({ Boreal: 0.08 });
    expect(parseScenario("freight to South Asia up $12", vocab)?.freight_change).toEqual({ "South Asia": 12 });
  });

  test("plain-English routing", () => {
    expect(parse("What should we charge next quarter?", vocab)[0]).toEqual({ kind: "brief" });
    expect(parse("What does capping any region at 22% of our volume cost us?", vocab)).toContainEqual({ kind: "rule", maxRegionShare: 0.22 });
    expect(parse("How robust is this?", vocab)).toContainEqual({ kind: "stress", draws: 200 });
    expect(parse("hello", vocab)[0].kind).toBe("help");
  });
});

const HOME = mkdtempSync(join(tmpdir(), "dr-home-"));

function connect() {
  const agent = spawn(process.execPath, [join(import.meta.dir, "..", "src", "main.ts")], {
    stdio: ["pipe", "pipe", "inherit"],
    env: { ...process.env, DECISION_ROOM_HOME: HOME },
  });
  const stream = acp.ndJsonStream(Writable.toWeb(agent.stdin!), Readable.toWeb(agent.stdout!) as ReadableStream<Uint8Array>);
  return { agent, stream };
}

async function drain(session: any, seen: any[]): Promise<string> {
  for (;;) {
    const m = await session.nextUpdate();
    if (m.kind === "stop") return m.response.stopReason;
    seen.push(m.notification.update);
  }
}

const DEMO_DATA = join(import.meta.dir, "..", "..", "data", "demo");

describe("ACP data and control", () => {
  test("an attached data folder (resource_link) becomes the market", async () => {
    const { agent, stream } = connect();
    const seen: any[] = [];
    try {
      const stop = await acp.client({ name: "test" }).connectWith(stream, async (ctx) => {
        await ctx.request("initialize", { protocolVersion: acp.PROTOCOL_VERSION, clientCapabilities: {} });
        return ctx.buildSession(mkdtempSync(join(tmpdir(), "dr-"))).withSession(async (session) => {
          const uri = "file:///" + join(DEMO_DATA, "observations.csv").replace(/\\/g, "/").replace(/^\//, "");
          session.prompt([{ type: "text", text: "/status" }, { type: "resource_link", name: "observations.csv", uri }]);
          return drain(session, seen);
        });
      });
      expect(stop).toBe("end_turn");
      const text = seen.filter((u) => u.sessionUpdate === "agent_message_chunk").map((u) => u.content.text).join("");
      expect(text).toContain("Market loaded:");
      expect(text).toContain("data");
      expect(text).toContain("Session status");
    } finally {
      agent.kill();
    }
  }, 120_000);

  test("session/cancel stops a long turn", async () => {
    const { agent, stream } = connect();
    try {
      const stop = await acp.client({ name: "test" }).connectWith(stream, async (ctx) => {
        await ctx.request("initialize", { protocolVersion: acp.PROTOCOL_VERSION, clientCapabilities: {} });
        return ctx.buildSession(mkdtempSync(join(tmpdir(), "dr-"))).withSession(async (session) => {
          session.prompt("/brief");
          for (;;) {
            const m = await session.nextUpdate();
            if (m.kind === "stop") return m.response.stopReason;
            if (m.notification.update.sessionUpdate === "tool_call") await ctx.notify("session/cancel", { sessionId: session.sessionId });
          }
        });
      });
      expect(stop).toBe("cancelled");
    } finally {
      agent.kill();
    }
  }, 120_000);
});

describe("ACP session persistence", () => {
  test("a session survives the agent: list, load with replay, continue, delete", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "dr-"));
    // 1. First agent process: run a turn, then die.
    let sessionId = "";
    {
      const { agent, stream } = connect();
      try {
        await acp.client({ name: "test" }).connectWith(stream, async (ctx) => {
          await ctx.request("initialize", { protocolVersion: acp.PROTOCOL_VERSION, clientCapabilities: {} });
          return ctx.buildSession(cwd).withSession(async (session) => {
            sessionId = session.sessionId;
            session.prompt("/rule cap 22%");
            return drain(session, []);
          });
        });
      } finally {
        agent.kill();
      }
    }
    // 2. A fresh agent process reopens it.
    const { agent, stream } = connect();
    const replayed: any[] = [];
    try {
      await acp
        .client({ name: "test" })
        .onNotification("session/update", (ctx) => {
          if (ctx.params.sessionId === sessionId) replayed.push(ctx.params.update);
        })
        .connectWith(stream, async (ctx) => {
          const init = await ctx.request("initialize", { protocolVersion: acp.PROTOCOL_VERSION, clientCapabilities: {} });
          expect(init.agentCapabilities?.loadSession).toBe(true);
          const listed = await ctx.request("session/list", { cwd });
          expect(listed.sessions.map((s) => s.sessionId)).toContain(sessionId);
          expect(listed.sessions.find((s) => s.sessionId === sessionId)?.title).toBe("/rule cap 22%");

          await ctx.request("session/load", { sessionId, cwd, mcpServers: [] });
          const user = replayed.filter((u) => u.sessionUpdate === "user_message_chunk").map((u) => u.content.text).join("");
          const agentText = replayed.filter((u) => u.sessionUpdate === "agent_message_chunk").map((u) => u.content.text).join("");
          expect(user).toContain("/rule cap 22%");
          expect(agentText).toContain("Rule priced");

          // The engine state came back too: no market reload is needed to answer.
          replayed.length = 0;
          const res = await ctx.request("session/prompt", { sessionId, prompt: [{ type: "text", text: "/status" }] });
          expect(res.stopReason).toBe("end_turn");
          const status = replayed.filter((u) => u.sessionUpdate === "agent_message_chunk").map((u) => u.content.text).join("");
          expect(status).toContain("Demand model: done");
          expect(replayed.some((u) => u.sessionUpdate === "tool_call" && /Data Steward/.test(u.title))).toBe(false);

          await ctx.request("session/delete", { sessionId });
          const after = await ctx.request("session/list", { cwd });
          expect(after.sessions.map((s) => s.sessionId)).not.toContain(sessionId);
        });
    } finally {
      agent.kill();
    }
  }, 180_000);
});

describe("ACP end to end", () => {
  test("initialize, session, slash commands, a priced decision", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "dr-"));
    const agent = spawn(process.execPath, [join(import.meta.dir, "..", "src", "main.ts")], { stdio: ["pipe", "pipe", "inherit"] });
    const stream = acp.ndJsonStream(Writable.toWeb(agent.stdin!), Readable.toWeb(agent.stdout!) as ReadableStream<Uint8Array>);
    const seen: any[] = [];
    try {
      const result = await acp.client({ name: "test" }).connectWith(stream, async (ctx) => {
        const init = await ctx.request("initialize", { protocolVersion: acp.PROTOCOL_VERSION, clientCapabilities: {} });
        expect(init.agentInfo?.name).toBe("decision-room");
        return ctx.buildSession(cwd).withSession(async (session) => {
          expect(session.modes?.availableModes.map((m) => m.id)).toEqual(["advisor", "autopilot"]);
          session.prompt("/rule cap 22%");
          for (;;) {
            const m = await session.nextUpdate();
            if (m.kind === "stop") return m.response.stopReason;
            seen.push(m.notification.update);
          }
        });
      });
      expect(result).toBe("end_turn");
      const kinds = new Set(seen.map((u) => u.sessionUpdate));
      for (const k of ["available_commands_update", "plan", "tool_call", "tool_call_update", "agent_message_chunk"]) expect(kinds.has(k)).toBe(true);
      const text = seen.filter((u) => u.sessionUpdate === "agent_message_chunk").map((u) => u.content.text).join("");
      expect(text).toContain("Rule priced");
      expect(text).toMatch(/costs \$\d+\.\d\dM per quarter/);
      expect(seen.some((u) => u.sessionUpdate === "tool_call_update" && u.status === "completed")).toBe(true);
    } finally {
      agent.kill();
    }
  }, 120_000);
});
