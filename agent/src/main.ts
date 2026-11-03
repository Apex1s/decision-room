#!/usr/bin/env bun
// Decision Room as an ACP agent: any ACP client (Zed, JetBrains, the bundled
// CLI or web console) launches this process and talks to it over stdio.
//
// Sessions persist: every update is recorded and the engine's workspace is
// snapshotted after each turn, so clients can list, reopen (session/load, with
// the full conversation replayed) and delete past war rooms.

import * as acp from "@agentclientprotocol/sdk";
import { Readable, Writable } from "node:stream";
import { randomUUID } from "node:crypto";
import { Room, display } from "./room";
import { COMMANDS } from "./intent";
import { llmEnabled, MODEL } from "./llm";
import { SessionStore } from "./store";

export const VERSION = "0.2.0";
const rooms = new Map<string, { room: Room; store: SessionStore }>();

const MODES = [
  { id: "advisor", name: "Advisor", description: "Runs the analysis; asks before writing files" },
  { id: "autopilot", name: "Autopilot", description: "Runs everything and writes the memo without asking" },
];

function get(sessionId: string) {
  const r = rooms.get(sessionId);
  if (!r) throw acp.RequestError.sessionNotFound(sessionId);
  return r;
}

function openRoom(sessionId: string, cwd: string, store: SessionStore, client: acp.AgentContext): Room {
  const room = new Room(cwd, {
    update: (update) => {
      store.append({ update });
      return client.notify("session/update", { sessionId, update });
    },
    askPermission: async (toolCallId, title, rawInput) => {
      const res = await client.request("session/request_permission", {
        sessionId,
        toolCall: { toolCallId, title, kind: "edit", status: "pending", rawInput },
        options: [
          { optionId: "allow", name: "Write it", kind: "allow_once" },
          { optionId: "always", name: "Always write (switch to Autopilot)", kind: "allow_always" },
          { optionId: "reject", name: "Skip", kind: "reject_once" },
        ],
      });
      if (res.outcome.outcome === "cancelled") return "reject";
      if (res.outcome.optionId === "always") {
        void client.notify("session/update", { sessionId, update: { sessionUpdate: "current_mode_update", currentModeId: "autopilot" } });
      }
      return res.outcome.optionId as "allow" | "always" | "reject";
    },
  });
  rooms.get(sessionId)?.room.close();
  rooms.set(sessionId, { room, store });
  return room;
}

function advertise(client: acp.AgentContext, sessionId: string, greeting: string | null) {
  // Sent once the client knows the session exists.
  setTimeout(() => {
    void client.notify("session/update", {
      sessionId,
      update: {
        sessionUpdate: "available_commands_update",
        availableCommands: COMMANDS.map((c) => ({ name: c.name, description: c.description, input: c.hint ? { hint: c.hint } : null })),
      },
    });
    if (greeting) void client.notify("session/update", { sessionId, update: { sessionUpdate: "agent_message_chunk", content: { type: "text", text: greeting } } });
  }, 50);
}

const input = Writable.toWeb(process.stdout);
const output = Readable.toWeb(process.stdin) as ReadableStream<Uint8Array>;

acp
  .agent({ name: "decision-room" })
  .onRequest("initialize", async (ctx) => {
    if (ctx.params.clientInfo?.name === "decision-room-web") display.svgCharts = true;
    return {
    protocolVersion: acp.PROTOCOL_VERSION,
    agentCapabilities: {
      loadSession: true,
      promptCapabilities: { image: false, audio: false, embeddedContext: true },
      sessionCapabilities: { list: {}, delete: {} },
    },
    agentInfo: { name: "decision-room", title: "Decision Room", version: VERSION },
    authMethods: [],
    };
  })
  .onRequest("authenticate", async () => ({}))
  .onRequest("session/new", async (ctx) => {
    const sessionId = randomUUID();
    const room = openRoom(sessionId, ctx.params.cwd, SessionStore.create(sessionId, ctx.params.cwd), ctx.client);
    advertise(ctx.client, sessionId,
      `**Decision Room ${VERSION}** ready. ${llmEnabled() ? `Claude (${MODEL}) routes questions and writes summaries; the engine computes every number.` : "Offline mode: no API key needed."} Ask *What should we charge next quarter?* or run \`/brief\`.\n`);
    return { sessionId, modes: { currentModeId: room.mode, availableModes: MODES } };
  })
  .onRequest("session/load", async (ctx) => {
    const { sessionId, cwd } = ctx.params;
    const store = SessionStore.open(sessionId);
    if (!store) throw acp.RequestError.resourceNotFound(sessionId);
    const room = openRoom(sessionId, cwd, store, ctx.client);
    if (store.hasState()) await room.engine.call("restore_state", { path: store.statePath });
    // Replay the conversation before answering, as the protocol requires.
    for (const entry of store.history()) {
      if ("user" in entry) {
        for (const block of entry.user) {
          await ctx.client.notify("session/update", { sessionId, update: { sessionUpdate: "user_message_chunk", content: block } });
        }
      } else {
        await ctx.client.notify("session/update", { sessionId, update: entry.update });
      }
    }
    advertise(ctx.client, sessionId, null);
    return { modes: { currentModeId: room.mode, availableModes: MODES } };
  })
  .onRequest("session/list", async (ctx) => ({
    sessions: SessionStore.list(ctx.params.cwd ?? undefined).map((m) => ({
      sessionId: m.sessionId, cwd: m.cwd, title: m.title, updatedAt: m.updatedAt,
    })),
  }))
  .onRequest("session/delete", async (ctx) => {
    rooms.get(ctx.params.sessionId)?.room.close();
    rooms.delete(ctx.params.sessionId);
    SessionStore.open(ctx.params.sessionId)?.delete();
    return {};
  })
  .onRequest("session/set_mode", async (ctx) => {
    const { room } = get(ctx.params.sessionId);
    if (!MODES.some((m) => m.id === ctx.params.modeId)) {
      throw acp.RequestError.invalidParams({ modeId: ctx.params.modeId }, `Unknown mode ${ctx.params.modeId}`);
    }
    room.mode = ctx.params.modeId as Room["mode"];
    return {};
  })
  .onRequest("session/prompt", async (ctx) => {
    const { room, store } = get(ctx.params.sessionId);
    store.recordPrompt(ctx.params.prompt);
    const stopReason = await room.handle(ctx.params.prompt);
    try {
      await room.engine.call("save_state", { path: store.statePath });
    } catch {
      // A failed snapshot only costs recomputation after a reload.
    }
    return { stopReason };
  })
  .onNotification("session/cancel", async (ctx) => {
    rooms.get(ctx.params.sessionId)?.room.cancel();
  })
  .connect(acp.ndJsonStream(input, output));

const shutdown = () => {
  for (const { room } of rooms.values()) room.close();
  process.exit(0);
};
process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
process.stdin.on("end", shutdown);
