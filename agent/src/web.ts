#!/usr/bin/env bun
// Web console: a browser front end that is itself an ACP client.
// Each browser tab gets its own agent process. The server holds the ACP
// connection, relays session updates over a WebSocket, answers permission
// requests from the browser, and exposes ACP session management (new, list,
// load with replay, delete) so past war rooms reopen with one click.
//
//   bun src/web.ts            then open http://localhost:4317

import * as acp from "@agentclientprotocol/sdk";
import { spawn } from "node:child_process";
import { Readable, Writable } from "node:stream";
import { dirname, join, normalize, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { mkdirSync, existsSync, readFileSync } from "node:fs";
import type { ServerWebSocket } from "bun";

const here = dirname(fileURLToPath(import.meta.url));
const PORT = Number(process.env.PORT ?? 4317);
const WORK = resolve(here, "..", "..", "workspace");
const OUT = join(WORK, "decision-room-out");
mkdirSync(WORK, { recursive: true });
const PAGE = readFileSync(join(here, "..", "web", "index.html"), "utf8");

type Conn = { handle: (msg: any) => void; close: () => void };
const conns = new WeakMap<ServerWebSocket<unknown>, Conn>();

function startAgent(ws: ServerWebSocket<unknown>): Conn {
  const send = (msg: unknown) => ws.readyState === 1 && ws.send(JSON.stringify(msg));
  const child = spawn(process.execPath, [join(here, "main.ts")], { stdio: ["pipe", "pipe", "inherit"], cwd: WORK });
  const stream = acp.ndJsonStream(Writable.toWeb(child.stdin!), Readable.toWeb(child.stdout!) as ReadableStream<Uint8Array>);
  const permission = new Map<string, (optionId: string) => void>();
  let current = "";
  let busy = false;
  const queue: string[] = [];
  let permId = 0;

  const conn = acp
    .client({ name: "decision-room-web" })
    .onNotification("session/update", (ctx) => {
      if (ctx.params.sessionId === current) send({ type: "update", update: ctx.params.update });
    })
    .onRequest("session/request_permission", async (ctx) => {
      const id = `perm_${++permId}`;
      send({ type: "permission", id, title: ctx.params.toolCall.title, options: ctx.params.options });
      const optionId = await new Promise<string>((r) => permission.set(id, r));
      return { outcome: { outcome: "selected", optionId } };
    })
    .onRequest("fs/read_text_file", async () => ({ content: "" }))
    .onRequest("fs/write_text_file", async () => ({}))
    .connect(stream);
  const agent = conn.agent;

  const fail = (e: unknown) => send({ type: "error", message: String((e as Error)?.message ?? e) });
  const list = async () => {
    const res = await agent.request("session/list", { cwd: WORK });
    send({ type: "sessions", current, sessions: res.sessions });
  };
  const pump = () => {
    if (busy || !queue.length || !current) return;
    const text = queue.shift()!;
    busy = true;
    send({ type: "turn_start", text });
    agent
      .request("session/prompt", { sessionId: current, prompt: [{ type: "text", text }] })
      .then((res) => send({ type: "turn_end", stopReason: res.stopReason }))
      .catch(fail)
      .finally(() => {
        busy = false;
        void list();
        pump();
      });
  };
  const newSession = async () => {
    const res = await agent.request("session/new", { cwd: WORK, mcpServers: [] });
    current = res.sessionId;
    send({ type: "session", sessionId: current, modes: res.modes ?? null, loaded: false });
    await list();
  };

  const ready = (async () => {
    const init = await agent.request("initialize", {
      protocolVersion: acp.PROTOCOL_VERSION,
      clientCapabilities: { fs: { readTextFile: false, writeTextFile: false } },
      clientInfo: { name: "decision-room-web", version: "0.2.0" },
    });
    send({ type: "connected", agent: init.agentInfo, protocolVersion: init.protocolVersion });
    await newSession();
  })().catch(fail);

  return {
    handle: (msg) => {
      void ready.then(async () => {
        try {
          switch (msg.type) {
            case "prompt":
              queue.push(String(msg.text));
              pump();
              break;
            case "new":
              if (!busy) await newSession();
              break;
            case "load":
              if (busy) break;
              current = String(msg.sessionId);
              send({ type: "session", sessionId: current, modes: null, loaded: true });
              await agent.request("session/load", { sessionId: current, cwd: WORK, mcpServers: [] });
              send({ type: "loaded" });
              await list();
              break;
            case "delete":
              await agent.request("session/delete", { sessionId: String(msg.sessionId) });
              if (msg.sessionId === current) await newSession();
              else await list();
              break;
            case "mode":
              await agent.request("session/set_mode", { sessionId: current, modeId: String(msg.modeId) });
              break;
            case "cancel":
              if (busy) await agent.notify("session/cancel", { sessionId: current });
              break;
            case "permission":
              permission.get(msg.id)?.(msg.optionId);
              permission.delete(msg.id);
              break;
          }
        } catch (e) {
          fail(e);
        }
      });
    },
    close: () => {
      conn.close();
      child.kill();
    },
  };
}

function serveFile(path: string): Response {
  const full = normalize(join(OUT, path));
  if (!full.startsWith(OUT) || !existsSync(full)) return new Response("not found", { status: 404 });
  const type = full.endsWith(".svg") ? "image/svg+xml" : full.endsWith(".html") ? "text/html; charset=utf-8" : "text/plain; charset=utf-8";
  return new Response(Bun.file(full), { headers: { "content-type": type } });
}

Bun.serve({
  port: PORT,
  fetch(req, server) {
    const url = new URL(req.url);
    if (url.pathname === "/ws") return server.upgrade(req) ? undefined : new Response("upgrade failed", { status: 400 });
    if (url.pathname.startsWith("/out/")) return serveFile(decodeURIComponent(url.pathname.slice(5)));
    if (url.pathname === "/" || url.pathname === "/index.html") return new Response(PAGE, { headers: { "content-type": "text/html; charset=utf-8" } });
    return new Response("not found", { status: 404 });
  },
  websocket: {
    open(ws) {
      conns.set(ws, startAgent(ws));
    },
    message(ws, raw) {
      conns.get(ws)?.handle(JSON.parse(String(raw)));
    },
    close(ws) {
      conns.get(ws)?.close();
    },
  },
});

console.log(`Decision Room web console: http://localhost:${PORT}`);
