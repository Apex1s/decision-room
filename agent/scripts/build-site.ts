#!/usr/bin/env bun
// Builds the static demo site (GitHub Pages): records a real ACP session on the
// synthetic demo market, then writes the web console in replay mode next to the
// recording and the board memo it produced.
//
//   bun scripts/build-site.ts            writes ../site
//
// The recording is genuine agent output: this script is an ACP client like Zed,
// it only stores what the agent streams and replays it with the original pacing.

import * as acp from "@agentclientprotocol/sdk";
import { spawn } from "node:child_process";
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { Readable, Writable } from "node:stream";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const agentDir = join(here, "..");
const site = join(agentDir, "..", "site");

export const SCRIPT = [
  "/brief",
  "What if Dune adds 0.4 Mt of capacity?",
  "What does capping any region at 22% of our volume cost us?",
  "Are the rivals really competing, or is this a cartel?",
];

type Event = { t: number; m: unknown };

async function record(work: string): Promise<{ agent: unknown; protocolVersion: number; turns: { prompt: string; events: Event[] }[] }> {
  const env = { ...process.env, DECISION_ROOM_HOME: join(work, "home"), DECISION_ROOM_OFFLINE: "1" };
  const child = spawn(process.execPath, [join(agentDir, "src", "main.ts")], { stdio: ["pipe", "pipe", "inherit"], env });
  const stream = acp.ndJsonStream(Writable.toWeb(child.stdin!), Readable.toWeb(child.stdout!) as ReadableStream<Uint8Array>);
  const turns: { prompt: string; events: Event[] }[] = [];
  let out: Awaited<ReturnType<typeof record>> | null = null;

  await acp
    .client({ name: "decision-room-site" })
    .onRequest("session/request_permission", async (ctx) => {
      const turn = turns.at(-1);
      turn?.events.push({ t: 0, m: { type: "permission_auto", title: ctx.params.toolCall.title } });
      return { outcome: { outcome: "selected", optionId: ctx.params.options[0]!.optionId } };
    })
    .onRequest("fs/read_text_file", async () => ({ content: "" }))
    .onRequest("fs/write_text_file", async () => ({}))
    .connectWith(stream, async (ctx) => {
      // Named like the web console so the agent attaches its SVG charts.
      const init = await ctx.request("initialize", {
        protocolVersion: acp.PROTOCOL_VERSION,
        clientCapabilities: { fs: { readTextFile: false, writeTextFile: false } },
        clientInfo: { name: "decision-room-web", version: "0.2.0" },
      });
      await ctx.buildSession(work).withSession(async (session) => {
        await new Promise((r) => setTimeout(r, 200));
        for (const prompt of SCRIPT) {
          process.stdout.write(`recording: ${prompt}\n`);
          const turn = { prompt, events: [] as Event[] };
          turns.push(turn);
          const t0 = Date.now();
          session.prompt(prompt);
          for (;;) {
            const msg = await session.nextUpdate();
            if (msg.kind === "stop") {
              turn.events.push({ t: Date.now() - t0, m: { type: "turn_end", stopReason: msg.response.stopReason } });
              break;
            }
            const u: any = msg.notification.update;
            // Commands and the welcome line belong to the session, not to the turn.
            if (u.sessionUpdate === "available_commands_update") continue;
            turn.events.push({ t: Date.now() - t0, m: { type: "update", update: u } });
          }
        }
      });
      out = { agent: init.agentInfo, protocolVersion: init.protocolVersion, turns };
    });
  child.kill();
  return out!;
}

function scrub(json: string, work: string): string {
  // Paths of the temporary working folder become a neutral relative path.
  const variants = [work, work.replaceAll("\\", "/"), JSON.stringify(work).slice(1, -1)];
  for (const v of variants) json = json.split(v).join(".");
  for (const leak of ["Users\\\\", "Users/", "AppData"]) {
    if (json.includes(leak)) throw new Error(`recording still contains a local path (${leak}); refusing to publish it`);
  }
  return json;
}

async function main() {
  const work = mkdtempSync(join(tmpdir(), "dr-site-"));
  try {
    const rec = await record(work);
    rmSync(site, { recursive: true, force: true });
    mkdirSync(join(site, "memo"), { recursive: true });
    writeFileSync(join(site, "replay.json"), scrub(JSON.stringify({ recorded: new Date().toISOString().slice(0, 10), ...rec }), work));
    cpSync(join(work, "decision-room-out"), join(site, "memo"), { recursive: true });
    const html = readFileSync(join(agentDir, "web", "index.html"), "utf8").replace('<html lang="en">', '<html lang="en" data-replay="replay.json">');
    writeFileSync(join(site, "index.html"), html);
    writeFileSync(join(site, ".nojekyll"), "");
    const n = rec.turns.reduce((a, t) => a + t.events.length, 0);
    process.stdout.write(`site written to ${site}: ${rec.turns.length} turns, ${n} events\n`);
  } finally {
    rmSync(work, { recursive: true, force: true });
  }
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
