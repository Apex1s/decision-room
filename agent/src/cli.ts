#!/usr/bin/env bun
// A terminal ACP client for Decision Room. It launches the agent exactly as an
// editor would (a subprocess speaking ACP over stdio) and renders the stream.
//
//   bun src/cli.ts            interactive chat
//   bun src/cli.ts --demo     scripted 4-question demo, permissions auto-approved
//   bun src/cli.ts "question" one-shot

import * as acp from "@agentclientprotocol/sdk";
import { spawn } from "node:child_process";
import { Readable, Writable } from "node:stream";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import readline from "node:readline/promises";

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const demo = args.includes("--demo");
const autoApprove = demo || args.includes("--yes");
const oneShot = args.filter((a) => !a.startsWith("--")).join(" ");

const c = {
  dim: (s: string) => `\x1b[2m${s}\x1b[0m`,
  bold: (s: string) => `\x1b[1m${s}\x1b[0m`,
  blue: (s: string) => `\x1b[34m${s}\x1b[0m`,
  green: (s: string) => `\x1b[32m${s}\x1b[0m`,
  yellow: (s: string) => `\x1b[33m${s}\x1b[0m`,
  red: (s: string) => `\x1b[31m${s}\x1b[0m`,
  cyan: (s: string) => `\x1b[36m${s}\x1b[0m`,
};

export const DEMO_SCRIPT = [
  "/brief",
  "What if Dune adds 0.4 Mt of capacity next quarter?",
  "What does it cost us if no region may take more than 22% of our volume?",
  "/status",
];

function md(text: string): string {
  // Light terminal styling for the agent's Markdown (code fences left alone).
  if (text.includes("```")) return text.replace(/^```.*$/gm, "");
  return text
    .replace(/^### (.*)$/gm, (_, t) => c.bold(c.cyan(`▌ ${t}`)))
    .replace(/\*\*(.+?)\*\*/g, (_, t) => c.bold(t))
    .replace(/`([^`]+)`/g, (_, t) => c.yellow(t));
}

const statusIcon: Record<string, string> = { pending: "○", in_progress: "◐", completed: "●", failed: "✗" };

function render(update: any, state: { lastPlan: string }) {
  switch (update.sessionUpdate) {
    case "agent_message_chunk":
      if (update.content.type === "text") process.stdout.write(md(update.content.text));
      break;
    case "plan": {
      const line = update.entries.map((e: any) => (e.status === "completed" ? c.green("●") : e.status === "in_progress" ? c.yellow("◐") : c.dim("○"))).join("");
      if (line !== state.lastPlan) {
        const current = update.entries.find((e: any) => e.status === "in_progress");
        process.stdout.write(`${c.dim("plan")} ${line} ${current ? c.dim(current.content) : ""}\n`);
        state.lastPlan = line;
      }
      break;
    }
    case "tool_call":
      process.stdout.write(`\n${c.blue(statusIcon[update.status] ?? "◐")} ${c.bold(update.title)} ${c.dim(`[${update.kind}]`)}\n`);
      break;
    case "tool_call_update":
      if (update.status === "completed") process.stdout.write(`${c.green("  ✓ done")}\n`);
      else if (update.status === "failed") process.stdout.write(`${c.red("  ✗ failed")}\n`);
      else {
        const text = update.content?.[0]?.content?.text;
        if (text) process.stdout.write(c.dim(`  ${text.split("\n").at(-1)}\n`));
        if (update.content?.[0]?.type === "diff") process.stdout.write(c.dim(`  wrote ${update.content[0].path}\n`));
      }
      break;
    case "available_commands_update":
      process.stdout.write(c.dim(`commands: ${update.availableCommands.map((x: any) => "/" + x.name).join(" ")}\n`));
      break;
    case "current_mode_update":
      process.stdout.write(c.dim(`mode -> ${update.currentModeId}\n`));
      break;
  }
}

async function main() {
  const agent = spawn(process.execPath, [join(here, "main.ts")], { stdio: ["pipe", "pipe", "inherit"] });
  const stream = acp.ndJsonStream(Writable.toWeb(agent.stdin!), Readable.toWeb(agent.stdout!) as ReadableStream<Uint8Array>);
  const rl = demo || oneShot ? null : readline.createInterface({ input: process.stdin, output: process.stdout });
  const state = { lastPlan: "" };

  await acp
    .client({ name: "decision-room-cli" })
    .onRequest("session/request_permission", async (ctx) => {
      const p = ctx.params;
      if (autoApprove) {
        process.stdout.write(c.dim(`  permission: "${p.toolCall.title}" -> approved (--demo)\n`));
        return { outcome: { outcome: "selected", optionId: p.options[0]!.optionId } };
      }
      process.stdout.write(`\n${c.yellow("?")} ${p.toolCall.title}\n`);
      p.options.forEach((o, i) => process.stdout.write(`  ${i + 1}. ${o.name}\n`));
      const ans = Number((await rl!.question("  choose: ")).trim()) - 1;
      const opt = p.options[ans] ?? p.options.at(-1)!;
      return { outcome: { outcome: "selected", optionId: opt.optionId } };
    })
    .onRequest("fs/read_text_file", async () => ({ content: "" }))
    .onRequest("fs/write_text_file", async () => ({}))
    .connectWith(stream, async (ctx) => {
      const init = await ctx.request("initialize", {
        protocolVersion: acp.PROTOCOL_VERSION,
        clientCapabilities: { fs: { readTextFile: false, writeTextFile: false } },
        clientInfo: { name: "decision-room-cli", version: "0.1.0" },
      });
      process.stdout.write(c.dim(`connected to ${init.agentInfo?.title ?? "agent"} over ACP v${init.protocolVersion}\n`));

      return ctx.buildSession(process.cwd()).withSession(async (session) => {
        const ask = async (q: string) => {
          process.stdout.write(`\n${c.bold(c.blue("you ›"))} ${q}\n\n`);
          session.prompt(q);
          for (;;) {
            const m = await session.nextUpdate();
            if (m.kind === "stop") {
              process.stdout.write(c.dim(`\n[${m.response.stopReason}]\n`));
              return;
            }
            render(m.notification.update, state);
          }
        };
        // Let the welcome message and command list arrive.
        await new Promise((r) => setTimeout(r, 150));
        if (demo) for (const q of DEMO_SCRIPT) await ask(q);
        else if (oneShot) await ask(oneShot);
        else {
          for (;;) {
            const q = (await rl!.question(`\n${c.bold("you ›")} `)).trim();
            if (!q || q === "exit" || q === "quit") break;
            await ask(q);
          }
        }
      });
    });
  rl?.close();
  agent.kill();
  process.exit(0);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
