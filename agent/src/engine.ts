// Bridge to the Python decision engine: one stateful process per ACP session,
// newline-delimited JSON-RPC 2.0 over its stdin/stdout.

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

export const ENGINE_DIR = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "engine");

type Pending = {
  resolve: (v: any) => void;
  reject: (e: Error) => void;
  onProgress?: (msg: string) => void;
};

export class EngineError extends Error {}

export class Engine {
  private proc: ChildProcessWithoutNullStreams;
  private nextId = 1;
  private pending = new Map<number, Pending>();
  private buffer = "";
  private ready: Promise<string[]>;
  private stderrTail: string[] = [];

  constructor(python = process.env.DECISION_ROOM_PYTHON ?? (process.platform === "win32" ? "python" : "python3")) {
    this.proc = spawn(python, ["-m", "decision_room", "serve"], {
      cwd: ENGINE_DIR,
      env: { ...process.env, PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" },
      stdio: ["pipe", "pipe", "pipe"],
      windowsHide: true,
    });
    let markReady: (m: string[]) => void;
    let failReady: (e: Error) => void;
    this.ready = new Promise((res, rej) => {
      markReady = res;
      failReady = rej;
    });
    this.proc.stdout.setEncoding("utf8");
    this.proc.stdout.on("data", (chunk: string) => {
      this.buffer += chunk;
      let nl: number;
      while ((nl = this.buffer.indexOf("\n")) >= 0) {
        const line = this.buffer.slice(0, nl).trim();
        this.buffer = this.buffer.slice(nl + 1);
        if (!line) continue;
        let msg: any;
        try {
          msg = JSON.parse(line);
        } catch {
          continue; // never let a stray print break the protocol
        }
        if (msg.method === "ready") markReady(msg.params.methods);
        else if (msg.method === "progress") this.pending.get(msg.params.id)?.onProgress?.(msg.params.message);
        else if (msg.id != null && this.pending.has(msg.id)) {
          const p = this.pending.get(msg.id)!;
          this.pending.delete(msg.id);
          if (msg.error) p.reject(new EngineError(msg.error.message));
          else p.resolve(msg.result);
        }
      }
    });
    this.proc.stderr.setEncoding("utf8");
    this.proc.stderr.on("data", (chunk: string) => {
      this.stderrTail.push(...chunk.split("\n").filter(Boolean));
      this.stderrTail = this.stderrTail.slice(-30);
    });
    const die = (why: string) => {
      const err = new EngineError(`${why}\n${this.stderrTail.join("\n")}`);
      failReady(err);
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
    };
    this.proc.on("error", (e) => die(`could not start the Python engine (${python}): ${e.message}`));
    this.proc.on("exit", (code) => die(`the Python engine exited (code ${code})`));
  }

  methods(): Promise<string[]> {
    return this.ready;
  }

  async call<T = any>(method: string, params: Record<string, unknown> = {}, onProgress?: (msg: string) => void): Promise<T> {
    await this.ready;
    const id = this.nextId++;
    return new Promise<T>((resolve, reject) => {
      this.pending.set(id, { resolve, reject, onProgress });
      this.proc.stdin.write(JSON.stringify({ jsonrpc: "2.0", id, method, params }) + "\n");
    });
  }

  close(): void {
    try {
      this.proc.stdin.write(JSON.stringify({ jsonrpc: "2.0", id: 0, method: "shutdown" }) + "\n");
    } catch {}
    setTimeout(() => this.proc.kill(), 500).unref?.();
  }
}
