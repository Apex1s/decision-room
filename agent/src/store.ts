// Persistent sessions: everything an ACP client needs to reopen a war room.
// Each session is a folder with its metadata, the full update history (replayed
// on session/load) and a snapshot of the engine's workspace (so a reloaded room
// does not have to recompute anything).

import { appendFileSync, existsSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export const HOME = process.env.DECISION_ROOM_HOME ?? join(homedir(), ".decision-room");
const SESSIONS = join(HOME, "sessions");

export type Meta = { sessionId: string; cwd: string; title: string | null; createdAt: string; updatedAt: string };
export type Entry = { user: any[] } | { update: any };

export class SessionStore {
  readonly dir: string;

  constructor(readonly sessionId: string) {
    if (!/^[\w-]+$/.test(sessionId)) throw new Error(`invalid session id ${sessionId}`);
    this.dir = join(SESSIONS, sessionId);
  }

  static create(sessionId: string, cwd: string): SessionStore {
    const s = new SessionStore(sessionId);
    mkdirSync(s.dir, { recursive: true });
    const now = new Date().toISOString();
    s.writeMeta({ sessionId, cwd, title: null, createdAt: now, updatedAt: now });
    writeFileSync(join(s.dir, "history.jsonl"), "");
    return s;
  }

  static open(sessionId: string): SessionStore | null {
    const s = new SessionStore(sessionId);
    return existsSync(join(s.dir, "meta.json")) ? s : null;
  }

  static list(cwd?: string): Meta[] {
    if (!existsSync(SESSIONS)) return [];
    return readdirSync(SESSIONS)
      .map((id) => SessionStore.open(id)?.meta())
      .filter((m): m is Meta => Boolean(m) && (!cwd || m!.cwd === cwd))
      .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  }

  meta(): Meta {
    return JSON.parse(readFileSync(join(this.dir, "meta.json"), "utf8"));
  }

  private writeMeta(m: Meta) {
    writeFileSync(join(this.dir, "meta.json"), JSON.stringify(m, null, 2));
  }

  append(entry: Entry) {
    appendFileSync(join(this.dir, "history.jsonl"), JSON.stringify(entry) + "\n");
  }

  /** Records a user prompt and titles the session after its first question. */
  recordPrompt(blocks: any[]) {
    this.append({ user: blocks });
    const m = this.meta();
    m.updatedAt = new Date().toISOString();
    if (!m.title) {
      const text = blocks.filter((b) => b.type === "text").map((b) => b.text).join(" ").trim();
      if (text) m.title = text.length > 60 ? `${text.slice(0, 57)}…` : text;
    }
    this.writeMeta(m);
  }

  history(): Entry[] {
    const raw = readFileSync(join(this.dir, "history.jsonl"), "utf8");
    return raw.split("\n").filter(Boolean).map((l) => JSON.parse(l));
  }

  get statePath(): string {
    return join(this.dir, "engine.pkl");
  }

  hasState(): boolean {
    return existsSync(this.statePath);
  }

  delete() {
    rmSync(this.dir, { recursive: true, force: true });
  }
}
