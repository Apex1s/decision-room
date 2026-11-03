# Using Decision Room from editors and agents

Decision Room ships two entry points over the same engine:

- `agent/src/main.ts`: an **ACP agent**. Editors launch it and you talk to it.
- `agent/src/mcp.ts`: an **MCP server**. Other agents call its tools.

Both are started with Bun. Replace the paths below with yours, as absolute paths.

## Zed (ACP)

In the Agent Panel, open **Agent Settings → External Agents → Add Agent → Add Custom Agent**, or edit `settings.json` directly:

```json
{
  "agent_servers": {
    "Decision Room": {
      "type": "custom",
      "command": "C:/path/to/bun.exe",
      "args": ["C:/path/to/decision-room/agent/src/main.ts"],
      "env": { "DECISION_ROOM_PYTHON": "python" }
    }
  }
}
```

Then:

1. Open the Agent Panel, start a new thread and pick **Decision Room**.
2. Type `/` to see the commands.
3. To analyse your own market, drag a folder with your CSV files into the prompt.
4. Past rooms appear in the thread history and reopen with their full conversation (`session/list`, `session/load`).

## JetBrains IDEs (ACP)

JetBrains IDEs with the AI Assistant support external ACP agents. Add a custom agent with the same command and arguments as above (AI Assistant settings, ACP agents). The exact menu path depends on the IDE version.

## Claude Code (MCP)

```bash
claude mcp add decision-room -- bun /absolute/path/to/decision-room/agent/src/mcp.ts
```

Then ask Claude Code something like *"use decision-room to price a 22% cap on any one region"*. The tools are:

| Tool | What it does |
|---|---|
| `load_market` | Load a CSV market folder, or `demo` |
| `validate_demand` | Select plain vs nested logit by out-of-sample equilibrium forecast |
| `estimate_demand` | Berry/2SLS demand: price sensitivity, nesting, elasticities |
| `solve_equilibrium` | Capacity-constrained Bertrand-Nash with KKT certificate |
| `test_conduct` | Bertrand vs collusion vs price = cost |
| `recommend_prices` | Stackelberg prices by region and grade; rules; robust mode |
| `price_rule` | What a commercial rule costs |
| `concentration_frontier` | Profit vs concentration cap |
| `what_if` | Scenario, with and without our reaction |
| `stress_test` | Monte Carlo on the decision |
| `red_team` | 13 checks and an A–D grade |
| `write_memo` | Board memo (Markdown, HTML, SVG) |
| `status` | What is computed so far |

The resource `decision-room://methodology` serves the methodology document.

## Claude Desktop (MCP)

In `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "decision-room": { "command": "C:/path/to/bun.exe", "args": ["C:/path/to/decision-room/agent/src/mcp.ts"] }
  }
}
```

## Any other client

| Entry point | Command | Protocol |
|---|---|---|
| ACP agent | `bun <path>/agent/src/main.ts` | ACP v1 over stdio |
| MCP server | `bun <path>/agent/src/mcp.ts` | MCP over stdio |

Neither requires authentication.

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `DECISION_ROOM_PYTHON` | `python` (Windows) / `python3` | Python used to run the engine |
| `DECISION_ROOM_HOME` | `~/.decision-room` | Where ACP sessions (history and engine snapshots) are stored |
| `ANTHROPIC_API_KEY` | unset | Enables Claude for routing free-form questions and writing the summary (ACP agent) |
| `DECISION_ROOM_MODEL` | `claude-opus-5-5` | Claude model used when the key is set |
| `DECISION_ROOM_OFFLINE` | unset | Set to `1` to ignore the API key |

## Without an editor

- `bun run web`: browser console on http://localhost:4317, with a rooms panel to reopen past sessions
- `bun run chat`: terminal client (the same ACP protocol)
- `bun run demo`: scripted demo
