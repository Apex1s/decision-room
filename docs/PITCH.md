# Decision Room: the pitch

## One line

The pricing war room for producers in concentrated markets: it models how rivals will answer your price, tells you what your own commercial rules cost, and shows you when not to trust it, inside the tools analysts already use.

## The problem

In concentrated industries (fertilisers, cement, steel, aluminium, chemicals, LNG, dry bulk), a handful of producers set prices knowing the others will respond. Pricing committees still decide with:

- **Spreadsheets** that treat market share as an input, so every "optimal" price assumes rivals stand still.
- **Consulting studies** that are excellent but static: a scenario answered in six weeks is stale by the time it arrives.
- **Generic AI copilots** that write fluent answers with numbers no one can trace.

The result is decisions worth tens of millions per quarter made on models that ignore competition, with no measure of how sure anyone should be.

## The product

Decision Room is an AI agent that speaks the **Agent Client Protocol (ACP)**, the open standard (created by Zed, co-developed with JetBrains) that lets one agent run inside any compatible editor. It also speaks **MCP**, so other agents (Claude Code, Claude Desktop, in-house agents) can call it as a tool. An analyst asks a question in plain English, and seven specialists answer it with real economics:

1. **Data Steward** loads and checks the market panel (bring your own CSVs).
2. **Demand Economist** estimates demand with Berry inversion and instruments, chooses between plain and nested logit on evidence, and proves the model forecasts next quarter's prices and shares better than a naive benchmark.
3. **Game Theorist** solves the capacity-constrained Bertrand-Nash equilibrium, certifies it (KKT conditions, uniqueness from random restarts), and tests whether rivals really compete or behave like a cartel.
4. **Strategist** sets our prices by region and grade as a Stackelberg leader, prices every commercial-security rule ("no client region above 22% of volume costs $1.25M per quarter"), and can optimise robustly over uncertainty instead of betting on one forecast.
5. **Risk Analyst** runs what-ifs (a rival's capacity expansion, a demand shock, a freight spike) and a Monte Carlo on the decision itself.
6. **Red Team** attacks the recommendation with 13 checks and gives a mechanical A–D grade.
7. **Memo Writer** produces the board memo: Markdown for the team, HTML with charts for the committee.

Every figure is computed by the engine. The language model, when enabled, only routes questions and phrases the summary.

## Why now

- **ACP and MCP make distribution cheap.** One agent runs in Zed, JetBrains IDEs and any ACP client with no plugin per editor, and the same engine plugs into any MCP-capable agent as a tool. Agents are becoming the interface to analytical software; these two standards put Decision Room wherever analysts and their agents already work.
- **Trust is the bottleneck for AI in high-stakes decisions.** Executives will not act on numbers a model made up. An agent whose numbers are computed, validated and red-teamed is the version they can sign off on.
- **Commodity volatility since 2021** has made quarterly repricing in concentrated markets a board-level topic.

## What makes it defensible

- **Domain depth:** structural demand estimation (nested logit, instruments), equilibrium with capacity under any conduct, conduct testing, leader-follower and robust optimisation, constraint pricing. Generic analytics tools do not do this, and building it takes economists, not prompt engineers.
- **Honesty as a feature:** the red team, the out-of-sample backtest and the "what this model cannot tell you" section turn a model into something a committee can defend.
- **Data gravity:** once a client's market panel, cost curves and rules live in Decision Room, every new question is minutes away.

## Who buys

| Segment | Buyer | Job to be done |
|---|---|---|
| Producers in oligopolies | Head of pricing, commercial director, strategy | Quarterly price setting, allocation by region, contract rules |
| Commodity traders | Desk heads | Read rivals' capacity moves before the market does |
| Strategy consultancies | Partners, analytics practices | Deliver competitive-response studies in days, not weeks |
| Development banks and regulators | Market analysis units | Assess market power and concentration effects |

## Business model (hypothesis to validate)

- **Team licence** per decision team (pricing committee plus analysts), annual.
- **Onboarding** fixed fee: connecting the client's data and calibrating the market model.
- **Private deployment** for clients who cannot send data out: the engine runs on their machines and the agent works fully offline already.

Pricing, conversion and market size are to be validated with design partners; this document makes no revenue claims.

## Traction and proof today

- A working ACP agent (with persistent sessions you can list, reopen and delete), MCP server, terminal client and web console, with 30 automated tests and CI. The economics are checked against known ground truth; the protocols are tested end to end with the official ACP and MCP SDK clients.
- On a synthetic oligopoly with known parameters:
  - True price sensitivity and nesting are recovered inside their 95% CIs on 5 of 5 independent markets, while plain logit is 68% off and OLS 50% off.
  - The equilibrium forecast beats "same as last quarter" by 14% on shares and 32% on prices.
  - The conduct test correctly finds competition, not collusion.
  - The recommendation adds 5.3% profit and wins in 99% of simulated futures.
  - The red team still flags honestly that some recommended prices lie outside the historical range (grade B).
- Built entirely on published methods and a synthetic market; no third-party or company data is included.

## Next 90 days

1. Two design partners in fertilisers or building materials, running their own data through `/market`.
2. Multi-period contracts and dynamic capacity decisions.
3. Shared workspaces: one market model, many analysts, each in their own editor over ACP or agent over MCP.
4. A security review for private deployment.

## Team

**Anas Sabbahi**: industrial management engineer (EMINES, UM6P), data science track. Background in commodity pricing and game-theoretic decision support, an ML early-warning system for landslides (32 of 33 events caught on held-out data), and open-source decision tools in optimisation and forecasting ([github.com/Apex1s](https://github.com/Apex1s)).
