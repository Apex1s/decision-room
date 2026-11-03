"""Quick end-to-end smoke run of the engine: data -> demand -> strategy -> risk -> red team."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from decision_room import demand, redteam, risk, strategy as S  # noqa: E402
from decision_room.synth import generate  # noqa: E402


def step(name, fn):
    t0 = time.time()
    out = fn()
    print(f"[{time.time() - t0:5.1f}s] {name}")
    return out


m, truth = step("generate synthetic market", generate)
model = step("fit demand (2SLS)", lambda: demand.fit(m))
print(f"        alpha={model.alpha:.4f} (truth {truth.alpha}), OLS={model.ols_alpha:.4f}")
bt = step("backtest windows", lambda: demand.backtest(m))
game = S.build_game(m, model)
L = m.leader_index
nash = step("Nash equilibrium", lambda: S.nash(game, L))
sq = step("status quo", lambda: S.status_quo(m, game))
rec, info = step("Stackelberg optimise", lambda: S.optimise(game, L))
print(f"        profit: status quo {sq.eq.profit[L]:.2f}  nash {nash.eq.profit[L]:.2f}  recommended {rec.eq.profit[L]:.2f}")
fr = step("frontier", lambda: S.frontier(game, L))
for p in fr["points"]:
    print(f"        cap {p['cap_pct']:5.1f}%  profit {p['profit_musd']:8.3f}  give-up {p['give_up_pct']:5.2f}%")
mc = step("Monte Carlo (100 draws)", lambda: risk.monte_carlo(m, model, rec.leader_markup, sq.leader_markup, draws=100))
print(f"        win prob {mc['prob_recommended_wins_pct']}%  uplift {mc['uplift']}")
wi = step("what-if: Dune +0.4 Mt", lambda: risk.whatif(m, model, S.Scenario("Dune expands", capacity_change={"Dune": 0.4})))
print(f"        impact {wi['impact_musd']}  value of reacting {wi['value_of_reacting_musd']}")
rt = step("red team", lambda: redteam.run(m, model, bt, nash.eq, rec, mc, truth.as_dict()))
print(f"        grade {rt['grade']}: {rt['verdict']}")
for c in rt["checks"]:
    print(f"        [{c['status']:4}] {c['check']}: {c['evidence']}")
