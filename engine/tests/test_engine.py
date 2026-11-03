"""Engine tests: every stage is checked against the synthetic market's ground truth."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from decision_room import demand, forecast, redteam, risk, strategy
from decision_room.equilibrium import Game, implied_costs, solve, uniqueness_check
from decision_room.market import load_market, save_market
from decision_room.synth import generate
from decision_room.workspace import Workspace

ENGINE = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def market():
    return generate()


@pytest.fixture(scope="module")
def model(market):
    return demand.fit(market[0], nested=True)


def small_game(capacity, sigma=0.0):
    return Game(delta0=np.full((4, 2), 1.5), alpha=0.03, freight=np.full((4, 2), 10.0),
                cost=np.array([50.0, 60.0, 55.0, 45.0]), product_firm=np.array([0, 0, 1, 2]),
                capacity=np.array(capacity, float), size=np.array([2.0, 1.5]), avail=np.ones((4, 2), bool),
                sigma=sigma, nest=np.array([0, 1, 0, 1]))


# ---- equilibrium ------------------------------------------------------------

@pytest.mark.parametrize("sigma", [0.0, 0.4, 0.8])
def test_equilibrium_satisfies_kkt_and_capacity(sigma):
    g = small_game([10.0, 10.0, 0.25], sigma)
    eq = solve(g)
    assert eq.kkt["certified"]
    assert eq.firm_volume[2] <= 0.25 * (1 + 1e-6)
    assert eq.shadow[2] > 0  # the tight firm carries a shadow price
    assert eq.shadow[0] == pytest.approx(0.0, abs=1e-9)
    assert uniqueness_check(g, eq)["unique"]


def test_logit_markup_matches_closed_form():
    g = small_game([99.0, 99.0, 99.0])
    eq = solve(g)
    S = g.onehot() @ eq.shares
    assert np.allclose(eq.prices, g.cost[:, None] + 1 / (g.alpha * (1 - S[g.product_firm])), atol=1e-8)


def test_nested_equilibrium_is_a_profit_maximum():
    g = small_game([99.0, 99.0, 99.0], sigma=0.5)
    eq = solve(g)
    base = g.firm_profit(eq.prices)
    for j in range(4):
        for r in range(2):
            p = eq.prices.copy()
            p[j, r] += 1e-4
            f = g.product_firm[j]
            assert abs((g.firm_profit(p)[f] - base[f]) / 1e-4) < 1e-4


def test_implied_costs_invert_the_equilibrium():
    g = small_game([99.0, 99.0, 0.25], sigma=0.4)
    eq = solve(g)
    expected = g.cost[:, None] + eq.shadow[g.product_firm][:, None]
    assert np.allclose(implied_costs(g, eq.prices), expected, atol=1e-8)


# ---- demand -----------------------------------------------------------------

@pytest.mark.parametrize("seed", [2026, 1, 2])
def test_nested_iv_recovers_truth(seed):
    m, truth = generate(seed=seed)
    d = demand.fit(m, nested=True)
    assert d.alpha - 1.96 * d.alpha_se <= truth.alpha <= d.alpha + 1.96 * d.alpha_se
    assert d.sigma - 1.96 * d.sigma_se <= truth.sigma <= d.sigma + 1.96 * d.sigma_se
    assert d.first_stage_f > 10 and d.first_stage_f_nest > 10


def test_plain_logit_is_misspecified_and_ols_is_biased(market, model):
    plain = demand.fit(market[0], nested=False)
    assert plain.alpha > 1.3 * market[1].alpha  # ignoring nesting overstates price sensitivity
    assert model.ols_alpha < model.alpha  # endogenous prices bias OLS towards zero


def test_iv_recovers_grade_premia(market, model):
    g = dict(market[1].grade)
    assert model.coef["grade:Mid"] == pytest.approx(g["Mid"], abs=0.08)
    assert model.coef["grade:High"] == pytest.approx(g["High"], abs=0.12)


def test_equilibrium_forecast_beats_naive_and_logit(market):
    nested = forecast.backtest(market[0], windows=(24,), test_periods=6, nested=True)["best"]
    plain = forecast.backtest(market[0], windows=(24,), test_periods=6, nested=False)["best"]
    assert nested["skill_vs_naive_pct"] > 0 and nested["price_skill_vs_naive_pct"] > 0
    assert nested["mae_share_pts"] < plain["mae_share_pts"]


def test_workspace_selects_nested_logit():
    ws = Workspace()
    ws.load("demo")
    bt = ws.backtest()
    assert bt["selected_model"] == "nested" and ws.model.kind == "nested"
    assert ws.rival_model.kind == "logit"


# ---- strategy -----------------------------------------------------------------

def test_leader_does_at_least_as_well_as_nash(market, model):
    m = market[0]
    game = strategy.build_game(m, model)
    nash = strategy.nash(game, m.leader_index)
    rec, info = strategy.optimise(game, m.leader_index)
    assert info["feasible"] and rec.eq.kkt["certified"]
    assert rec.eq.profit[m.leader_index] >= nash.eq.profit[m.leader_index] - 1e-6


def test_tighter_rule_never_raises_profit(market, model):
    m = market[0]
    game = strategy.build_game(m, model)
    fr = strategy.frontier(game, m.leader_index, caps=[0.30, 0.24, 0.20])
    profits = [p["profit_musd"] for p in fr["points"]]
    assert profits == sorted(profits, reverse=True)
    for p in fr["points"]:
        assert p["top_region_share_pct"] <= p["cap_pct"] + 0.1


def test_robust_plan_wins_on_its_own_worlds(market, model):
    m = market[0]
    L = m.leader_index
    game = strategy.build_game(m, model)
    worlds = risk.sample_worlds(m, model, 6, seed=5)
    nominal, _ = strategy.optimise(game, L)
    robust, info = strategy.robust_optimise(game, worlds, L)
    mean = lambda mk: np.mean([strategy.respond(w, L, mk).profit[L] for w in worlds])  # noqa: E731
    assert info["feasible"]
    assert mean(robust.leader_markup) >= mean(nominal.leader_markup) - 1e-6


def test_scenario_capacity_expansion_hurts_leader(market, model):
    out = risk.whatif(market[0], model, strategy.Scenario("Dune expands", capacity_change={"Dune": 0.4}))
    assert out["impact_musd"] < 0
    assert out["value_of_reacting_musd"] >= -1e-6


def test_conduct_test_finds_bertrand(market, model):
    game = strategy.build_game(market[0], model)
    ct = redteam.conduct_test(market[0], model, solve(game))
    assert ct["best"] == "bertrand"
    assert ct["median_abs_gap_pct"]["bertrand"] < 5 < ct["median_abs_gap_pct"]["collusion"]


# ---- data and protocol ----------------------------------------------------------

def test_csv_round_trip(tmp_path, market, model):
    m = market[0]
    save_market(m, tmp_path)
    assert (tmp_path / "capacity.csv").exists()
    back = load_market(tmp_path)
    assert back.firms == m.firms and back.regions == m.regions and back.leader == m.leader
    assert np.allclose(back.capacity_at(10), m.capacity_at(10), atol=1e-4)
    assert demand.fit(back, nested=True).alpha == pytest.approx(model.alpha, rel=1e-3)


def test_json_rpc_server_round_trip():
    proc = subprocess.Popen([sys.executable, "-m", "decision_room", "serve"], cwd=ENGINE, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, encoding="utf-8")
    try:
        ready = json.loads(proc.stdout.readline())
        assert ready["method"] == "ready" and "recommend" in ready["params"]["methods"]
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "load", "params": {"source": "demo"}}) + "\n")
        proc.stdin.flush()
        while True:
            msg = json.loads(proc.stdout.readline())
            if msg.get("id") == 1:
                break
        assert msg["result"]["leader"] == "Atlas"
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "nope"}) + "\n")
        proc.stdin.flush()
        msg = json.loads(proc.stdout.readline())
        assert msg["error"]["code"] == -32601
    finally:
        proc.kill()


def test_settings_and_extra_instruments_round_trip(tmp_path):
    """settings.json (year effects, implied costs, annual wording) and instruments.csv load, fit and export."""
    import json

    from decision_room import demand as dm
    from decision_room.market import load_market, save_market
    from decision_room.synth import generate
    from decision_room.units import relabel

    m, _ = generate(periods=16)
    save_market(m, tmp_path / "a")
    (tmp_path / "a" / "settings.json").write_text(
        json.dumps({"time_effects": True, "costs": "implied", "period_unit": "year"}), encoding="utf-8")
    rng = np.random.default_rng(0)
    with (tmp_path / "a" / "instruments.csv").open("w", encoding="utf-8") as fh:
        fh.write("period,product,z1\n")
        for t in m.periods:
            for p in m.products:
                fh.write(f"{int(t)},{p},{rng.normal():.5f}\n")
    a = load_market(tmp_path / "a")
    assert a.time_effects and a.costs_source == "implied" and a.period_unit == "year"
    assert a.obs_instruments.shape == (len(a.obs_price), 1)
    model = dm.fit(a)
    assert any(n.startswith("period:") for n in model.names)
    assert model.alpha > 0
    save_market(a, tmp_path / "b")
    b = load_market(tmp_path / "b")
    assert b.time_effects and b.costs_source == "implied" and b.period_unit == "year"
    assert np.allclose(b.obs_instruments, a.obs_instruments, atol=1e-5)
    assert relabel("$1.0M per quarter, 8q window, 2 Mt/q", "year") == "$1.0M per year, 8y window, 2 Mt/yr"
    assert relabel("per quarter", "quarter") == "per quarter"
