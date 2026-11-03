"""Uncertainty: what-if scenarios, sampled worlds, and Monte Carlo on the decision.

A *world* is one draw of everything we are unsure about: demand parameters
(alpha, sigma, product and region effects, from the estimator's covariance),
next quarter's demand shocks (from their estimated persistence) and market
sizes. Monte Carlo compares the recommendation with the status quo *in the
same worlds*, so the answer is "how often does the recommendation win", not
just a fan chart. Robust optimisation uses the same worlds as its objective.
"""

from __future__ import annotations

import numpy as np

from .demand import DemandModel, _delta_bar
from .equilibrium import Game
from .market import Market
from .strategy import Scenario, build_game, nash, optimise, respond


def sample_worlds(m: Market, model: DemandModel, draws: int, size_sd: float = 0.04, seed: int = 11) -> list[Game]:
    rng = np.random.default_rng(seed)
    names = model.names
    beta_hat = np.array([
        -model.alpha if n == "delivered_price" else model.sigma if n == "ln_within_nest_share" else model.coef.get(n, 0.0)
        for n in names])
    i_p = names.index("delivered_price")
    i_s = names.index("ln_within_nest_share") if "ln_within_nest_share" in names else None
    innov_sd = model.residual_sd * np.sqrt(max(1 - model.xi_rho ** 2, 0.0))
    worlds = []
    for _ in range(draws):
        beta = rng.multivariate_normal(beta_hat, model.cov)
        alpha = max(-beta[i_p], 1e-4)
        sigma = float(np.clip(beta[i_s], 0.0, 0.9)) if i_s is not None else 0.0
        xi = model.xi_rho * model.xi_last + rng.normal(0, innov_sd, model.xi_last.shape)
        game = build_game(m, model, alpha=alpha, delta0=_delta_bar(m, beta, names) + xi, sigma=sigma)
        game.size = game.size * (1 + rng.normal(0, size_sd, game.size.shape))
        worlds.append(game)
    return worlds


def monte_carlo(m: Market, model: DemandModel, recommended: np.ndarray, baseline: np.ndarray,
                draws: int = 200, size_sd: float = 0.04, seed: int = 11, progress=None) -> dict:
    """Profit of two leader markup matrices (n_leader_products, R) across the same sampled worlds."""
    L = m.leader_index
    rec, base, uplift = [], [], []
    worlds = sample_worlds(m, model, draws, size_sd, seed)
    for i, game in enumerate(worlds):
        p_rec = respond(game, L, recommended).profit[L]
        p_base = respond(game, L, baseline).profit[L]
        rec.append(p_rec); base.append(p_base); uplift.append(p_rec - p_base)
        if progress and (i + 1) % max(draws // 10, 1) == 0:
            progress(f"Monte Carlo {i + 1}/{draws} draws")
    rec, base, uplift = map(np.array, (rec, base, uplift))

    def pct(a):
        return {k: round(float(np.percentile(a, q)), 3) for k, q in (("p10", 10), ("p50", 50), ("p90", 90))}

    return {
        "draws": draws,
        "recommended_profit": pct(rec),
        "status_quo_profit": pct(base),
        "uplift": pct(uplift),
        "prob_recommended_wins_pct": round(100 * float(np.mean(uplift > 0)), 1),
        "expected_uplift_musd": round(float(uplift.mean()), 3),
        "worst_case_uplift_musd": round(float(uplift.min()), 3),
        "histogram": _hist(uplift),
    }


def _hist(a: np.ndarray, bins: int = 24) -> dict:
    counts, edges = np.histogram(a, bins=bins)
    return {"counts": counts.tolist(), "edges": [round(float(e), 4) for e in edges]}


def whatif(m: Market, model: DemandModel, scenario: Scenario) -> dict:
    """Compare today's world with the scenario, with and without reacting."""
    L = m.leader_index
    g0 = build_game(m, model)
    g1 = build_game(m, model, scenario)
    rec0, _ = optimise(g0, L)
    rec1, info = optimise(g1, L, start=rec0.shift)
    stay = respond(g1, L, rec0.leader_markup, certify=True)  # world changes, we do not
    n0, n1 = nash(g0, L), nash(g1, L)
    v0 = rec0.leader_view(m, g0)
    v1 = rec1.leader_view(m, g1)
    return {
        "scenario": scenario.to_dict(),
        "before": v0,
        "after_reacting": v1,
        "after_not_reacting_profit_musd": round(float(stay.profit[L]), 3),
        "value_of_reacting_musd": round(float(rec1.eq.profit[L] - stay.profit[L]), 3),
        "impact_musd": round(float(rec1.eq.profit[L] - rec0.eq.profit[L]), 3),
        "price_moves": {r: round(v1["by_region"][r]["price"] - v0["by_region"][r]["price"], 2) for r in m.regions},
        "rival_shadow_prices_after": {m.firms[f]: round(float(n1.eq.shadow[f]), 2) for f in range(m.F) if f != L},
        "rival_shadow_prices_before": {m.firms[f]: round(float(n0.eq.shadow[f]), 2) for f in range(m.F) if f != L},
        "rival_profit_change_musd": {m.firms[f]: round(float(rec1.eq.profit[f] - rec0.eq.profit[f]), 3)
                                     for f in range(m.F) if f != L},
        "optimiser": info,
    }
