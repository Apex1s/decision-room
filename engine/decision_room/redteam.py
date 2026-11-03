"""The red team: every recommendation is attacked before anyone sees it.

Each check returns pass / warn / fail with the evidence. The grade is
mechanical (no check can be talked out of a failure), and the memo always
ships with the list of things the model cannot tell you.
"""

from __future__ import annotations

import numpy as np

from . import demand as dm
from .equilibrium import elasticities, implied_costs, uniqueness_check
from .market import Market
from .strategy import build_game, leader_prices

LIMITS = [
    "Substitution follows the nest structure (grades). Buyers who would switch region rather than grade, "
    "or long-term offtake contracts, are not represented.",
    "The game is static, one quarter at a time. Long-term contracts, retaliation and reputation are not modelled.",
    "Marginal costs and capacities are taken from the input files. The engine checks them against prices "
    "but cannot verify them.",
    "The conduct test compares three textbook behaviours. Partial collusion or tacit regional splits sit "
    "between them and would need a richer test.",
]

CONDUCTS = {"bertrand": "Bertrand competition", "collusion": "joint profit maximisation (collusion)",
            "competitive": "perfect competition (price = cost)"}


def _check(name: str, status: str, evidence: str, detail: dict | None = None) -> dict:
    return {"check": name, "status": status, "evidence": evidence, "detail": detail or {}}


def conduct_test(m: Market, model: dm.DemandModel, nash_eq) -> dict:
    """Which market behaviour makes observed prices consistent with stated costs?

    For each conduct, invert last quarter's prices into the marginal costs that
    would make them optimal, and compare with stated costs. Firms at capacity are
    excluded: their implied costs include a shadow price the test cannot separate.
    """
    game = build_game(m, model)
    sl = m.period_slice(int(m.periods.max()))
    prices = np.nan_to_num(sl["price"])
    stated = (m.cost * np.nan_to_num(sl["cost_index"], nan=1.0))[:, None] * np.ones((1, m.R))
    slack = nash_eq.shadow < 0.01
    rows = slack[m.product_firm][:, None] & game.avail
    out = {}
    for k in CONDUCTS:
        ic = implied_costs(game, prices, k)
        gap = np.abs(ic - stated)[rows] / stated[rows]
        out[k] = round(float(np.median(gap)) * 100, 2)
    best = min(out, key=out.get)
    return {"median_abs_gap_pct": out, "best": best, "firms_used": [f for f, s in zip(m.firms, slack) if s]}


def run(m: Market, model: dm.DemandModel, backtest: dict | None, nash_eq, rec_outcome,
        mc: dict | None, truth: dict | None = None, rival_model: dm.DemandModel | None = None) -> dict:
    checks = []
    L = m.leader_index

    # 1. Instruments
    f = model.first_stage_f
    if f is None:
        checks.append(_check("Instrument strength", "warn", "Demand fitted by OLS: price endogeneity not handled."))
    else:
        weakest = min(x for x in (f, model.first_stage_f_nest) if x is not None)
        extra = f", within-nest share F = {model.first_stage_f_nest:.0f}" if model.first_stage_f_nest else ""
        checks.append(_check("Instrument strength", "pass" if weakest > 10 else "fail",
                             f"First-stage F for price = {f:.0f}{extra} (rule of thumb: > 10)."))

    # 2. Endogeneity bias
    bias = 100 * (model.ols_alpha - model.alpha) / model.alpha
    checks.append(_check("Endogeneity correction", "pass" if f is not None else "warn",
                         f"OLS price sensitivity is {bias:+.1f}% off the instrumented one; using OLS would have "
                         f"{'overstated' if bias < 0 else 'understated'} optimal markups."))

    # 3. Significance of price sensitivity
    lo, hi = model.alpha - 1.96 * model.alpha_se, model.alpha + 1.96 * model.alpha_se
    checks.append(_check("Price sensitivity identified", "pass" if lo > 0 else "fail",
                         f"alpha = {model.alpha:.4f}, 95% CI [{lo:.4f}, {hi:.4f}] (SEs clustered by product-region)."))

    # 4. Substitution pattern: is the nesting right?
    if rival_model is not None:
        nested, plain = (model, rival_model) if model.kind == "nested" else (rival_model, model)
        s_lo = nested.sigma - 1.96 * nested.sigma_se
        gap = 100 * (plain.alpha - nested.alpha) / nested.alpha
        if model.kind == "nested":
            checks.append(_check("Substitution pattern", "pass" if s_lo > 0.02 else "warn",
                                 f"Nested logit by grade, sigma = {nested.sigma:.2f} (95% CI from {s_lo:.2f}): buyers "
                                 f"substitute mostly within a grade. Plain logit would put price sensitivity "
                                 f"{gap:+.0f}% off and misprice every grade."))
        else:
            checks.append(_check("Substitution pattern", "pass" if s_lo <= 0.02 else "warn",
                                 f"Plain logit kept: within-grade nesting is not significant (sigma = "
                                 f"{nested.sigma:.2f}, CI from {s_lo:.2f})."))

    # 5. Stability across sub-samples
    ps = m.periods
    half = len(ps) // 2
    nested = model.kind == "nested"
    a1 = dm.fit(m, ps[:half], nested=nested).alpha
    a2 = dm.fit(m, ps[half:], nested=nested).alpha
    drift = abs(a1 - a2) / model.alpha
    checks.append(_check("Parameter stability", "pass" if drift < 0.15 else "warn" if drift < 0.3 else "fail",
                         f"alpha first half {a1:.4f} vs second half {a2:.4f} ({100 * drift:.0f}% apart).",
                         {"first_half": a1, "second_half": a2}))

    # 6. Out-of-sample skill
    if backtest and backtest.get("best"):
        b = backtest["best"]
        skill, hit = b["skill_vs_naive_pct"], b["directional_accuracy_pct"]
        status = "pass" if skill > 0 and hit > 55 else "warn" if skill > 0 else "fail"
        checks.append(_check("Out-of-sample skill", status,
                             f"Window {b['window']} periods: share MAE {b['mae_share_pts']:.2f} pts vs naive "
                             f"{b['naive_mae_share_pts']:.2f} ({skill:+.1f}% skill), direction right {hit:.0f}% "
                             f"of {b['moves_scored']} moves."))

    # 7. Elasticities at the recommendation
    game = build_game(m, model)
    el = elasticities(game, rec_outcome.eq.prices)
    mine = el[m.product_firm == L]
    mine = mine[~np.isnan(mine)]
    ok = np.all(mine < -1)
    checks.append(_check("Pricing on the elastic side", "pass" if ok else "fail",
                         f"Our own-price elasticities range {mine.min():.2f} to {mine.max():.2f} "
                         f"(a profit-maximiser never prices where |e| < 1)."))

    # 8. Market conduct
    ct = conduct_test(m, model, nash_eq)
    gaps = ct["median_abs_gap_pct"]
    if getattr(m, "costs_source", "stated") == "implied":
        checks.append(_check("Market conduct", "warn",
                             "Not testable on this market: its costs were themselves implied from prices under "
                             "Bertrand, so any conduct test would be circular. Supply observed cash costs to run it.",
                             ct))
    else:
        checks.append(_check("Market conduct", "pass" if ct["best"] == "bertrand" and gaps["bertrand"] < 15 else "warn",
                             f"Observed prices fit {CONDUCTS[ct['best']]} best: stated vs implied costs differ by "
                             f"{gaps['bertrand']:.1f}% under Bertrand, {gaps['collusion']:.1f}% under collusion, "
                             f"{gaps['competitive']:.1f}% under perfect competition (firms with spare capacity: "
                           f"{', '.join(ct['firms_used'])}).", ct))

    # 9. Equilibrium certificates
    kn, kr = nash_eq.kkt, rec_outcome.eq.kkt
    checks.append(_check("Equilibrium certified (KKT)", "pass" if kn.get("certified") and kr.get("certified") else "fail",
                         f"Nash stationarity {kn['stationarity_max_abs_usd_t']:.1e} $/t, complementarity "
                         f"{kn['complementarity']:.1e}; follower response stationarity "
                         f"{kr['stationarity_max_abs_usd_t']:.1e} $/t."))
    uq = uniqueness_check(game, rec_outcome.eq, fixed_firms=[L],
                          fixed_prices=leader_prices(game, L, rec_outcome.leader_markup))
    checks.append(_check("Rivals' response is unique", "pass" if uq["unique"] else "warn",
                         f"{uq['starts']} random restarts land within {uq['max_price_spread_usd_t']:.1e} $/t."))

    # 10. Extrapolation
    lead_obs = m.obs_price[m.product_firm[m.obs_product] == L]
    lo_p, hi_p = np.percentile(lead_obs, 1), np.percentile(lead_obs, 99)
    rp = rec_outcome.eq.prices[m.product_firm == L][game.avail[m.product_firm == L]]
    outside = float(np.mean((rp < lo_p * 0.95) | (rp > hi_p * 1.05)))
    checks.append(_check("Inside the data", "pass" if outside == 0 else "warn",
                         f"Recommended prices span {rp.min():.0f}-{rp.max():.0f} $/t; history spans "
                         f"{lo_p:.0f}-{hi_p:.0f}. {100 * outside:.0f}% of recommended prices extrapolate."))

    # 11. Robustness
    if mc:
        pw = mc["prob_recommended_wins_pct"]
        checks.append(_check("Robust to uncertainty", "pass" if pw >= 70 else "warn" if pw >= 50 else "fail",
                             f"Beats the status quo in {pw:.1f}% of {mc['draws']} joint draws "
                             f"(parameters, demand shocks, market sizes); P10 uplift {mc['uplift']['p10']:+.2f} $M."))

    # 12. Oracle (only when ground truth exists: synthetic data)
    if truth:
        a = truth["alpha"]
        inside = lo <= a <= hi
        msg = f"True alpha {a:.4f} is {'inside' if inside else 'outside'} the 95% CI"
        if model.kind == "nested" and "sigma" in truth:
            s_lo, s_hi = model.sigma - 1.96 * model.sigma_se, model.sigma + 1.96 * model.sigma_se
            s_in = s_lo <= truth["sigma"] <= s_hi
            inside = inside and s_in
            msg += f"; true sigma {truth['sigma']:.2f} is {'inside' if s_in else 'outside'} [{s_lo:.2f}, {s_hi:.2f}]"
        checks.append(_check("Ground truth recovered", "pass" if inside else "warn",
                             msg + ". (Available only on synthetic data.)"))

    fails = sum(c["status"] == "fail" for c in checks)
    warns = sum(c["status"] == "warn" for c in checks)
    grade = "D" if fails else "C" if warns >= 3 else "B" if warns else "A"
    verdict = {
        "A": "Ship it. Every check passed.",
        "B": "Usable, with the caveat flagged below.",
        "C": "Directionally useful; several caveats. Do not act on point estimates.",
        "D": "Do not act on this. At least one check failed.",
    }[grade]
    return {"grade": grade, "verdict": verdict, "fails": fails, "warns": warns, "checks": checks, "limits": LIMITS}
