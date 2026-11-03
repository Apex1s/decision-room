"""Out-of-sample validation of the whole engine: forecast next quarter's
equilibrium (prices AND shares) and compare with what happened.

Why not "predict shares from realised prices"? Producers set those prices
knowing the demand shock, so conditioning on them leaks the shock and biases
the test against any structural model (more so under nesting, which
amplifies within-grade shocks). The fair question is the one a pricing team
faces: knowing costs, capacities and market sizes for next quarter, but not
its demand shock, where will prices and shares land?

The naive benchmark is "everything stays where it was last quarter".
"""

from __future__ import annotations

import numpy as np

from . import demand as dm
from .equilibrium import Game, solve
from .market import Market


def game_for_period(m: Market, model: dm.DemandModel, t: int) -> Game:
    """The game of quarter t as seen at the end of quarter t-1."""
    sl = m.period_slice(t)
    avail = ~np.isnan(sl["price"])
    return Game(
        delta0=model.delta0(), alpha=model.alpha, freight=m.product_freight(),
        cost=m.cost * np.nan_to_num(sl["cost_index"], nan=1.0), product_firm=m.product_firm,
        capacity=m.capacity_at(t), size=sl["size"], avail=avail, sigma=model.sigma,
        nest=model.nest if model.nest is not None else np.zeros(m.J, int),
    )


def backtest(m: Market, windows: tuple[int, ...] = (8, 12, 16, 20, 24), test_periods: int = 8,
             nested: bool = False) -> dict:
    periods = m.periods
    # Short panels (e.g. 13 years): test on the last quarter of the sample and use windows that fit.
    test_periods = min(test_periods, max(2, len(periods) // 4))
    room = len(periods) - test_periods
    windows = tuple(w for w in windows if w <= room) or tuple(sorted({max(4, room // 2), room}))
    tests = periods[-test_periods:]
    results = []
    for w in windows:
        if tests[0] - w < periods.min():
            continue
        s_err, s_naive, p_err, p_naive = [], [], [], []
        hits = total = p_hits = p_total = 0
        for t in tests:
            model = dm.fit(m, np.arange(t - w, t), nested=nested)
            eq = solve(game_for_period(m, model, int(t)), certify=False)
            now, prev = m.period_slice(int(t)), m.period_slice(int(t - 1))
            ok = ~np.isnan(now["price"]) & ~np.isnan(prev["price"])
            s_act = np.nan_to_num(now["quantity"]) / now["size"][None, :]
            s_prev = np.nan_to_num(prev["quantity"]) / prev["size"][None, :]
            s_err.append(np.abs(eq.shares - s_act)[ok]); s_naive.append(np.abs(s_prev - s_act)[ok])
            p_err.append(np.abs(eq.prices - now["price"])[ok]); p_naive.append(np.abs(prev["price"] - now["price"])[ok])
            ds, dsh = (s_act - s_prev)[ok], (eq.shares - s_prev)[ok]
            big = np.abs(ds) > 1e-3
            hits += int(np.sum(np.sign(ds[big]) == np.sign(dsh[big]))); total += int(big.sum())
            dp, dph = (now["price"] - prev["price"])[ok], (eq.prices - prev["price"])[ok]
            bigp = np.abs(dp) > 0.25
            p_hits += int(np.sum(np.sign(dp[bigp]) == np.sign(dph[bigp]))); p_total += int(bigp.sum())
        mae, naive = float(np.mean(np.concatenate(s_err))), float(np.mean(np.concatenate(s_naive)))
        pmae, pnaive = float(np.mean(np.concatenate(p_err))), float(np.mean(np.concatenate(p_naive)))
        results.append({
            "window": w,
            "mae_share_pts": round(100 * mae, 3), "naive_mae_share_pts": round(100 * naive, 3),
            "skill_vs_naive_pct": round(100 * (1 - mae / naive), 1),
            "mae_price_usd_t": round(pmae, 2), "naive_mae_price_usd_t": round(pnaive, 2),
            "price_skill_vs_naive_pct": round(100 * (1 - pmae / pnaive), 1),
            "directional_accuracy_pct": round(100 * hits / max(total, 1), 1), "moves_scored": total,
            "price_direction_pct": round(100 * p_hits / max(p_total, 1), 1), "price_moves_scored": p_total,
        })
    best = min(results, key=lambda r: r["mae_share_pts"]) if results else None
    longest = max(results, key=lambda r: r["window"]) if results else None
    shorten = bool(best and longest and best["mae_share_pts"] < 0.98 * longest["mae_share_pts"])
    return {"test": "equilibrium forecast (prices and shares, next quarter's demand shock unknown)",
            "test_periods": [int(t) for t in tests], "windows": results,
            "retained_window": best["window"] if shorten else None, "best": best,
            "rule": "keep the full sample unless a shorter window cuts share MAE by more than 2%"}
