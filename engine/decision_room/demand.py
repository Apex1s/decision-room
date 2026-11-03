"""Logit and nested-logit demand estimated by Berry inversion, with and without instruments.

Berry (1994): with logit demand and an outside option, observed shares invert
to mean utilities in closed form,

    delta_jrt = ln s_jrt - ln s_0rt = x_jr' beta - alpha * (p_jrt + freight_jr) + xi_jrt.

Prices are set knowing xi (sellers see demand shocks), so OLS on this equation
underestimates alpha. Cost shifters and freight move prices but not demand,
which makes them valid instruments; we report both estimates side by side.

Nested logit (Berry 1994) groups products into nests (grades) and adds the log
within-nest share:

    ln s_j - ln s_0 = x' beta - alpha * p + sigma * ln s_{j|g} + xi.

ln s_{j|g} is endogenous too; it is instrumented by the cost shifters of the
other products in the same nest and by the number of products in the nest.
sigma near 0 means plain logit is enough; larger sigma means buyers substitute
mostly within their grade, which plain logit cannot represent (IIA).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .equilibrium import nested_shares
from .market import Market


@dataclass
class DemandModel:
    alpha: float
    alpha_se: float
    coef: dict[str, float]
    se: dict[str, float]
    method: str
    n: int
    first_stage_f: float | None
    xi_rho: float  # persistence of demand shocks, estimated from residuals
    residual_sd: float
    ols_alpha: float
    xi_last: np.ndarray = field(repr=False)  # (J, R) latest demand shocks
    delta_bar: np.ndarray = field(repr=False)  # (J, R) mean utility excl. price and shocks
    window: tuple[int, int] = (0, 0)
    cov: np.ndarray = field(repr=False, default=None)
    names: list[str] = field(default_factory=list)
    kind: str = "logit"  # "logit" or "nested"
    sigma: float = 0.0
    sigma_se: float = 0.0
    nest: np.ndarray | None = field(repr=False, default=None)  # (J,) nest id per product
    first_stage_f_nest: float | None = None

    def delta0(self, shock: str = "persist") -> np.ndarray:
        """Mean utility (excl. price) for next period's game."""
        if shock == "none":
            return self.delta_bar
        return self.delta_bar + self.xi_rho * self.xi_last

    def to_dict(self) -> dict:
        lo, hi = self.alpha - 1.96 * self.alpha_se, self.alpha + 1.96 * self.alpha_se
        return {
            "method": self.method,
            "alpha": round(self.alpha, 5),
            "alpha_ci95": [round(lo, 5), round(hi, 5)],
            "ols_alpha": round(self.ols_alpha, 5),
            "endogeneity_bias_pct": round(100 * (self.ols_alpha - self.alpha) / self.alpha, 1),
            "first_stage_f": None if self.first_stage_f is None else round(self.first_stage_f, 1),
            "coefficients": {k: round(v, 4) for k, v in self.coef.items()},
            "std_errors": {k: round(v, 4) for k, v in self.se.items()},
            "xi_persistence": round(self.xi_rho, 3),
            "residual_sd": round(self.residual_sd, 4),
            "observations": self.n,
            "window": list(self.window),
            "kind": self.kind,
            "sigma": round(self.sigma, 4),
            "sigma_ci95": [round(self.sigma - 1.96 * self.sigma_se, 4), round(self.sigma + 1.96 * self.sigma_se, 4)],
            "first_stage_f_nest": None if self.first_stage_f_nest is None else round(self.first_stage_f_nest, 1),
        }


@dataclass
class Design:
    y: np.ndarray
    X: np.ndarray
    Z: np.ndarray
    w: np.ndarray
    names: list[str]
    price_col: int
    rows: np.ndarray  # indices into market observations
    endog: list[int] = field(default_factory=list)  # columns of X that are instrumented
    clusters: np.ndarray | None = None  # cluster id per row for robust standard errors


def _design(m: Market, periods: np.ndarray, nested: bool = False) -> Design:
    rows = np.where(np.isin(m.obs_period, periods))[0]
    t, r, j = m.obs_period[rows], m.obs_region[rows], m.obs_product[rows]
    s = m.obs_quantity[rows] / m.obs_size[rows]
    inside = {}
    for tt, rr, ss in zip(t, r, s):
        inside[(tt, rr)] = inside.get((tt, rr), 0.0) + ss
    s0 = np.array([1.0 - inside[(tt, rr)] for tt, rr in zip(t, r)])
    if np.any(s0 <= 0):
        raise ValueError("inside shares sum to 1 or more in some region-period: market_size must include the "
                         "outside option (domestic supply, substitutes)")
    y = np.log(s) - np.log(s0)
    freight = m.freight[m.product_firm[j], r]
    p_del = m.obs_price[rows] + freight

    cols, names = [np.ones(len(rows))], ["const"]
    for g in m.grades[1:]:
        cols.append(np.array([m.product_grade[k] == g for k in j], float)); names.append(f"grade:{g}")
    for f in range(1, m.F):
        cols.append((m.product_firm[j] == f).astype(float)); names.append(f"firm:{m.firms[f]}")
    for k in range(1, m.R):
        cols.append((r == k).astype(float)); names.append(f"region:{m.regions[k]}")
    if m.time_effects:
        for p in np.unique(t)[1:]:
            cols.append((t == p).astype(float)); names.append(f"period:{int(p)}")
    exog = np.column_stack(cols)
    X = np.column_stack([exog, p_del])
    names.append("delivered_price")
    endog = [X.shape[1] - 1]

    # Excluded instruments: own cost shifter, freight, rivals' mean cost shifter in the period.
    own_cost = m.cost[j] * m.obs_cost_index[rows]
    zcols = [exog, own_cost, freight, _rival_cost(m)[rows]]
    if m.obs_instruments is not None:
        zcols.append(m.obs_instruments[rows])
    if nested:
        g = m.nest_ids()[j]
        # Within-nest share and nest-level instruments, per (period, region, nest).
        key = list(zip(t, r, g))
        tot, cnt, csum = {}, {}, {}
        for k_, ss, cc in zip(key, s, own_cost):
            tot[k_] = tot.get(k_, 0.0) + ss
            cnt[k_] = cnt.get(k_, 0) + 1
            csum[k_] = csum.get(k_, 0.0) + cc
        s_in = np.array([ss / tot[k_] for k_, ss in zip(key, s)])
        n_nest = np.array([cnt[k_] for k_ in key], float)
        other_cost = np.array([(csum[k_] - cc) / max(cnt[k_] - 1, 1) for k_, cc in zip(key, own_cost)])
        X = np.column_stack([X, np.log(s_in)])
        names.append("ln_within_nest_share")
        endog.append(X.shape[1] - 1)
        zcols += [other_cost, n_nest, other_cost * n_nest]
    Z = np.column_stack(zcols)
    # Unweighted on purpose: volume weights depend on the demand shock itself, which biases even
    # the instrumented estimate (we measured it on the synthetic market). Clusters = product-region
    # pairs, because demand shocks persist over time.
    w = np.ones(len(rows))
    clusters = j * m.R + r
    return Design(clusters=clusters, y=y, X=X, Z=Z, w=w, names=names, price_col=endog[0], rows=rows, endog=endog)


def _rival_cost(m: Market) -> np.ndarray:
    """Per observation: mean cost shifter of rival firms' rows in the same period."""
    c = m.cost[m.obs_product] * m.obs_cost_index
    firm = m.product_firm[m.obs_product]
    out = np.zeros(len(c))
    for t in np.unique(m.obs_period):
        in_t = m.obs_period == t
        tot, cnt = c[in_t].sum(), in_t.sum()
        for f in range(m.F):
            own = in_t & (firm == f)
            n_rival = cnt - own.sum()
            out[own] = (tot - c[own].sum()) / n_rival if n_rival else 0.0
    return out


def _cluster_cov(X: np.ndarray, resid: np.ndarray, clusters: np.ndarray | None) -> np.ndarray:
    """Cluster-robust sandwich covariance (HC1 when clusters is None), small-sample corrected."""
    n, k = X.shape
    bread = np.linalg.pinv(X.T @ X)
    if clusters is None:
        meat = (X * resid[:, None] ** 2).T @ X
        return bread @ meat @ bread * n / (n - k)
    ids, inv = np.unique(clusters, return_inverse=True)
    G = len(ids)
    scores = np.zeros((G, k))
    np.add.at(scores, inv, X * resid[:, None])
    meat = scores.T @ scores
    return bread @ meat @ bread * (G / (G - 1)) * ((n - 1) / (n - k))


def _wls(X, y, w):
    Xw, yw = X * w[:, None], y * w
    beta, *_ = np.linalg.lstsq(Xw, yw, rcond=None)
    return beta


def _tsls(d: Design):
    Zw, Xw, yw = d.Z * d.w[:, None], d.X * d.w[:, None], d.y * d.w
    # First stage: project X on Z.
    pi, *_ = np.linalg.lstsq(Zw, Xw, rcond=None)
    Xhat = Zw @ pi
    beta, *_ = np.linalg.lstsq(Xhat, yw, rcond=None)
    resid = yw - Xw @ beta
    n, k = Xhat.shape
    cov = _cluster_cov(Xhat, resid, d.clusters)
    # First-stage F on the excluded instruments, one per endogenous column.
    exog = Zw[:, : d.X.shape[1] - len(d.endog)]
    q = Zw.shape[1] - exog.shape[1]
    fs = []
    for col in d.endog:
        px = Xw[:, col]
        r_full = px - Zw @ np.linalg.lstsq(Zw, px, rcond=None)[0]
        r_rest = px - exog @ np.linalg.lstsq(exog, px, rcond=None)[0]
        fs.append(float(((r_rest @ r_rest - r_full @ r_full) / q) / (r_full @ r_full / (n - Zw.shape[1]))))
    return beta, cov, fs


def fit(m: Market, periods: np.ndarray | None = None, instruments: bool = True,
        nested: bool = False) -> DemandModel:
    periods = m.periods if periods is None else np.asarray(periods)
    d = _design(m, periods, nested)
    ols = _wls(d.X, d.y, d.w)
    fstat = fnest = None
    label = "nested logit by grade" if nested else "logit"
    if instruments:
        beta, cov, fs = _tsls(d)
        fstat = fs[0]
        fnest = fs[1] if nested else None
        method = f"2SLS {label} (Berry inversion, SEs clustered by product-region)"
    else:
        beta = ols
        resid = (d.y - d.X @ beta) * d.w
        cov = _cluster_cov(d.X * d.w[:, None], resid, d.clusters)
        method = f"OLS {label} (Berry inversion, SEs clustered by product-region)"

    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    resid = d.y - d.X @ beta
    xi = _xi_panel(m, d, resid)
    xi_rho = _persistence(xi)
    last_t = int(periods.max())
    xi_last = np.nan_to_num(xi.get(last_t, np.zeros((m.J, m.R))))
    delta_bar = _delta_bar(m, beta, d.names)
    skip = ("delivered_price", "ln_within_nest_share")
    sig_i = d.names.index("ln_within_nest_share") if nested else None
    return DemandModel(
        alpha=float(-beta[d.price_col]), alpha_se=float(se[d.price_col]),
        coef={n: float(b) for n, b in zip(d.names, beta) if n not in skip},
        se={n: float(s) for n, s in zip(d.names, se) if n not in skip},
        method=method, n=len(d.y), first_stage_f=fstat, xi_rho=xi_rho,
        residual_sd=float(np.std(resid)), ols_alpha=float(-ols[d.price_col]),
        xi_last=xi_last, delta_bar=delta_bar, window=(int(periods.min()), last_t), cov=cov, names=d.names,
        kind="nested" if nested else "logit",
        sigma=float(np.clip(beta[sig_i], 0.0, 0.95)) if nested else 0.0,
        sigma_se=float(se[sig_i]) if nested else 0.0,
        nest=m.nest_ids() if nested else np.zeros(m.J, int),
        first_stage_f_nest=fnest,
    )


def _delta_bar(m: Market, beta: np.ndarray, names: list[str]) -> np.ndarray:
    coef = dict(zip(names, beta))
    out = np.full((m.J, m.R), coef["const"])
    for j in range(m.J):
        out[j] += coef.get(f"grade:{m.product_grade[j]}", 0.0)
        out[j] += coef.get(f"firm:{m.firms[m.product_firm[j]]}", 0.0)
    for r in range(m.R):
        out[:, r] += coef.get(f"region:{m.regions[r]}", 0.0)
    # With period effects, next period inherits the latest one (a random walk in the market-wide level).
    periods = [int(n.split(":", 1)[1]) for n in names if n.startswith("period:")]
    if periods:
        out += coef[f"period:{max(periods)}"]
    return out


def _xi_panel(m: Market, d: Design, resid: np.ndarray) -> dict[int, np.ndarray]:
    out: dict[int, np.ndarray] = {}
    for i, row in enumerate(d.rows):
        t = int(m.obs_period[row])
        if t not in out:
            out[t] = np.full((m.J, m.R), np.nan)
        out[t][m.obs_product[row], m.obs_region[row]] = resid[i]
    return out


def _persistence(xi: dict[int, np.ndarray]) -> float:
    ts = sorted(xi)
    a, b = [], []
    for t0, t1 in zip(ts, ts[1:]):
        ok = ~np.isnan(xi[t0]) & ~np.isnan(xi[t1])
        a.append(xi[t0][ok]); b.append(xi[t1][ok])
    if not a:
        return 0.0
    a, b = np.concatenate(a), np.concatenate(b)
    return float(np.clip(a @ b / max(a @ a, 1e-12), 0.0, 0.99))


def predict_shares(m: Market, model: DemandModel, t: int) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and actual (J, R) shares for period t, given that period's prices."""
    sl = m.period_slice(t)
    avail = ~np.isnan(sl["price"])
    d = model.delta0() - model.alpha * (np.nan_to_num(sl["price"]) + m.product_freight())
    pred = nested_shares(d, model.sigma, model.nest, avail)[0]
    actual = np.where(avail, np.nan_to_num(sl["quantity"]) / sl["size"][None, :], 0.0)
    return pred, actual


def backtest(m: Market, windows: tuple[int, ...] = (8, 12, 16, 20, 24), test_periods: int = 8,
             nested: bool = False) -> dict:
    """Rolling one-step-ahead validation for several calibration-window lengths.

    For each test period t, fit on the w periods before t, predict t's shares
    from t's prices, and compare with what happened. The naive benchmark is
    "shares stay where they were last period"; a model that cannot beat it is
    not worth its complexity.
    """
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
        err, naive_err, hits, total = [], [], 0, 0
        for t in tests:
            model = fit(m, np.arange(t - w, t), nested=nested)
            pred, actual = predict_shares(m, model, int(t))
            _, prev = predict_shares(m, model, int(t - 1))
            avail = actual > 0
            err.append(np.abs(pred - actual)[avail]); naive_err.append(np.abs(prev - actual)[avail])
            move = actual - prev
            called = pred - prev
            big = avail & (np.abs(move) > 1e-3)
            hits += int(np.sum(np.sign(move[big]) == np.sign(called[big]))); total += int(big.sum())
        mae = float(np.mean(np.concatenate(err)))
        naive = float(np.mean(np.concatenate(naive_err)))
        results.append({
            "window": w, "mae_share_pts": round(100 * mae, 3), "naive_mae_share_pts": round(100 * naive, 3),
            "skill_vs_naive_pct": round(100 * (1 - mae / naive), 1),
            "directional_accuracy_pct": round(100 * hits / max(total, 1), 1), "moves_scored": total,
        })
    best = min(results, key=lambda r: r["mae_share_pts"]) if results else None
    # A shorter window trades precision for recency. Only accept that trade when
    # it buys a material forecasting gain; otherwise keep the full sample, which
    # pins down the price sensitivity far more tightly.
    longest = max(results, key=lambda r: r["window"]) if results else None
    shorten = bool(best and longest and best["mae_share_pts"] < 0.98 * longest["mae_share_pts"])
    return {"test_periods": [int(t) for t in tests], "windows": results,
            "retained_window": best["window"] if shorten else None, "best": best,
            "rule": "keep the full sample unless a shorter window cuts share MAE by more than 2%"}
