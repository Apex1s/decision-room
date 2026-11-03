"""Capacity-constrained Bertrand-Nash equilibrium for (nested) logit demand.

Demand. Products are grouped into nests (by default, grades). With nesting
parameter sigma in [0, 1), buyers substitute more within a nest than across
nests; sigma = 0 is plain logit. Shares in region r:

    s_j = s_{j|g} * s_g,  s_{j|g} = e^{d_j/(1-s)} / D_g,  s_g = D_g^{1-s} / (1 + sum_h D_h^{1-s})

with d_j = delta0_j - alpha * (p_j + freight_j).

Supply. Firm f sets FOB prices to maximise profit subject to capacity.
Writing m_j = p_j - c_j - lambda_f, the first-order conditions are

    s + (Omega o dS/dp)' m = 0,

which we solve with the Morrow-Skerlos zeta-markup fixed point: split
dS/dp = -Lambda + Gamma (Lambda diagonal) and iterate

    m <- Lambda^{-1} (s + (Omega o Gamma) m).

lambda_f >= 0 is the capacity shadow price (KKT multiplier of Q_f <= K_f),
found by a projected subgradient with per-firm adaptive steps. The result
carries a KKT certificate and can be checked for uniqueness from random starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def nested_shares(d: np.ndarray, sigma: float, nest: np.ndarray, avail: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shares and within-nest shares for mean utilities d (J, R). Returns (s, s_within)."""
    one = 1.0 - sigma
    u = np.where(avail, d / one, -np.inf)
    G = int(nest.max()) + 1
    s = np.zeros_like(d)
    s_in = np.zeros_like(d)
    # Log-sum-exp per nest and region, stabilised.
    lD = np.full((G, d.shape[1]), -np.inf)
    for g in range(G):
        rows = nest == g
        if not rows.any():
            continue
        ug = u[rows]
        mx = np.max(ug, axis=0)
        mx_safe = np.where(np.isfinite(mx), mx, 0.0)
        tot = np.exp(ug - mx_safe).sum(axis=0)
        lD[g] = np.where(tot > 0, mx_safe + np.log(np.where(tot > 0, tot, 1.0)), -np.inf)
        s_in[rows] = np.where(avail[rows], np.exp(ug - np.where(np.isfinite(lD[g]), lD[g], 0.0)), 0.0)
    a = one * lD  # log of D_g^{1-sigma}
    amax = np.maximum(np.max(np.where(np.isfinite(a), a, -np.inf), axis=0), 0.0)
    denom = np.exp(-amax) + np.where(np.isfinite(a), np.exp(a - amax), 0.0).sum(axis=0)
    s_g = np.where(np.isfinite(a), np.exp(a - amax), 0.0) / denom
    s = s_in * s_g[nest]
    return s, s_in


@dataclass
class Game:
    """Everything the price game needs for one period."""

    delta0: np.ndarray  # (J, R) mean utility excluding price (incl. demand shocks)
    alpha: float  # price sensitivity, per $/t of delivered price
    freight: np.ndarray  # (J, R) $/t
    cost: np.ndarray  # (J,) marginal FOB cost
    product_firm: np.ndarray  # (J,)
    capacity: np.ndarray  # (F,)
    size: np.ndarray  # (R,) Mt
    avail: np.ndarray  # (J, R) bool: product sold in region
    sigma: float = 0.0  # nesting parameter; 0 = plain logit
    nest: np.ndarray | None = None  # (J,) nest id per product

    def __post_init__(self):
        if self.nest is None:
            self.nest = np.zeros(len(self.cost), int)

    @property
    def F(self) -> int:
        return len(self.capacity)

    def onehot(self) -> np.ndarray:
        oh = np.zeros((self.F, len(self.cost)))
        oh[self.product_firm, np.arange(len(self.cost))] = 1.0
        return oh

    def ownership(self, conduct: str = "bertrand") -> np.ndarray:
        """Omega: which products' profits each first-order condition internalises."""
        J = len(self.cost)
        if conduct == "collusion":
            return np.ones((J, J))
        if conduct == "competitive":
            return np.zeros((J, J))
        return (self.product_firm[:, None] == self.product_firm[None, :]).astype(float)

    def _d(self, prices: np.ndarray) -> np.ndarray:
        return self.delta0 - self.alpha * (prices + self.freight)

    def shares(self, prices: np.ndarray) -> np.ndarray:
        return nested_shares(self._d(prices), self.sigma, self.nest, self.avail)[0]

    def derivative_parts(self, prices: np.ndarray):
        """Lambda (J, R) and Gamma (R, J, J) with dS/dp = -diag(Lambda) + Gamma, per region."""
        s, s_in = nested_shares(self._d(prices), self.sigma, self.nest, self.avail)
        a, sg = self.alpha, self.sigma
        lam = a * s / (1.0 - sg)
        same = (self.nest[:, None] == self.nest[None, :]).astype(float)
        # Gamma_jk = alpha * (sigma/(1-sigma) * s_j * s_{k|g} * 1[same nest] + s_j * s_k)
        sT, sinT = s.T, s_in.T  # (R, J)
        gamma = a * (sg / (1.0 - sg) * sT[:, :, None] * sinT[:, None, :] * same[None]
                     + sT[:, :, None] * sT[:, None, :])
        return s, lam, gamma

    def quantities(self, prices: np.ndarray) -> np.ndarray:
        return self.shares(prices) * self.size[None, :]

    def firm_profit(self, prices: np.ndarray) -> np.ndarray:
        q = self.quantities(prices)
        margin = (prices - self.cost[:, None]) * q
        return self.onehot() @ np.where(self.avail, margin, 0.0).sum(axis=1)


@dataclass
class Equilibrium:
    prices: np.ndarray  # (J, R) FOB
    shares: np.ndarray
    quantities: np.ndarray
    shadow: np.ndarray  # (F,) capacity shadow prices, $/t
    markup: np.ndarray  # (J, R) p - c - lambda (the first-order-condition markup)
    profit: np.ndarray  # (F,) per period, $M (Mt x $/t)
    firm_volume: np.ndarray  # (F,)
    iterations: dict
    kkt: dict = field(default_factory=dict)

    def to_dict(self, firms: list[str], regions: list[str], products: list[str], product_firm: np.ndarray) -> dict:
        q = np.where(self.quantities > 0, self.quantities, 0.0)
        firm_markup = {}
        for i, f in enumerate(firms):
            rows = product_firm == i
            w = q[rows]
            firm_markup[f] = {r: round(float((self.markup[rows, k] * w[:, k]).sum() / max(w[:, k].sum(), 1e-12)), 2)
                              for k, r in enumerate(regions)}
        return {
            "shadow_price": {f: round(float(v), 3) for f, v in zip(firms, self.shadow)},
            "profit_musd": {f: round(float(v), 3) for f, v in zip(firms, self.profit)},
            "volume_mt": {f: round(float(v), 4) for f, v in zip(firms, self.firm_volume)},
            "avg_price": {
                p: round(float(np.nanmean(np.where(self.quantities[j] > 0, self.prices[j], np.nan))), 2)
                for j, p in enumerate(products)
            },
            "markup": firm_markup,
            "iterations": self.iterations,
            "kkt": self.kkt,
        }


def _inner(game: Game, lam: np.ndarray, omega: np.ndarray, free: np.ndarray, fixed_prices: np.ndarray | None,
           m0: np.ndarray | None, tol: float, max_iter: int):
    """zeta-markup fixed point for given shadow prices. Returns prices, markups, iterations."""
    base = game.cost[:, None] + lam[game.product_firm][:, None]
    m = np.full_like(game.delta0, 1.0 / game.alpha) if m0 is None else m0.copy()
    it = 0
    weight, last = 1.0, np.inf  # undamped zeta map; damp only if it starts to oscillate
    for it in range(1, max_iter + 1):
        prices = np.where(free, base + m, fixed_prices if fixed_prices is not None else base + m)
        s, lam_d, gamma = game.derivative_parts(prices)
        og = omega[None] * gamma  # (R, J, J)
        new = (s + np.einsum("rjk,rk->jr", og, np.where(free, m, 0.0).T)) / np.maximum(lam_d, 1e-300)
        new = np.where(game.avail & free, new, m)
        step = np.max(np.abs(new - m))
        if step > last and weight > 0.2:
            weight *= 0.5
        last = step
        new = (1 - weight) * m + weight * new
        if step < tol:
            m = new
            break
        m = new
    prices = np.where(free, base + m, fixed_prices if fixed_prices is not None else base + m)
    return prices, m, it


def solve(game: Game, fixed_firms: list[int] | None = None, fixed_prices: np.ndarray | None = None,
          tol: float = 1e-8, max_outer: int = 400, max_inner: int = 3000,
          lam0: np.ndarray | None = None, mu0: np.ndarray | None = None,
          certify: bool = True, conduct: str = "bertrand") -> Equilibrium:
    """Solve the game. Firms in fixed_firms keep fixed_prices (a committed Stackelberg leader)."""
    oh = game.onehot()
    F = game.F
    free_firm = np.ones(F, bool)
    if fixed_firms:
        free_firm[fixed_firms] = False
    free = free_firm[game.product_firm][:, None] & np.ones_like(game.avail, bool)
    omega = game.ownership(conduct)

    lam = np.zeros(F) if lam0 is None else lam0.copy()
    step = np.full(F, 0.25 / game.alpha)
    last_sign = np.zeros(F)
    m = None if mu0 is None or mu0.shape != game.delta0.shape else mu0
    inner_total = 0
    outer = 0
    for outer in range(1, max_outer + 1):
        prices, m, n = _inner(game, lam, omega, free, fixed_prices, m, tol * 1e-2, max_inner)
        inner_total += n
        Q = oh @ np.where(game.avail, game.quantities(prices), 0.0).sum(axis=1)
        gap = np.where(free_firm, (Q - game.capacity) / game.capacity, 0.0)  # > 0 means over capacity
        viol = np.maximum(gap, 0.0)
        slack_with_price = np.where(lam > 1e-9, np.abs(gap), 0.0)
        if max(viol.max(initial=0), slack_with_price.max(initial=0)) < 1e-7:
            break
        sign = np.sign(gap)
        flipped = (sign * last_sign) < 0
        step = np.where(flipped, step * 0.5, step * 1.2)
        last_sign = sign
        lam = np.where(free_firm, np.maximum(0.0, lam + step * gap), 0.0)

    shares = game.shares(prices)
    q = shares * game.size[None, :]
    eq = Equilibrium(
        prices=prices, shares=shares, quantities=q, shadow=lam, markup=m,
        profit=game.firm_profit(prices), firm_volume=oh @ np.where(game.avail, q, 0.0).sum(axis=1),
        iterations={"outer": outer, "inner_total": inner_total},
    )
    if certify:
        eq.kkt = kkt_certificate(game, eq, free_firm, conduct)
    return eq


def foc_residual(game: Game, prices: np.ndarray, markup: np.ndarray, omega: np.ndarray) -> np.ndarray:
    """s + (Omega o dS/dp)' m for every product and region; zero at a first-order point."""
    s, lam_d, gamma = game.derivative_parts(prices)
    return s - lam_d * markup + np.einsum("rjk,rk->jr", omega[None] * gamma, markup.T)


def kkt_certificate(game: Game, eq: Equilibrium, free_firm: np.ndarray | None = None,
                    conduct: str = "bertrand") -> dict:
    F = game.F
    if free_firm is None:
        free_firm = np.ones(F, bool)
    free_prod = free_firm[game.product_firm][:, None] & game.avail
    # Express the FOC residual in $/t: divide by the own-price slope.
    res = foc_residual(game, eq.prices, eq.prices - game.cost[:, None] - eq.shadow[game.product_firm][:, None],
                       game.ownership(conduct))
    _, lam_d, _ = game.derivative_parts(eq.prices)
    stationarity = float(np.max(np.abs(np.where(free_prod, res / np.maximum(lam_d, 1e-300), 0.0)), initial=0.0))
    slack = (game.capacity - eq.firm_volume) / game.capacity
    primal = float(np.max(np.where(free_firm, np.maximum(-slack, 0.0), 0.0)))
    dual = float(np.min(eq.shadow)) if F else 0.0
    comp = float(np.max(np.where(free_firm, np.abs(eq.shadow * slack), 0.0)))
    ok = stationarity < 1e-4 and primal < 1e-5 and dual >= -1e-12 and comp < 1e-3
    return {
        "stationarity_max_abs_usd_t": stationarity,
        "primal_infeasibility": primal,
        "dual_min_shadow": dual,
        "complementarity": comp,
        "binding_capacity": [int(i) for i in np.where((eq.shadow > 1e-6) & free_firm)[0]],
        "certified": bool(ok),
    }


def uniqueness_check(game: Game, eq: Equilibrium, starts: int = 6, seed: int = 7,
                     fixed_firms: list[int] | None = None, fixed_prices: np.ndarray | None = None) -> dict:
    """Re-solve from random starting markups and shadow prices; report the spread."""
    rng = np.random.default_rng(seed)
    spread = 0.0
    for _ in range(starts):
        mu0 = rng.uniform(0.2, 4.0, size=game.delta0.shape) / game.alpha
        lam0 = rng.uniform(0.0, 2.0, size=game.F) / game.alpha
        other = solve(game, fixed_firms, fixed_prices, lam0=lam0, mu0=mu0, certify=False)
        spread = max(spread, float(np.max(np.abs(np.where(game.avail, other.prices - eq.prices, 0.0)))))
    return {"starts": starts, "max_price_spread_usd_t": spread, "unique": spread < 1e-3}


def elasticities(game: Game, prices: np.ndarray) -> np.ndarray:
    """Own FOB-price elasticity: -alpha * p * (1/(1-sigma) - sigma/(1-sigma) * s_{j|g} - s_j)."""
    s, s_in = nested_shares(game._d(prices), game.sigma, game.nest, game.avail)
    sg = game.sigma
    return np.where(game.avail, -game.alpha * prices * (1 / (1 - sg) - sg / (1 - sg) * s_in - s), np.nan)


def implied_costs(game: Game, prices: np.ndarray, conduct: str = "bertrand") -> np.ndarray:
    """Marginal costs (incl. any capacity shadow price) that make observed prices optimal under a conduct.

    Solves (diag(Lambda) - Omega o Gamma) m = s region by region; implied cost = p - m.
    """
    s, lam_d, gamma = game.derivative_parts(prices)
    omega = game.ownership(conduct)
    out = np.full_like(prices, np.nan)
    if conduct == "competitive":
        return np.where(game.avail, prices, np.nan)
    for r in range(prices.shape[1]):
        idx = np.where(game.avail[:, r])[0]
        A = np.diag(lam_d[idx, r]) - (omega * gamma[r])[np.ix_(idx, idx)]
        m = np.linalg.solve(A, s[idx, r])
        out[idx, r] = prices[idx, r] - m
    return out
