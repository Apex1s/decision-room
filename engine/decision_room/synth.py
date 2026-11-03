"""Synthetic oligopoly market with a known ground truth.

The demo market is a seaborne mined commodity with four
fictional producers, three grades and six importing regions. Prices in the
panel are generated *by the equilibrium model itself*, with demand shocks the
analyst never sees. That makes prices endogenous, exactly as in real data, so
naive OLS demand estimates are biased and the instrumented estimate is not.
Because the true parameters are known, every stage of the pipeline can be
tested against them.

Demand is nested logit by grade (sigma = 0.45), so a plain-logit model is
misspecified and the engine has to discover that. None of these numbers are
real market data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .equilibrium import Game, solve
from .market import Market

FIRMS = ["Atlas", "Boreal", "Cerro", "Dune"]
LEADER = "Atlas"
CAPACITY = np.array([3.3, 1.65, 2.1, 0.85])  # Mt per quarter
REGIONS = ["South Asia", "Latin America", "Europe", "East Asia", "Africa", "North America"]
BASE_SIZE = np.array([3.0, 2.4, 1.6, 1.8, 1.0, 1.2])  # Mt per quarter, all suppliers + outside option
FREIGHT = np.array([  # $/t from each producer to each region
    [28, 30, 14, 38, 16, 26],  # Atlas
    [34, 32, 18, 30, 30, 12],  # Boreal
    [36, 12, 32, 34, 34, 22],  # Cerro
    [18, 40, 22, 26, 22, 38],  # Dune
], dtype=float)
PRODUCTS = [  # (firm, grade, marginal FOB cost $/t)
    ("Atlas", "Low", 55.0), ("Atlas", "Mid", 68.0), ("Atlas", "High", 82.0),
    ("Boreal", "Mid", 72.0), ("Boreal", "High", 88.0),
    ("Cerro", "Low", 50.0), ("Cerro", "Mid", 63.0),
    ("Dune", "Low", 45.0),
]
GRADES = ["Low", "Mid", "High"]
UNAVAILABLE = [("Dune-Low", "North America"), ("Cerro-Low", "Europe")]


@dataclass(frozen=True)
class Truth:
    alpha: float = 0.022
    sigma: float = 0.45  # nesting by grade: buyers substitute mostly within a grade
    const: float = 1.55
    grade: tuple = (("Low", 0.0), ("Mid", 0.35), ("High", 0.70))
    firm: tuple = (("Atlas", 0.20), ("Boreal", 0.10), ("Cerro", 0.0), ("Dune", -0.10))
    region: tuple = (0.10, 0.05, -0.15, 0.0, -0.20, 0.05)
    xi_rho: float = 0.6
    xi_sd: float = 0.15
    cost_sd: float = 0.06
    growth_per_year: float = 0.02
    measurement_noise: float = 0.02

    def as_dict(self) -> dict:
        return {"alpha": self.alpha, "sigma": self.sigma, "const": self.const, "grade": dict(self.grade), "firm": dict(self.firm),
                "region": dict(zip(REGIONS, self.region))}


def generate(periods: int = 40, seed: int = 2026, truth: Truth = Truth()) -> tuple[Market, Truth]:
    rng = np.random.default_rng(seed)
    J, R, F = len(PRODUCTS), len(REGIONS), len(FIRMS)
    names = [f"{f}-{g}" for f, g, _ in PRODUCTS]
    pf = np.array([FIRMS.index(f) for f, _, _ in PRODUCTS])
    grades = [g for _, g, _ in PRODUCTS]
    cost = np.array([c for _, _, c in PRODUCTS])
    avail = np.ones((J, R), bool)
    for p, r in UNAVAILABLE:
        avail[names.index(p), REGIONS.index(r)] = False

    gmap, fmap = dict(truth.grade), dict(truth.firm)
    delta_bar = (truth.const + np.array([gmap[g] for g in grades])[:, None]
                 + np.array([fmap[FIRMS[f]] for f in pf])[:, None] + np.array(truth.region)[None, :])

    xi = rng.normal(0, truth.xi_sd, (J, R))
    cidx = np.ones(F)
    rows = {k: [] for k in ("t", "r", "j", "p", "q", "m", "c")}
    capacity_path = {}
    lam = mu = None
    for t in range(1, periods + 1):
        xi = truth.xi_rho * xi + np.sqrt(1 - truth.xi_rho ** 2) * rng.normal(0, truth.xi_sd, (J, R))
        cidx = 1 + 0.7 * (cidx - 1) + rng.normal(0, truth.cost_sd, F)
        season = 1 + 0.06 * np.sin(2 * np.pi * (t % 4) / 4 + np.arange(R))
        size = BASE_SIZE * (1 + truth.growth_per_year) ** (t / 4) * season
        capacity_path[t] = CAPACITY * (1 + truth.growth_per_year) ** (t / 4)
        game = Game(delta0=delta_bar + xi, alpha=truth.alpha, freight=FREIGHT[pf], cost=cost * cidx[pf],
                    product_firm=pf, capacity=capacity_path[t], size=size,
                    avail=avail, sigma=truth.sigma, nest=np.array([GRADES.index(g) for g in grades]))
        eq = solve(game, lam0=lam, mu0=mu, certify=False)
        lam, mu = eq.shadow, eq.markup
        noise = 1 + rng.normal(0, truth.measurement_noise, (J, R))
        for j in range(J):
            for r in range(R):
                if not avail[j, r]:
                    continue
                rows["t"].append(t); rows["r"].append(r); rows["j"].append(j)
                rows["p"].append(eq.prices[j, r]); rows["q"].append(eq.quantities[j, r] * noise[j, r])
                rows["m"].append(size[r]); rows["c"].append(cidx[pf[j]])

    market = Market(
        firms=list(FIRMS), leader=LEADER, capacity=CAPACITY * (1 + truth.growth_per_year) ** (periods / 4),
        products=names, product_firm=pf, product_grade=grades, cost=cost, regions=list(REGIONS),
        freight=FREIGHT.copy(),
        obs_period=np.array(rows["t"]), obs_region=np.array(rows["r"]), obs_product=np.array(rows["j"]),
        obs_price=np.array(rows["p"]), obs_quantity=np.array(rows["q"]), obs_size=np.array(rows["m"]),
        obs_cost_index=np.array(rows["c"]), source="synthetic demo (fictional producers, known ground truth)",
        capacity_path=capacity_path,
    )
    return market, truth
