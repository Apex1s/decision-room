"""Leader-follower pricing: what should the firm we advise charge, and what do
its own commercial-security rules cost?

The leader commits to its prices first; followers then play the
capacity-constrained Bertrand-Nash game among themselves; the leader
optimises anticipating that response (Stackelberg).

Decision variables. The leader moves one price level per region: its markups
are the competitive (Nash) markups of each grade plus a regional shift. This
keeps the grade spreads that demand supports and leaves one lever per region,
which is how regional pricing committees actually decide. Six regions mean
six variables, so every optimisation re-solves the rivals' game a few hundred
times in seconds.

Robust mode optimises the average profit over a set of sampled worlds
(sample-average approximation), optionally penalised by its standard
deviation, instead of the profit in the single most likely world.

Commercial-security rules, all optional:
  max_region_share  no region may take more than this fraction of our volume
  min_market_share  our share of total market (incl. outside option) stays above this
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .demand import DemandModel
from .equilibrium import Equilibrium, Game, solve
from .market import Market


@dataclass
class Scenario:
    """Edits applied on top of the next-period baseline game."""

    name: str = "baseline"
    capacity_change: dict | None = None  # firm -> Mt added (negative = outage)
    demand_change: dict | None = None  # region -> fractional change in market size
    cost_change: dict | None = None  # firm -> fractional change in marginal cost
    freight_change: dict | None = None  # region -> $/t added for every seller
    alpha_scale: float = 1.0  # buyers' price sensitivity multiplier

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if v not in (None, 1.0) or k == "name"}


def build_game(m: Market, model: DemandModel, scenario: Scenario | None = None, alpha: float | None = None,
               delta0: np.ndarray | None = None, sigma: float | None = None) -> Game:
    t = int(m.periods.max())
    sl = m.period_slice(t)
    avail = ~np.isnan(sl["price"])
    cidx = np.nan_to_num(sl["cost_index"], nan=1.0)
    size = sl["size"].copy()
    cost = m.cost * cidx
    capacity = m.capacity.copy()
    freight = m.product_freight().copy()
    a = model.alpha if alpha is None else alpha
    d0 = model.delta0() if delta0 is None else delta0
    if scenario:
        for f, dk in (scenario.capacity_change or {}).items():
            capacity[m.firm_index(f)] = max(1e-3, capacity[m.firm_index(f)] + dk)
        for r, ds in (scenario.demand_change or {}).items():
            size[m.region_index(r)] *= 1 + ds
        for f, dc in (scenario.cost_change or {}).items():
            cost[m.product_firm == m.firm_index(f)] *= 1 + dc
        for r, dfr in (scenario.freight_change or {}).items():
            freight[:, m.region_index(r)] += dfr
        a *= scenario.alpha_scale
    nest = model.nest if model.nest is not None else np.zeros(m.J, int)
    return Game(delta0=d0, alpha=a, freight=freight, cost=cost, product_firm=m.product_firm,
                capacity=capacity, size=size, avail=avail, sigma=model.sigma if sigma is None else sigma, nest=nest)


def leader_prices(game: Game, leader: int, markups: np.ndarray) -> np.ndarray:
    """Full (J, R) price matrix with the leader's rows set to cost + markups (n_leader_products, R)."""
    prices = np.zeros_like(game.delta0)
    mine = game.product_firm == leader
    prices[mine] = game.cost[mine][:, None] + markups
    return prices


@dataclass
class Outcome:
    label: str
    eq: Equilibrium
    leader_markup: np.ndarray  # (n_leader_products, R): price minus marginal cost
    shift: np.ndarray | None = None  # (R,) regional shift over the Nash markups, when optimised

    def leader_view(self, m: Market, game: Game) -> dict:
        L = m.leader_index
        mine = game.product_firm == L
        q = np.where(game.avail, self.eq.quantities, 0.0)
        reg_q = q[mine].sum(axis=0)
        vol = reg_q.sum()
        total = game.size.sum()
        avg_price = (self.eq.prices[mine] * q[mine]).sum(axis=0) / np.maximum(reg_q, 1e-12)
        avg_markup = (self.leader_markup * q[mine]).sum(axis=0) / np.maximum(reg_q, 1e-12)
        grades = [m.product_grade[j] for j in np.where(mine)[0]]
        return {
            "label": self.label,
            "profit_musd": round(float(self.eq.profit[L]), 3),
            "volume_mt": round(float(vol), 4),
            "capacity_use_pct": round(100 * float(vol / game.capacity[L]), 1),
            "market_share_pct": round(100 * float(vol / total), 2),
            "max_region_share_pct": round(100 * float(reg_q.max() / max(vol, 1e-12)), 1),
            "top_region": m.regions[int(np.argmax(reg_q))],
            "by_region": {
                r: {"price": round(float(avg_price[k]), 2), "volume_mt": round(float(reg_q[k]), 4),
                    "share_of_our_volume_pct": round(100 * float(reg_q[k] / max(vol, 1e-12)), 1),
                    "markup": round(float(avg_markup[k]), 2),
                    "by_grade": {g: round(float(self.eq.prices[j, k]), 2)
                                 for g, j in zip(grades, np.where(mine)[0]) if game.avail[j, k]}}
                for k, r in enumerate(m.regions)
            },
            "rivals_profit_musd": {m.firms[f]: round(float(self.eq.profit[f]), 3)
                                   for f in range(m.F) if f != L},
            "rivals_shadow_price": {m.firms[f]: round(float(self.eq.shadow[f]), 2)
                                    for f in range(m.F) if f != L},
        }


def respond(game: Game, leader: int, markups: np.ndarray, warm: dict | None = None,
            certify: bool = False) -> Equilibrium:
    """Followers' equilibrium given the leader's committed markups."""
    fixed = leader_prices(game, leader, markups)
    eq = solve(game, fixed_firms=[leader], fixed_prices=fixed, certify=certify,
               lam0=None if warm is None else warm.get("lam"), mu0=None if warm is None else warm.get("mu"))
    if warm is not None:
        warm["lam"], warm["mu"] = eq.shadow, eq.markup
    return eq


def _region_volume(game: Game, eq: Equilibrium, leader: int) -> np.ndarray:
    mine = game.product_firm == leader
    return np.where(game.avail, eq.quantities, 0.0)[mine].sum(axis=0)


def nash(game: Game, leader: int) -> Outcome:
    eq = solve(game)
    mine = game.product_firm == leader
    return Outcome("Nash (everyone best-responds)", eq, eq.prices[mine] - game.cost[mine][:, None],
                   np.zeros(game.delta0.shape[1]))


def status_quo(m: Market, game: Game) -> Outcome:
    """Leader keeps last period's observed prices; rivals re-optimise."""
    L = m.leader_index
    sl = m.period_slice(int(m.periods.max()))
    mine = game.product_firm == L
    observed = sl["price"][mine] - game.cost[mine][:, None]
    if np.isnan(observed[game.avail[mine]]).any():
        # A grade not sold last quarter keeps its competitive markup.
        observed = np.where(np.isnan(observed), nash(game, L).leader_markup, observed)
    markups = np.nan_to_num(observed)
    eq = respond(game, L, markups, certify=True)
    return Outcome("Status quo (keep last quarter's prices)", eq, markups)


def _optimise(games: list[Game], leader: int, base: np.ndarray, max_region_share: float | None,
              min_market_share: float | None, start: np.ndarray | None, risk_aversion: float = 0.0):
    """Shared SLSQP core: maximise mean (minus risk_aversion x sd) profit over games."""
    g0 = games[0]
    R = g0.delta0.shape[1]
    scale = 1.0 / g0.alpha  # shifts live around 1/alpha; optimise in those units
    warms = [dict() for _ in games]
    cache: dict[bytes, list[Equilibrium]] = {}
    floor = 1.0 - base.min(axis=0)  # keep every markup at least $1/t

    def eqs_at(x: np.ndarray) -> list[Equilibrium]:
        key = np.round(x, 10).tobytes()
        if key not in cache:
            cache[key] = [respond(g, leader, base + x[None, :] * scale, w) for g, w in zip(games, warms)]
        return cache[key]

    def profits(x):
        return np.array([e.profit[leader] for e in eqs_at(x)])

    def objective(x):
        p = profits(x)
        return -(p.mean() - risk_aversion * (p.std() if len(p) > 1 else 0.0))

    def vols(x):
        return np.mean([_region_volume(g, e, leader) for g, e in zip(games, eqs_at(x))], axis=0)

    cons = [{"type": "ineq", "fun": lambda x: g0.capacity[leader] - vols(x).sum()}]
    if max_region_share is not None:
        cons.append({"type": "ineq", "fun": lambda x: max_region_share * vols(x).sum() - vols(x)})
    if min_market_share is not None:
        cons.append({"type": "ineq", "fun": lambda x: vols(x).sum() - min_market_share * g0.size.sum()})
    x0 = np.zeros(R) if start is None else start / scale
    bounds = [(fl / scale, 2.0) for fl in floor]
    x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
    res = minimize(objective, x0, method="SLSQP", constraints=cons, bounds=bounds,
                   options={"maxiter": 200, "ftol": 1e-9, "eps": 1e-3})
    shift = res.x * scale
    vol = vols(res.x)
    feasible = (vol.sum() <= g0.capacity[leader] * (1 + 1e-4)
                and (max_region_share is None or vol.max() <= max_region_share * vol.sum() * (1 + 1e-3))
                and (min_market_share is None or vol.sum() >= min_market_share * g0.size.sum() * (1 - 1e-3)))
    info = {"converged": bool(res.success), "feasible": bool(feasible), "message": str(res.message),
            "iterations": int(res.nit), "evaluations": len(cache) * len(games)}
    return shift, info, profits(res.x)


def optimise(game: Game, leader: int, max_region_share: float | None = None,
             min_market_share: float | None = None, start: np.ndarray | None = None,
             label: str = "Recommended (Stackelberg leader)") -> tuple[Outcome, dict]:
    """Best regional price shifts for the leader in the expected world. `start` is a shift vector (R,)."""
    base = nash(game, leader).leader_markup
    shift, info, _ = _optimise([game], leader, base, max_region_share, min_market_share, start)
    markups = base + shift[None, :]
    eq = respond(game, leader, markups, certify=True)
    return Outcome(label, eq, markups, shift), info


def robust_optimise(game: Game, worlds: list[Game], leader: int, risk_aversion: float = 0.0,
                    max_region_share: float | None = None, min_market_share: float | None = None,
                    start: np.ndarray | None = None) -> tuple[Outcome, dict]:
    """Best shifts on average over sampled worlds (sample-average approximation)."""
    base = nash(game, leader).leader_markup
    shift, info, profits = _optimise(worlds, leader, base, max_region_share, min_market_share, start, risk_aversion)
    markups = base + shift[None, :]
    eq = respond(game, leader, markups, certify=True)
    info.update({"worlds": len(worlds), "risk_aversion": risk_aversion,
                 "mean_profit_musd": round(float(profits.mean()), 3), "sd_profit_musd": round(float(profits.std()), 3),
                 "worst_profit_musd": round(float(profits.min()), 3)})
    label = "Robust (best on average over sampled worlds)" if risk_aversion == 0 else \
        f"Robust (mean - {risk_aversion:g} x sd over sampled worlds)"
    return Outcome(label, eq, markups, shift), info


def frontier(game: Game, leader: int, caps: list[float] | None = None, min_market_share: float | None = None) -> dict:
    """Profit given up as the cap on any single region's share of our volume tightens."""
    base, _ = optimise(game, leader, min_market_share=min_market_share)
    vol0 = _region_volume(game, base.eq, leader)
    top0 = float(vol0.max() / vol0.sum())
    R = game.delta0.shape[1]
    floor = 1.0 / R
    if caps is None:
        caps = list(np.round(np.linspace(top0, floor + 0.02, 7), 4))
    points = []
    start = base.shift
    for cap in caps:
        if cap >= top0 - 1e-9:
            out, info = base, {"feasible": True}
        else:
            out, info = optimise(game, leader, max_region_share=cap, min_market_share=min_market_share, start=start)
            start = out.shift
        vol = _region_volume(game, out.eq, leader)
        points.append({"cap_pct": round(100 * cap, 2), "profit_musd": round(float(out.eq.profit[leader]), 3),
                       "give_up_pct": round(100 * (1 - out.eq.profit[leader] / base.eq.profit[leader]), 3),
                       "volume_mt": round(float(vol.sum()), 4), "feasible": info["feasible"],
                       "top_region_share_pct": round(100 * float(vol.max() / vol.sum()), 2)})
    for a, b in zip(points, points[1:]):
        dcap = a["cap_pct"] - b["cap_pct"]
        b["marginal_cost_musd_per_pt"] = round((a["profit_musd"] - b["profit_musd"]) / dcap, 4) if dcap else None
    return {"unconstrained_top_share_pct": round(100 * top0, 2), "points": points}


def price_constraint(game: Game, leader: int, max_region_share: float | None = None,
                     min_market_share: float | None = None) -> dict:
    free, _ = optimise(game, leader)
    tied, info = optimise(game, leader, max_region_share=max_region_share, min_market_share=min_market_share,
                          start=free.shift)
    cost = float(free.eq.profit[leader] - tied.eq.profit[leader])
    # Marginal cost of tightening by one more point (shadow price of the rule).
    marginal = None
    if max_region_share is not None and info["feasible"]:
        tighter, i2 = optimise(game, leader, max_region_share=max_region_share - 0.01,
                               min_market_share=min_market_share, start=tied.shift)
        if i2["feasible"]:
            marginal = round(float(tied.eq.profit[leader] - tighter.eq.profit[leader]), 4)
    elif min_market_share is not None and info["feasible"]:
        tighter, i2 = optimise(game, leader, max_region_share=max_region_share,
                               min_market_share=min_market_share + 0.01, start=tied.shift)
        if i2["feasible"]:
            marginal = round(float(tied.eq.profit[leader] - tighter.eq.profit[leader]), 4)
    return {
        "rule": {"max_region_share": max_region_share, "min_market_share": min_market_share},
        "profit_free_musd": round(float(free.eq.profit[leader]), 3),
        "profit_with_rule_musd": round(float(tied.eq.profit[leader]), 3),
        "cost_musd_per_period": round(cost, 3),
        "cost_pct": round(100 * cost / float(free.eq.profit[leader]), 2),
        "marginal_cost_next_point_musd": marginal,
        "feasible": info["feasible"],
    }
