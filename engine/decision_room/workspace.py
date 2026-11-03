"""A stateful analysis session: the object the ACP agent drives, one call per specialist.

Each public method is one specialist's job and returns plain JSON-able dicts,
so the same workspace serves the JSON-RPC server, the CLI and the tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from . import demand as dm
from .units import relabel
from . import forecast, redteam, risk, strategy
from .market import Market, load_market, save_market
from .synth import generate

Progress = Callable[[str], None]


class Workspace:
    def __init__(self, progress: Progress | None = None):
        self.progress = progress or (lambda _msg: None)
        self.market: Market | None = None
        self.truth: dict | None = None
        self.model: dm.DemandModel | None = None
        self.backtest_result: dict | None = None
        self.rival_model: dm.DemandModel | None = None  # the demand model not selected, for comparison
        self.model_kind: str = "auto"
        self.nash: strategy.Outcome | None = None
        self.status_quo: strategy.Outcome | None = None
        self.recommended: strategy.Outcome | None = None
        self.rec_rules: dict = {}
        self.rec_info: dict = {}
        self.mc: dict | None = None
        self.frontier: dict | None = None
        self.rule_price: dict | None = None
        self.whatifs: list[dict] = []
        self.redteam: dict | None = None

    # ---- persistence ---------------------------------------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("progress", None)  # callbacks belong to the live process
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.progress = lambda _msg: None

    def save_state(self, path: str) -> dict:
        """Snapshot the whole workspace (market, models, solutions) so a session can be reopened."""
        import pickle
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(path).with_suffix(".tmp")
        tmp.write_bytes(pickle.dumps(self, protocol=pickle.HIGHEST_PROTOCOL))
        tmp.replace(path)
        return {"saved": str(path), "bytes": Path(path).stat().st_size}

    def restore_state(self, path: str) -> dict:
        """Reload a snapshot written by save_state. Only ever reads files the agent itself wrote."""
        import pickle
        other = pickle.loads(Path(path).read_bytes())
        progress = self.progress
        self.__dict__.update(other.__dict__)
        self.progress = progress
        return self.context()

    # ---- guards -----------------------------------------------------------
    def _need_market(self) -> Market:
        if self.market is None:
            self.load()
        return self.market

    def _need_model(self) -> dm.DemandModel:
        if self.model is None:
            self.fit_demand()
        return self.model

    def _game(self, scenario: strategy.Scenario | None = None):
        return strategy.build_game(self._need_market(), self._need_model(), scenario)

    def _need_recommendation(self) -> strategy.Outcome:
        if self.recommended is None:
            self.recommend()
        return self.recommended

    # ---- specialists --------------------------------------------------------
    def load(self, source: str = "demo") -> dict:
        """Data steward: load a market folder, or build the synthetic demo."""
        if source in ("demo", "", None):
            self.progress("Generating the synthetic demo market (fictional producers, known ground truth)")
            self.market, truth = generate()
            self.truth = truth.as_dict()
        else:
            self.progress(f"Loading market files from {source}")
            self.market = load_market(source)
            self.truth = None
        self.model = self.rival_model = self.backtest_result = self.nash = self.status_quo = self.recommended = None
        self.model_kind = "auto"
        self.mc = self.frontier = self.rule_price = self.redteam = None
        self.whatifs = []
        out = self.market.summary()
        out["checks"] = self._data_checks()
        return out

    def _data_checks(self) -> list[str]:
        m = self.market
        notes = []
        inside = {}
        for t, r, q, s in zip(m.obs_period, m.obs_region, m.obs_quantity, m.obs_size):
            inside[(t, r)] = inside.get((t, r), 0.0) + q / s
        worst = max(inside.values())
        notes.append(f"Inside shares peak at {100 * worst:.0f}% of a region (outside option always present).")
        if np.any(m.obs_price <= 0) or np.any(m.obs_quantity <= 0):
            notes.append("WARNING: non-positive prices or quantities found.")
        last = m.period_slice(int(m.periods.max()))
        q = np.nan_to_num(last["quantity"])
        for f, name in enumerate(m.firms):
            used = q[m.product_firm == f].sum() / m.capacity[f]
            if used > 1.0:
                notes.append(f"{name} reports {100 * used:.1f}% of its stated capacity last quarter: measurement noise "
                             f"in the volumes or understated capacity. The model treats capacity as binding.")
        expected = len(m.periods) * int((~np.isnan(last["price"])).sum())
        notes.append(f"{len(m.obs_price)} of {expected} expected product-region-quarter rows present.")
        return notes

    def export(self, folder: str) -> dict:
        """Write the current market (the demo by default) as CSV files users can copy."""
        m = self._need_market()
        save_market(m, folder)
        return {"saved_to": str(Path(folder).resolve()), "files": sorted(p.name for p in Path(folder).glob("*.csv"))}

    def fit_demand(self, window: int | None = None, model: str | None = None, quiet: bool = False) -> dict:
        """Demand economist: Berry inversion, OLS vs 2SLS, plain vs nested logit.

        model: "logit", "nested", or None/"auto" (keep the session's choice, or pick nested
        logit when its nesting parameter is significant).
        """
        m = self._need_market()
        periods = m.periods if window is None else m.periods[-window:]
        if not quiet:
            self.progress(f"Estimating demand on {len(periods)} periods: plain and nested logit, OLS and 2SLS")
        nested = dm.fit(m, periods, nested=True)
        plain = dm.fit(m, periods)
        choice = model if model in ("logit", "nested") else self.model_kind
        if choice not in ("logit", "nested"):
            choice = "nested" if nested.sigma - 1.96 * nested.sigma_se > 0.02 else "logit"
        self.model, self.rival_model = (nested, plain) if choice == "nested" else (plain, nested)
        self.model_kind = choice
        self.nash = self.status_quo = self.recommended = self.mc = self.frontier = self.redteam = None
        out = self.model.to_dict()
        out["alternative"] = {"kind": self.rival_model.kind, "alpha": round(self.rival_model.alpha, 5),
                              "sigma": round(self.rival_model.sigma, 4)}
        game = self._game()
        from .equilibrium import elasticities
        sl = m.period_slice(int(m.periods.max()))
        el = elasticities(game, np.nan_to_num(sl["price"]))
        out["own_price_elasticity_by_firm"] = {
            f: round(float(np.nanmean(el[m.product_firm == i])), 2) for i, f in enumerate(m.firms)}
        if self.truth:
            out["ground_truth_alpha"] = self.truth["alpha"]
            out["ground_truth_sigma"] = self.truth.get("sigma")
        return out

    def backtest(self) -> dict:
        """Demand economist: rolling out-of-sample backtest across window lengths."""
        m = self._need_market()
        self.progress("Forecasting the last periods' equilibrium out of sample with plain logit")
        plain = forecast.backtest(m, nested=False)
        self.progress("Forecasting the last periods' equilibrium out of sample with nested logit")
        nested = forecast.backtest(m, nested=True)
        # Pick the demand model: nested only if its nesting is significant on the full sample
        # and it forecasts at least as well out of sample (parsimony otherwise).
        full = dm.fit(m, nested=True)
        sig_ok = full.sigma - 1.96 * full.sigma_se > 0.02
        better = nested["best"]["mae_share_pts"] <= plain["best"]["mae_share_pts"]
        kind = "nested" if sig_ok and better else "logit"
        reason = ("nesting is significant (sigma {:.2f}) and nested logit forecasts better out of sample ({:.3f} vs "
                  "{:.3f} share pts)".format(full.sigma, nested["best"]["mae_share_pts"], plain["best"]["mae_share_pts"])
                  if kind == "nested" else
                  "nested logit adds no significant within-grade substitution or does not forecast better")
        chosen = nested if kind == "nested" else plain
        self.backtest_result = dict(chosen)
        self.backtest_result.update({
            "selected_model": kind, "selection_reason": reason,
            "comparison": {"logit": plain["best"], "nested": nested["best"]},
        })
        self.model_kind = kind
        keep = chosen["retained_window"] or len(m.periods)
        self.fit_demand(window=keep, model=kind, quiet=True)  # the agent asks for the summary next and logs it
        return self.backtest_result

    def equilibrium(self) -> dict:
        """Game theorist: capacity-constrained Bertrand-Nash with KKT certificate."""
        m = self._need_market()
        game = self._game()
        self.progress("Solving capacity-constrained Bertrand-Nash (inner markups, outer shadow prices)")
        self.nash = strategy.nash(game, m.leader_index)
        self.status_quo = strategy.status_quo(m, game)
        out = self.nash.eq.to_dict(m.firms, m.regions, m.products, m.product_firm)
        out["leader_at_nash"] = self.nash.leader_view(m, game)
        out["leader_status_quo"] = self.status_quo.leader_view(m, game)
        return out

    def recommend(self, max_region_share: float | None = None, min_market_share: float | None = None,
                  robust: bool = False, worlds: int = 16, risk_aversion: float = 0.0) -> dict:
        """Strategist: Stackelberg leader prices under the firm's commercial rules.

        robust=True optimises the average profit over sampled worlds (parameters, shocks, sizes)
        instead of the single expected world; risk_aversion > 0 also penalises the spread.
        """
        m = self._need_market()
        game = self._game()
        if self.nash is None:
            self.equilibrium()
        if robust:
            self.progress(f"Sampling {worlds} worlds from the demand estimates for a robust decision")
            sampled = risk.sample_worlds(m, self._need_model(), worlds, seed=97)
            self.progress(f"Optimising our prices on average over {worlds} worlds (each step re-solves rivals {worlds}x)")
            self.recommended, self.rec_info = strategy.robust_optimise(
                game, sampled, m.leader_index, risk_aversion=risk_aversion,
                max_region_share=max_region_share, min_market_share=min_market_share)
        else:
            self.progress("Optimising our prices while anticipating rivals' response (Stackelberg)")
            self.recommended, self.rec_info = strategy.optimise(
                game, m.leader_index, max_region_share=max_region_share, min_market_share=min_market_share)
        self.rec_rules = {"max_region_share": max_region_share, "min_market_share": min_market_share,
                          "robust": robust, "risk_aversion": risk_aversion if robust else None}
        self.mc = self.redteam = None
        view = self.recommended.leader_view(m, game)
        sq = self.status_quo.leader_view(m, game)
        nv = self.nash.leader_view(m, game)
        return {
            "rules": self.rec_rules,
            "optimiser": self.rec_info,
            "recommended": view,
            "status_quo": sq,
            "nash": nv,
            "uplift_vs_status_quo_musd": round(view["profit_musd"] - sq["profit_musd"], 3),
            "uplift_vs_status_quo_pct": round(100 * (view["profit_musd"] / sq["profit_musd"] - 1), 2),
            "price_moves_vs_status_quo": {r: round(view["by_region"][r]["price"] - sq["by_region"][r]["price"], 2)
                                          for r in m.regions},
        }

    def conduct(self) -> dict:
        """Game theorist: which market behaviour do observed prices support?"""
        m = self._need_market()
        if self.nash is None:
            self.equilibrium()
        self.progress("Inverting last quarter's prices into implied costs under three conducts")
        out = redteam.conduct_test(m, self._need_model(), self.nash.eq)
        out["labels"] = redteam.CONDUCTS
        return out

    def price_rule(self, max_region_share: float | None = None, min_market_share: float | None = None) -> dict:
        """Strategist: what a commercial-security rule costs."""
        m = self._need_market()
        self.progress("Pricing the rule: optimum with and without it, then one point tighter")
        self.rule_price = strategy.price_constraint(self._game(), m.leader_index, max_region_share, min_market_share)
        return self.rule_price

    def tradeoff_frontier(self) -> dict:
        """Strategist: profit vs concentration cap, point by point."""
        m = self._need_market()
        self.progress("Tracing the profit vs concentration frontier (7 optimisations)")
        self.frontier = strategy.frontier(self._game(), m.leader_index)
        return self.frontier

    def whatif(self, spec: dict) -> dict:
        """Risk analyst: a scenario, with and without our reaction."""
        m = self._need_market()
        sc = strategy.Scenario(**spec)
        self.progress(f"Re-solving the market under scenario: {sc.name}")
        out = risk.whatif(m, self._need_model(), sc)
        self.whatifs.append(out)
        return out

    def stress(self, draws: int = 200) -> dict:
        """Risk analyst: Monte Carlo of the recommendation against the status quo."""
        m = self._need_market()
        rec = self._need_recommendation()
        if self.status_quo is None:
            self.equilibrium()
        self.progress(f"Stress test: {draws} joint draws of parameters, demand shocks and market sizes")
        self.mc = risk.monte_carlo(m, self._need_model(), rec.leader_markup, self.status_quo.leader_markup,
                                   draws=draws, progress=self.progress)
        return self.mc

    def red_team(self) -> dict:
        """Red team: attack the recommendation before anyone sees it."""
        m = self._need_market()
        if self.backtest_result is None:
            self.backtest()
        rec = self._need_recommendation()
        if self.mc is None:
            self.stress()
        self.progress("Red team: running adversarial checks on data, model, solver and decision")
        self.redteam = redteam.run(m, self._need_model(), self.backtest_result, self.nash.eq, rec, self.mc, self.truth,
                                   rival_model=self.rival_model)
        return self.redteam

    # ---- memo ---------------------------------------------------------------
    def brief_data(self) -> dict:
        """Everything the memo needs, computing whatever is still missing."""
        m = self._need_market()
        if self.backtest_result is None:
            self.backtest()  # may refit demand, which resets everything downstream
        rec = self._need_recommendation()
        if self.frontier is None:
            self.tradeoff_frontier()
        if self.mc is None:
            self.stress()
        if self.redteam is None:
            self.red_team()
        game = self._game()
        return {
            "market": m.summary(), "demand": self.model.to_dict(), "backtest": self.backtest_result,
            "recommended": rec.leader_view(m, game), "status_quo": self.status_quo.leader_view(m, game),
            "nash": self.nash.leader_view(m, game), "frontier": self.frontier, "rule": self.rule_price,
            "monte_carlo": self.mc, "redteam": self.redteam, "whatifs": self.whatifs,
        }

    def memo_preview(self) -> dict:
        from . import memo
        b = self.brief_data()
        u = self.market.period_unit
        return {"markdown": relabel(memo.markdown(b), u), "headline": relabel(memo.headline(b), u),
                "grade": b["redteam"]["grade"]}

    def write_memo(self, out_dir: str) -> dict:
        """Memo writer: Markdown + HTML + standalone SVG charts in out_dir."""
        from . import memo
        b = self.brief_data()
        self.progress("Writing the decision memo (Markdown, HTML and charts)")
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        files = {}
        u = self.market.period_unit
        for name, svg in memo.charts_for(b).items():
            (out / f"{name}.svg").write_text(relabel(svg, u), encoding="utf-8")
            files[name] = f"{name}.svg"
        md = relabel(memo.markdown(b, files), u)
        (out / "memo.md").write_text(md, encoding="utf-8")
        (out / "memo.html").write_text(relabel(memo.html(b), u), encoding="utf-8")
        return {"markdown": md, "headline": relabel(memo.headline(b), u), "grade": b["redteam"]["grade"],
                "dir": str(out.resolve()), "files": ["memo.md", "memo.html", *files.values()]}

    def chart(self, kind: str) -> dict:
        """One SVG chart from what is already computed: prices, frontier, risk or backtest."""
        from . import charts
        m = self._need_market()
        if kind == "frontier" and self.frontier:
            pts = [p for p in self.frontier["points"] if p["feasible"]]
            svg = charts.line_chart("Profit as the cap on any one region's share of our volume tightens",
                                    [p["cap_pct"] for p in pts], [p["profit_musd"] for p in pts],
                                    "cap on top region's share of our volume (%)", "$M per quarter",
                                    annotate=[f"-{p['give_up_pct']:.1f}%" if p["give_up_pct"] > 0.05 else "" for p in pts],
                                    reverse_x=True)
        elif kind == "risk" and self.mc:
            svg = charts.histogram("Uplift versus status quo across simulated futures ($M per quarter)",
                                   self.mc["histogram"]["counts"], self.mc["histogram"]["edges"], "$M per quarter")
        elif kind == "prices" and self.recommended and self.status_quo:
            game = self._game()
            rec, sq = self.recommended.leader_view(m, game), self.status_quo.leader_view(m, game)
            served = [r for r in m.regions
                      if rec["by_region"][r]["volume_mt"] > 0 or sq["by_region"][r]["volume_mt"] > 0]
            svg = charts.grouped_bars("Our price by region ($/t FOB)", served,
                                      {"Status quo": [sq["by_region"][r]["price"] for r in served],
                                       "Recommended": [rec["by_region"][r]["price"] for r in served]},
                                      "$/t", highlight=1)
        elif kind == "backtest" and self.backtest_result:
            w = self.backtest_result["windows"]
            svg = charts.grouped_bars("Out-of-sample equilibrium forecast: share error by calibration window",
                                      [f"{x['window']}q" for x in w],
                                      {"Model": [x["mae_share_pts"] for x in w],
                                       "Naive (last quarter)": [x["naive_mae_share_pts"] for x in w]},
                                      "MAE, share points", zero_based=True, fmt="{:.2f}")
        else:
            return {"kind": kind, "svg": None}
        return {"kind": kind, "svg": relabel(svg, m.period_unit)}

    def context(self) -> dict:
        """What has been computed so far: lets the agent answer follow-ups without recomputing."""
        m = self.market
        out = {"market_loaded": m is not None, "demand_fitted": self.model is not None,
               "recommendation": self.recommended is not None, "stress_tested": self.mc is not None,
               "red_team_grade": None if self.redteam is None else self.redteam["grade"],
               "whatifs": [w["scenario"].get("name") for w in self.whatifs]}
        if m is not None:
            out["firms"], out["regions"], out["leader"] = m.firms, m.regions, m.leader
            out["period_unit"] = m.period_unit
        return out

    def brief(self, out_dir: str | None = None, max_region_share: float | None = None,
              draws: int = 200) -> dict:
        """The whole war room, end to end."""
        self._need_market()
        backtest = self.backtest()  # picks the calibration window and fits demand on it
        steps = {
            "market": self.market.summary(),
            "demand": self.model.to_dict(),
            "backtest": backtest,
            "equilibrium": self.equilibrium(),
            "recommendation": self.recommend(max_region_share=max_region_share),
            "frontier": self.tradeoff_frontier(),
            "stress": self.stress(draws),
            "red_team": self.red_team(),
        }
        if out_dir:
            steps["memo"] = self.write_memo(out_dir)
        return steps
