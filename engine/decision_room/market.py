"""Market data model: products, regions, periods, and the CSV format users bring.

A *product* is one firm selling one grade. A *region* is an importing market
(a client cluster). Prices are FOB ($/t); buyers pay FOB + freight.

CSV layout of a market folder (see data/demo for a full example):

    products.csv     product,firm,grade,cost            cost = marginal FOB cost, $/t
    firms.csv        firm,capacity,is_leader            capacity in Mt per period
    regions.csv      region
    freight.csv      firm,region,freight               $/t
    observations.csv period,region,product,price,quantity,market_size,cost_index
    capacity.csv     period,firm,capacity                optional: capacity by period (else firms.csv)
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Market:
    firms: list[str]
    leader: str
    capacity: np.ndarray  # (F,) Mt per period
    products: list[str]
    product_firm: np.ndarray  # (J,) firm index
    product_grade: list[str]
    cost: np.ndarray  # (J,) marginal FOB cost, $/t
    regions: list[str]
    freight: np.ndarray  # (F, R) $/t
    # Panel of observations, long format, one row per (period, region, product).
    obs_period: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    obs_region: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    obs_product: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    obs_price: np.ndarray = field(default_factory=lambda: np.zeros(0))
    obs_quantity: np.ndarray = field(default_factory=lambda: np.zeros(0))
    obs_size: np.ndarray = field(default_factory=lambda: np.zeros(0))
    obs_cost_index: np.ndarray = field(default_factory=lambda: np.zeros(0))
    source: str = "unknown"
    capacity_path: dict[int, np.ndarray] | None = None  # period -> (F,) capacities, if known
    # Optional demand settings (settings.json) and extra excluded instruments (instruments.csv).
    time_effects: bool = False  # period fixed effects absorb market-wide demand cycles
    obs_instruments: np.ndarray | None = None  # (n_obs, k) supply shifters, one column per instrument
    instrument_names: list[str] = field(default_factory=list)
    period_unit: str = "quarter"  # wording only: "quarter" or "year"
    costs_source: str = "stated"  # "implied" when costs were backed out of prices (conduct test is then circular)

    @property
    def F(self) -> int:
        return len(self.firms)

    @property
    def J(self) -> int:
        return len(self.products)

    @property
    def R(self) -> int:
        return len(self.regions)

    @property
    def periods(self) -> np.ndarray:
        return np.unique(self.obs_period)

    @property
    def leader_index(self) -> int:
        return self.firms.index(self.leader)

    @property
    def grades(self) -> list[str]:
        return sorted(set(self.product_grade), key=self.product_grade.index)

    def capacity_at(self, t: int) -> np.ndarray:
        """Capacities in period t: from capacity.csv when given, else the firms.csv figures."""
        if self.capacity_path and t in self.capacity_path:
            return self.capacity_path[t].copy()
        return self.capacity.copy()

    def nest_ids(self) -> np.ndarray:
        """(J,) nest of each product for nested logit: products of the same grade share a nest."""
        return np.array([self.grades.index(g) for g in self.product_grade])

    def product_freight(self) -> np.ndarray:
        """(J, R) freight each product pays to reach each region."""
        return self.freight[self.product_firm]

    def firm_index(self, name: str) -> int:
        lowered = [f.lower() for f in self.firms]
        if name.lower() not in lowered:
            raise KeyError(f"unknown firm {name!r}; known: {', '.join(self.firms)}")
        return lowered.index(name.lower())

    def region_index(self, name: str) -> int:
        lowered = [r.lower() for r in self.regions]
        if name.lower() in lowered:
            return lowered.index(name.lower())
        # Loose match ("asia" -> "South Asia") so chat commands stay forgiving.
        hits = [i for i, r in enumerate(lowered) if name.lower() in r]
        if len(hits) == 1:
            return hits[0]
        raise KeyError(f"unknown region {name!r}; known: {', '.join(self.regions)}")

    def period_slice(self, t: int) -> dict[str, np.ndarray]:
        """Observations of one period as dense (J, R) arrays; missing = NaN."""
        mask = self.obs_period == t
        price = np.full((self.J, self.R), np.nan)
        qty = np.full((self.J, self.R), np.nan)
        size = np.full(self.R, np.nan)
        cidx = np.full(self.J, np.nan)
        j, r = self.obs_product[mask], self.obs_region[mask]
        price[j, r] = self.obs_price[mask]
        qty[j, r] = self.obs_quantity[mask]
        size[r] = self.obs_size[mask]
        cidx[j] = self.obs_cost_index[mask]
        return {"price": price, "quantity": qty, "size": size, "cost_index": cidx}

    def summary(self) -> dict:
        last = self.period_slice(int(self.periods.max()))
        q = np.nan_to_num(last["quantity"])
        firm_q = np.array([q[self.product_firm == f].sum() for f in range(self.F)])
        total = np.nansum(last["size"])
        return {
            "source": self.source,
            "firms": self.firms,
            "leader": self.leader,
            "regions": self.regions,
            "grades": self.grades,
            "products": self.J,
            "periods": int(len(self.periods)),
            "observations": int(len(self.obs_price)),
            "last_period": int(self.periods.max()),
            "period_unit": self.period_unit,
            "market_size_mt": round(float(total), 3),
            "firm_volume_mt": {f: round(float(v), 3) for f, v in zip(self.firms, firm_q)},
            "firm_share_pct": {f: round(100 * float(v) / total, 1) for f, v in zip(self.firms, firm_q)},
            "capacity_mt": {f: round(float(k), 3) for f, k in zip(self.firms, self.capacity)},
        }


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load_market(folder: str | Path) -> Market:
    folder = Path(folder)
    missing = [n for n in ("products.csv", "firms.csv", "regions.csv", "freight.csv", "observations.csv")
               if not (folder / n).exists()]
    if missing:
        raise FileNotFoundError(f"{folder} is missing {', '.join(missing)}")

    firms_rows = _read(folder / "firms.csv")
    firms = [r["firm"] for r in firms_rows]
    leaders = [r["firm"] for r in firms_rows if r.get("is_leader", "0").strip() in ("1", "true", "yes")]
    if len(leaders) != 1:
        raise ValueError("firms.csv must flag exactly one firm with is_leader=1 (the firm you advise)")
    capacity = np.array([float(r["capacity"]) for r in firms_rows])

    regions = [r["region"] for r in _read(folder / "regions.csv")]
    prod_rows = _read(folder / "products.csv")
    products = [r["product"] for r in prod_rows]
    product_firm = np.array([firms.index(r["firm"]) for r in prod_rows])
    grades = [r["grade"] for r in prod_rows]
    cost = np.array([float(r["cost"]) for r in prod_rows])

    freight = np.zeros((len(firms), len(regions)))
    for r in _read(folder / "freight.csv"):
        freight[firms.index(r["firm"]), regions.index(r["region"])] = float(r["freight"])

    obs = _read(folder / "observations.csv")
    capacity_path = None
    if (folder / "capacity.csv").exists():
        capacity_path = {}
        for r in _read(folder / "capacity.csv"):
            t = int(r["period"])
            capacity_path.setdefault(t, capacity.copy())[firms.index(r["firm"])] = float(r["capacity"])
    settings = {}
    if (folder / "settings.json").exists():
        settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    inst, inst_names = None, []
    if (folder / "instruments.csv").exists():
        rows = _read(folder / "instruments.csv")
        inst_names = [c for c in rows[0] if c not in ("period", "product")] if rows else []
        table = {(int(r["period"]), r["product"]): [float(r[c]) for c in inst_names] for r in rows}
        missing_rows = [(int(r["period"]), r["product"]) for r in obs
                        if (int(r["period"]), r["product"]) not in table]
        if missing_rows:
            raise ValueError(f"instruments.csv has no row for {len(missing_rows)} (period, product) pairs, "
                             f"e.g. {missing_rows[0]}")
        inst = np.array([table[(int(r["period"]), r["product"])] for r in obs])
    return Market(
        firms=firms, leader=leaders[0], capacity=capacity, products=products,
        product_firm=product_firm, product_grade=grades, cost=cost, regions=regions, freight=freight,
        obs_period=np.array([int(r["period"]) for r in obs]),
        obs_region=np.array([regions.index(r["region"]) for r in obs]),
        obs_product=np.array([products.index(r["product"]) for r in obs]),
        obs_price=np.array([float(r["price"]) for r in obs]),
        obs_quantity=np.array([float(r["quantity"]) for r in obs]),
        obs_size=np.array([float(r["market_size"]) for r in obs]),
        obs_cost_index=np.array([float(r["cost_index"]) for r in obs]),
        source=str(folder), capacity_path=capacity_path,
        time_effects=bool(settings.get("time_effects", False)),
        obs_instruments=inst, instrument_names=inst_names,
        costs_source=str(settings.get("costs", "stated")),
        period_unit=str(settings.get("period_unit", "quarter")),
    )


def save_market(m: Market, folder: str | Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    def write(name: str, header: list[str], rows: list[list]) -> None:
        with (folder / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(header)
            w.writerows(rows)

    write("firms.csv", ["firm", "capacity", "is_leader"],
          [[f, round(float(k), 4), int(f == m.leader)] for f, k in zip(m.firms, m.capacity)])
    write("regions.csv", ["region"], [[r] for r in m.regions])
    write("products.csv", ["product", "firm", "grade", "cost"],
          [[p, m.firms[m.product_firm[j]], m.product_grade[j], round(float(m.cost[j]), 2)]
           for j, p in enumerate(m.products)])
    write("freight.csv", ["firm", "region", "freight"],
          [[f, r, round(float(m.freight[i, k]), 2)] for i, f in enumerate(m.firms) for k, r in enumerate(m.regions)])
    write("observations.csv", ["period", "region", "product", "price", "quantity", "market_size", "cost_index"],
          [[int(m.obs_period[i]), m.regions[m.obs_region[i]], m.products[m.obs_product[i]],
            round(float(m.obs_price[i]), 3), round(float(m.obs_quantity[i]), 6),
            round(float(m.obs_size[i]), 4), round(float(m.obs_cost_index[i]), 4)]
           for i in range(len(m.obs_price))])
    settings = {k: v for k, v in (("time_effects", m.time_effects), ("costs", m.costs_source),
                                                    ("period_unit", m.period_unit))
                if v not in (False, "stated", "quarter")}
    if settings:
        (folder / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    if m.obs_instruments is not None:
        seen = {}
        for i in range(len(m.obs_price)):
            seen[(int(m.obs_period[i]), m.products[m.obs_product[i]])] = m.obs_instruments[i]
        write("instruments.csv", ["period", "product", *m.instrument_names],
              [[t, p, *[round(float(v), 6) for v in z]] for (t, p), z in sorted(seen.items())])
    if m.capacity_path:
        write("capacity.csv", ["period", "firm", "capacity"],
              [[t, f, round(float(k[i]), 4)] for t, k in sorted(m.capacity_path.items()) for i, f in enumerate(m.firms)])
