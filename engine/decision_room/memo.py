"""The decision memo: one page a pricing committee can act on.

Built from a brief (see Workspace.brief). Markdown for the editor, HTML with
inline charts for the board pack. Numbers come from the brief only, so the
memo can never say something the engine did not compute.
"""

from __future__ import annotations

from html import escape

from . import charts

ICON = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}


def _money(v: float) -> str:
    return f"${v:,.1f}M"


def headline(b: dict) -> str:
    rec, sq, mc, rt = b["recommended"], b["status_quo"], b.get("monte_carlo"), b["redteam"]
    up = rec["profit_musd"] - sq["profit_musd"]
    win = f", winning in {mc['prob_recommended_wins_pct']:.1f}% of simulated futures" if mc else ""
    return (f"Reprice {b['market']['leader']} region by region: {_money(up)} per quarter versus keeping last "
            f"quarter's prices ({100 * up / sq['profit_musd']:+.1f}%){win}. Red-team grade {rt['grade']}.")


def served(rec: dict, sq: dict) -> list[tuple[str, dict]]:
    """Regions we sell to in either plan; the rest have no route in the data."""
    return [(r, v) for r, v in rec["by_region"].items()
            if v["volume_mt"] > 0 or sq["by_region"][r]["volume_mt"] > 0]


def charts_for(b: dict) -> dict[str, str]:
    out = {}
    # Regions we do not serve in either plan (no route in the data) would only show as empty bars.
    regions = [r for r in b["recommended"]["by_region"]
               if b["recommended"]["by_region"][r]["volume_mt"] > 0 or b["status_quo"]["by_region"][r]["volume_mt"] > 0]
    out["prices"] = charts.grouped_bars(
        "Our price by region ($/t FOB)", [r.replace(" ", " ") for r in regions],
        {"Status quo": [b["status_quo"]["by_region"][r]["price"] for r in regions],
         "Recommended": [b["recommended"]["by_region"][r]["price"] for r in regions]}, "$/t", highlight=1)
    fr = b.get("frontier")
    if fr:
        pts = [p for p in fr["points"] if p["feasible"]]
        out["frontier"] = charts.line_chart(
            "Profit as the cap on any one region's share of our volume tightens",
            [p["cap_pct"] for p in pts], [p["profit_musd"] for p in pts],
            "cap on top region's share of our volume (%)", "$M per quarter",
            annotate=[f"-{p['give_up_pct']:.1f}%" if p["give_up_pct"] > 0.05 else "" for p in pts], reverse_x=True)
    mc = b.get("monte_carlo")
    if mc:
        out["risk"] = charts.histogram("Uplift versus status quo across simulated futures ($M per quarter)",
                                       mc["histogram"]["counts"], mc["histogram"]["edges"], "$M per quarter")
    bt = b.get("backtest")
    if bt and bt["windows"]:
        out["backtest"] = charts.grouped_bars(
            "Out-of-sample equilibrium forecast: share error by calibration window (lower is better)",
            [f"{w['window']}q" for w in bt["windows"]],
            {"Model": [w["mae_share_pts"] for w in bt["windows"]],
             "Naive (last quarter)": [w["naive_mae_share_pts"] for w in bt["windows"]]},
            "MAE, share points", zero_based=True, fmt="{:.2f}")
    return out


def how_we_know(b: dict) -> list[str]:
    d, rec, bt, rt = b["demand"], b["recommended"], b.get("backtest"), b["redteam"]
    out = []
    sig = (f", within-grade nesting sigma {d['sigma']:.2f} (95% CI {d['sigma_ci95'][0]:.2f} to {d['sigma_ci95'][1]:.2f})"
           if d.get("kind") == "nested" else "")
    out.append(f"- **Demand:** {d['method']}. Price sensitivity {d['alpha']:.4f} per $/t "
               f"(95% CI {d['alpha_ci95'][0]:.4f} to {d['alpha_ci95'][1]:.4f}){sig}; first-stage F = {d['first_stage_f']}. "
               f"Plain OLS would have been {d['endogeneity_bias_pct']:+.1f}% off.")
    if bt and bt.get("selection_reason"):
        out.append(f"- **Model choice:** {bt['selected_model']} logit selected because {bt['selection_reason']}."
                   .replace("nested logit selected", "nested logit selected").replace("logit logit", "logit"))
    if bt and bt.get("best"):
        w = bt["best"]
        price = (f"; price error {w['mae_price_usd_t']:.2f} $/t vs {w['naive_mae_price_usd_t']:.2f} naive, price direction "
                 f"right {w['price_direction_pct']:.0f}% of the time" if "mae_price_usd_t" in w else "")
        out.append(f"- **Validation:** for each of quarters {bt['test_periods'][0]} to {bt['test_periods'][-1]}, the engine "
                   f"forecast the whole equilibrium (prices and shares) without seeing that quarter's demand shock. "
                   f"Share error {w['mae_share_pts']:.2f} pts vs {w['naive_mae_share_pts']:.2f} for \"same as last "
                   f"quarter\"{price}.")
    conduct = next((c for c in rt["checks"] if c["check"] == "Market conduct"), None)
    if conduct:
        out.append(f"- **Conduct:** {conduct['evidence']}")
    out.append("- **Competition:** capacity-constrained Bertrand-Nash among rivals (zeta-markup fixed point for nested "
               "logit, outer subgradient on capacity shadow prices), certified by KKT conditions. Rivals at capacity: "
               f"{', '.join(k for k, v in rec['rivals_shadow_price'].items() if v > 0.01) or 'none'}.")
    out.append("- **Our move:** Stackelberg leader: we commit to regional price levels (grade spreads kept at their "
               "competitive structure) anticipating rivals' response.\n")
    return out


def markdown(b: dict, chart_files: dict[str, str] | None = None) -> str:
    m, rec, sq, nash = b["market"], b["recommended"], b["status_quo"], b["nash"]
    d, rt, mc, fr, rule = b["demand"], b["redteam"], b.get("monte_carlo"), b.get("frontier"), b.get("rule")
    L = []
    L.append(f"# Decision memo: {m['leader']} pricing, next quarter\n")
    L.append(f"*Data: {m['source']}. {m['periods']} quarters, {m['observations']:,} observations, "
             f"{len(m['firms'])} producers, {len(m['regions'])} regions.*\n")
    L.append("## The decision\n")
    L.append(headline(b) + "\n")
    L.append("| Region | Status quo $/t | Recommended $/t | Move | Volume Mt | Share of our volume |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for r, v in served(rec, sq):
        s = sq["by_region"][r]
        L.append(f"| {r} | {s['price']:.1f} | {v['price']:.1f} | {v['price'] - s['price']:+.1f} | "
                 f"{v['volume_mt']:.3f} | {v['share_of_our_volume_pct']:.1f}% |")
    L.append("")
    L.append("| Outcome per quarter | Status quo | Nash | Recommended |")
    L.append("|---|---:|---:|---:|")
    for key, label, fmt in (("profit_musd", "Profit", "${:,.2f}M"), ("volume_mt", "Volume", "{:.3f} Mt"),
                            ("capacity_use_pct", "Capacity used", "{:.1f}%"),
                            ("market_share_pct", "Market share", "{:.2f}%"),
                            ("max_region_share_pct", "Largest region, % of our volume", "{:.1f}%")):
        L.append(f"| {label} | {fmt.format(sq[key])} | {fmt.format(nash[key])} | {fmt.format(rec[key])} |")
    L.append("")
    if chart_files and "prices" in chart_files:
        L.append(f"![Prices by region]({chart_files['prices']})\n")

    if fr or rule:
        L.append("## What the commercial-security rule costs\n")
        if rule:
            r = rule["rule"]
            what = (f"no region above {100 * r['max_region_share']:.0f}% of our volume" if r.get("max_region_share")
                    else f"market share at least {100 * r['min_market_share']:.0f}%")
            L.append(f"Rule tested: **{what}**. It costs **{_money(rule['cost_musd_per_period'])} per quarter "
                     f"({rule['cost_pct']:.2f}% of profit)**."
                     + (f" Tightening it one more point would cost a further "
                        f"{_money(rule['marginal_cost_next_point_musd'])}." if rule.get("marginal_cost_next_point_musd")
                        is not None else "") + "\n")
        if fr:
            L.append("| Cap on top region | Profit $M | Given up | Marginal cost per point $M |")
            L.append("|---:|---:|---:|---:|")
            for p in fr["points"]:
                if not p["feasible"]:
                    continue
                mc_pt = p.get("marginal_cost_musd_per_pt")
                L.append(f"| {p['cap_pct']:.1f}% | {p['profit_musd']:.2f} | {abs(p['give_up_pct']):.2f}% | "
                         f"{'' if mc_pt is None else f'{mc_pt:.3f}'} |")
            L.append("")
            if chart_files and "frontier" in chart_files:
                L.append(f"![Frontier]({chart_files['frontier']})\n")

    if mc:
        L.append("## Risk\n")
        L.append(f"{mc['draws']} joint draws of demand parameters, next-quarter demand shocks and market sizes; "
                 f"rivals re-optimise in every draw.\n")
        L.append(f"- Recommended beats status quo in **{mc['prob_recommended_wins_pct']:.1f}%** of draws.")
        L.append(f"- Uplift P10 / P50 / P90: {mc['uplift']['p10']:+.2f} / {mc['uplift']['p50']:+.2f} / "
                 f"{mc['uplift']['p90']:+.2f} $M per quarter. Worst draw: {mc['worst_case_uplift_musd']:+.2f}.")
        L.append(f"- Profit P10 / P50 / P90 if we act: {mc['recommended_profit']['p10']:.1f} / "
                 f"{mc['recommended_profit']['p50']:.1f} / {mc['recommended_profit']['p90']:.1f} $M.\n")
        if chart_files and "risk" in chart_files:
            L.append(f"![Risk]({chart_files['risk']})\n")

    L.append("## How we know\n")
    L.extend(how_we_know(b))
    if chart_files and "backtest" in chart_files:
        L.append(f"![Backtest]({chart_files['backtest']})\n")

    L.append(f"## Red team: grade {rt['grade']}\n")
    L.append(f"{rt['verdict']}\n")
    L.append("| Check | Result | Evidence |")
    L.append("|---|---|---|")
    for c in rt["checks"]:
        L.append(f"| {c['check']} | {ICON[c['status']]} | {c['evidence']} |")
    L.append("")
    L.append("## What this model cannot tell you\n")
    for lim in rt["limits"]:
        L.append(f"- {lim}")
    L.append("")
    L.append("*Generated by Decision Room. Every number above was computed by the engine, not written by a "
             "language model.*")
    return "\n".join(L) + "\n"


HTML_HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Decision Memo</title>
<style>
:root{--dr-bg:#fafaf9;--dr-ink:#18181b;--dr-muted:#52525b;--dr-muted2:#a1a1aa;--dr-grid:#e4e4e7;
--dr-accent:#2563eb;--dr-neg:#dc2626;--dr-card:#fff;--dr-pass:#15803d;--dr-warn:#b45309;--dr-fail:#b91c1c}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--dr-bg:#0c0c0e;--dr-ink:#f4f4f5;
--dr-muted:#a1a1aa;--dr-muted2:#52525b;--dr-grid:#27272a;--dr-accent:#60a5fa;--dr-neg:#f87171;--dr-card:#151518;
--dr-pass:#4ade80;--dr-warn:#fbbf24;--dr-fail:#f87171}}
:root[data-theme="dark"]{--dr-bg:#0c0c0e;--dr-ink:#f4f4f5;--dr-muted:#a1a1aa;--dr-muted2:#52525b;--dr-grid:#27272a;
--dr-accent:#60a5fa;--dr-neg:#f87171;--dr-card:#151518;--dr-pass:#4ade80;--dr-warn:#fbbf24;--dr-fail:#f87171}
*{box-sizing:border-box}body{margin:0;background:var(--dr-bg);color:var(--dr-ink);
font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:880px;margin:0 auto;padding:40px 16px 80px}h1{font-size:28px;letter-spacing:-.02em;margin:0 0 4px}
h2{font-size:18px;margin:40px 0 12px;padding-top:16px;border-top:1px solid var(--dr-grid)}
.sub{color:var(--dr-muted);margin:0 0 24px}.lead{font-size:18px;background:var(--dr-card);border:1px solid var(--dr-grid);
border-left:4px solid var(--dr-accent);padding:16px 18px;border-radius:8px}
.tbl{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--dr-grid);text-align:left}td.n,th.n{text-align:right}
th{color:var(--dr-muted);font-weight:600}svg{width:100%;height:auto;margin:12px 0}
.badge{font-weight:700;font-size:12px;letter-spacing:.04em}.pass{color:var(--dr-pass)}.warn{color:var(--dr-warn)}
.fail{color:var(--dr-fail)}.grade{display:inline-block;font-size:22px;font-weight:800;border:2px solid currentColor;
border-radius:8px;padding:0 10px;margin-right:8px}ul{padding-left:20px}footer{color:var(--dr-muted);font-size:13px;margin-top:40px}
</style></head><body><main>"""


def _inline(text: str) -> str:
    import re
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(text.strip()))


def html(b: dict) -> str:
    ch = charts_for(b)
    m, rec, sq, nash, rt = b["market"], b["recommended"], b["status_quo"], b["nash"], b["redteam"]
    out = [HTML_HEAD]
    out.append(f"<h1>Decision memo: {escape(m['leader'])} pricing, next quarter</h1>")
    out.append(f"<p class='sub'>{escape(m['source'])} · {m['periods']} quarters · {m['observations']:,} observations</p>")
    out.append(f"<p class='lead'>{escape(headline(b))}</p>")
    rows = "".join(
        f"<tr><td>{escape(r)}</td><td class='n'>{sq['by_region'][r]['price']:.1f}</td><td class='n'>{v['price']:.1f}</td>"
        f"<td class='n'>{v['price'] - sq['by_region'][r]['price']:+.1f}</td><td class='n'>{v['volume_mt']:.3f}</td>"
        f"<td class='n'>{v['share_of_our_volume_pct']:.1f}%</td></tr>" for r, v in served(rec, sq))
    out.append("<h2>The decision</h2><div class='tbl'><table><tr><th>Region</th><th class='n'>Status quo $/t</th>"
               "<th class='n'>Recommended $/t</th><th class='n'>Move</th><th class='n'>Volume Mt</th>"
               f"<th class='n'>Share of our volume</th></tr>{rows}</table></div>")
    out.append(ch["prices"])
    comp = "".join(
        f"<tr><td>{label}</td><td class='n'>{fmt.format(sq[k])}</td><td class='n'>{fmt.format(nash[k])}</td>"
        f"<td class='n'>{fmt.format(rec[k])}</td></tr>"
        for k, label, fmt in (("profit_musd", "Profit per quarter", "${:,.2f}M"), ("volume_mt", "Volume", "{:.3f} Mt"),
                              ("capacity_use_pct", "Capacity used", "{:.1f}%"),
                              ("market_share_pct", "Market share", "{:.2f}%")))
    out.append("<div class='tbl'><table><tr><th></th><th class='n'>Status quo</th><th class='n'>Nash</th>"
               f"<th class='n'>Recommended</th></tr>{comp}</table></div>")
    if "frontier" in ch:
        rule = b.get("rule")
        out.append("<h2>What the commercial-security rule costs</h2>")
        if rule:
            out.append(f"<p>Tested rule costs <b>{_money(rule['cost_musd_per_period'])} per quarter "
                       f"({rule['cost_pct']:.2f}% of profit)</b>.</p>")
        out.append(ch["frontier"])
    mc = b.get("monte_carlo")
    if mc:
        out.append(f"<h2>Risk</h2><p>Recommended beats the status quo in <b>{mc['prob_recommended_wins_pct']:.1f}%</b> "
                   f"of {mc['draws']} simulated futures. Uplift P10 / P50 / P90: {mc['uplift']['p10']:+.2f} / "
                   f"{mc['uplift']['p50']:+.2f} / {mc['uplift']['p90']:+.2f} $M per quarter.</p>")
        out.append(ch["risk"])
    if "backtest" in ch:
        out.append("<h2>How we know</h2>")
        d = b["demand"]
        out.append("<ul>" + "".join(f"<li>{_inline(x[2:])}</li>" for x in how_we_know(b)) + "</ul>")
        out.append(ch["backtest"])
    checks = "".join(f"<tr><td>{escape(c['check'])}</td><td><span class='badge {c['status']}'>{ICON[c['status']]}</span>"
                     f"</td><td>{escape(c['evidence'])}</td></tr>" for c in rt["checks"])
    gclass = {"A": "pass", "B": "pass", "C": "warn", "D": "fail"}[rt["grade"]]
    out.append(f"<h2>Red team</h2><p><span class='grade {gclass}'>{rt['grade']}</span>{escape(rt['verdict'])}</p>"
               f"<div class='tbl'><table><tr><th>Check</th><th>Result</th><th>Evidence</th></tr>{checks}</table></div>")
    out.append("<h2>What this model cannot tell you</h2><ul>" +
               "".join(f"<li>{escape(x)}</li>" for x in rt["limits"]) + "</ul>")
    out.append("<footer>Generated by Decision Room. Every number was computed by the engine, not written by a "
               "language model.</footer></main></body></html>")
    return "".join(out)
