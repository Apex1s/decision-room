// Period wording. The demo market is quarterly; a market folder can declare
// another period (settings.json "period_unit": "year"), and every message the
// agent sends is relabelled to match, so annual data never reads "per quarter".

let unit: "quarter" | "year" = "quarter";

export function setPeriodUnit(u?: string) {
  unit = u === "year" ? "year" : "quarter";
}

export function periodUnit() {
  return unit;
}

export function relabel(text: string): string {
  if (unit === "quarter") return text;
  return text
    .replace(/\bQuarters\b/g, "Years")
    .replace(/\bquarters\b/g, "years")
    .replace(/\bQuarter\b/g, "Year")
    .replace(/\bquarter\b/g, "year")
    .replace(/\bMt\/q\b/g, "Mt/yr")
    .replace(/\$M\/q\b/g, "$M/yr")
    .replace(/\b(\d+)q\b/g, "$1y");
}
