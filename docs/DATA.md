# Bring your own market data

A market is a folder of CSV files. [`data/demo`](../data/demo) is a complete example (the synthetic market). `/export my-market` writes a fresh copy you can edit.

| File | Columns | Notes |
|---|---|---|
| `firms.csv` | `firm, capacity, is_leader` | Capacity in Mt per period. Exactly one firm has `is_leader=1`: the firm you advise. |
| `regions.csv` | `region` | Importing markets or client clusters. |
| `products.csv` | `product, firm, grade, cost` | A product is one firm's grade. `cost` is the base marginal FOB cost, $/t. Grades also define the demand nests. |
| `freight.csv` | `firm, region, freight` | $/t from each producer to each region. |
| `observations.csv` | `period, region, product, price, quantity, market_size, cost_index` | One row per product, region and period. `price` is FOB $/t. `quantity` is in Mt. `market_size` is in Mt and includes the outside option (domestic supply, substitutes). `cost_index` multiplies `cost` that period (energy, sulphur, wages). |
| `capacity.csv` *(optional)* | `period, firm, capacity` | Capacity by period, for expansions and outages. Without it, `firms.csv` applies to every period. It makes the out-of-sample equilibrium forecast more accurate. |
| `settings.json` *(optional)* | `time_effects`, `costs`, `period_unit` | `"time_effects": true` adds period fixed effects to demand, absorbing market-wide cycles (a global price spike) that move every price and buyer at once. `"costs": "implied"` says costs were backed out of prices, so the red team reports the conduct test as circular instead of passing it. `"period_unit": "year"` makes every message and memo say "year". |
| `instruments.csv` *(optional)* | `period, product, <one column per instrument>` | Extra excluded instruments (supply shifters such as mine capacity), one row per product and period present in `observations.csv`. They join the built-in cost and freight instruments. |

## Rules the loader enforces

- Inside shares (the sum of quantities over market size) must stay below 1 in every region and period: the outside option must exist.
- Missing product-region pairs are allowed, for a product not sold somewhere. The Data Steward reports them, along with any firm reporting more volume than its stated capacity.

## Then

In the editor, attach the folder to your prompt or type `/market path/to/folder`, then `/brief`. Over MCP, call `load_market` with the folder's absolute path.

## Building a market from raw industry files

Industry workbooks rarely arrive in this shape. Write a small adapter that maps them to the five CSV files above, keep a provenance note for every column, and keep licensed data out of version control.
