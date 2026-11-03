# Methodology

Every assumption the engine makes, in the order the specialists use them.

## 1. Market

Products are (firm, grade) pairs, sold FOB in several importing regions. Buyers pay FOB + freight. Each region has a market size that includes an **outside option** (domestic supply, substitutes), so total inside shares stay below one.

Data:
- A panel of price and quantity by period, region and product.
- Marginal costs, with a cost-shifter index per period.
- Freight from each producer to each region.
- Capacities: fixed, or given quarter by quarter.

## 2. Demand: nested logit, estimated by Berry inversion

Buyers choose one product or the outside option. Products are grouped into **nests by grade**: a buyer of high grade substitutes mostly to another producer's high grade. With nesting parameter σ ∈ [0, 1), where σ = 0 is plain logit, Berry (1994) gives a linear estimating equation:

  ln s_jrt − ln s_0rt = x′β − α (p_jrt + freight_jr) + σ ln s_{j|g,rt} + ξ_jrt

`x` contains grade, firm and region effects. ξ is the demand shock the analyst never sees.

**Two variables are endogenous:**
- **Price**, because sellers observe ξ when they set it. It is instrumented by the product's own cost shifter, its freight, and rivals' mean cost shifter.
- **The within-nest share.** It is instrumented by the cost shifters of the other products in the same nest, the number of products in the nest, and their interaction.

The estimator is unweighted 2SLS, with standard errors clustered by product-region, because demand shocks persist over time.

> We tried weighting by volume and measured the result on the synthetic market: volume depends on the demand shock, so the weights biased even the instrumented estimate. The engine therefore does not weight.

Demand shocks follow an AR(1) whose coefficient is estimated from residuals and carried into the next-quarter forecast.

## 3. Choosing and validating the demand model

The engine estimates both plain and nested logit, then validates each **on the decision it has to support**. For each of the last 8 quarters and each window length (8–24 quarters):

1. Fit on the window.
2. Forecast that quarter's **entire equilibrium**, prices and shares, from known costs, capacities and market sizes, without that quarter's demand shock.
3. Compare with what happened.

The benchmark is "everything stays where it was last quarter".

> A tempting shortcut is to predict shares from that quarter's *realised* prices. It is biased: realised prices already reflect the demand shock. The shortcut makes even the correct model lose to the naive benchmark, which is exactly what we observed.

**Selection rule.** Nested logit is kept only if σ is significant (95% CI lower bound above 0.02) **and** it forecasts at least as well as plain logit. The full sample is kept unless a shorter window cuts the share error by more than 2%.

## 4. Competition: capacity-constrained Bertrand-Nash

Each firm sets one FOB price per product and region to maximise profit subject to its total capacity. With m_j = p_j − c_j − λ_f, the first-order conditions are

  s + (Ω ∘ ∂s/∂p)′ m = 0,

where Ω marks which products each firm owns and λ_f ≥ 0 is the capacity shadow price.

**Solver.**
- *Inner loop:* the Morrow–Skerlos ζ-markup fixed point. It splits ∂s/∂p = −Λ + Γ with Λ diagonal and iterates m ← Λ⁻¹(s + (Ω ∘ Γ) m). It runs undamped, and damps only if it starts to oscillate.
- *Outer loop:* projected subgradient on λ, with per-firm adaptive steps.

**Certificate.**
- Stationarity, primal feasibility, dual feasibility and complementarity λ_f (K_f − Q_f) = 0.
- Uniqueness: six random restarts must land on the same prices.

The tests also check the closed-form logit markup and that the solution is a profit maximum, by finite differences.

## 5. Conduct: are rivals really competing?

The equilibrium inverts. Given observed prices and the demand model, solving (Λ − Ω ∘ Γ) m = s gives the marginal costs that would make those prices optimal. This is done under three ownership matrices:

- **Bertrand:** Ω is own-firm only.
- **Joint profit maximisation (collusion):** Ω is all ones.
- **Perfect competition:** m = 0.

Each set of implied costs is compared with stated costs. The red team flags any market where Bertrand is not the best fit. Firms at capacity are excluded from the test, because their implied costs include a shadow price the test cannot separate.

## 6. Our move: Stackelberg leader

The firm we advise commits first, and rivals then play the capacity-constrained Nash game among themselves. The leader's lever is **one price level per region**: its markups are the competitive (Nash) markups of each grade plus a regional shift. That keeps the grade spreads that demand supports, and it matches how regional pricing committees decide. SLSQP optimises the shifts, re-solving the rivals' game at every evaluation. The result is never worse than Nash, which the tests check.

**Commercial-security rules** are constraints on the leader:
- *max region share*: no region takes more than X% of our volume (client concentration);
- *min market share*: our share of the whole market stays above Y%.

A rule is priced as the profit difference between the optimum with and without it, plus the marginal cost of one more point. The **frontier** traces profit as the concentration cap tightens.

**Robust mode** replaces the single expected world with a sample-average approximation. It maximises mean profit, optionally minus a multiple of its standard deviation, over worlds drawn as in §7. The tests check that the robust plan does at least as well as the nominal plan on those worlds.

## 7. Risk

**Worlds.** Each world is a joint draw of:
- demand parameters (α, σ, effects) from the 2SLS covariance;
- next-quarter demand shocks from their AR(1);
- market sizes (±4%).

Rivals re-optimise in every world.

**Monte Carlo on the decision.** The recommended and status-quo prices face the same worlds. The output is the probability that the recommendation wins, plus percentiles of the uplift.

**What-if.** A scenario changes capacities, market sizes, costs, freight or price sensitivity. The engine compares three outcomes: today's recommendation, the new optimum after reacting, and the profit if we do not react. The difference between the last two is the value of reacting.

## 8. Red team

| Check | Fails or warns when |
|---|---|
| Instrument strength | any first-stage F ≤ 10 |
| Endogeneity correction | demand fitted without instruments |
| Price sensitivity identified | 95% CI includes zero |
| Substitution pattern | the chosen model contradicts the nesting evidence |
| Parameter stability | α differs by >15% (warn) or >30% (fail) between sample halves |
| Out-of-sample skill | the equilibrium forecast does not beat naive, or direction right ≤ 55% |
| Pricing on the elastic side | any own-price elasticity above −1 |
| Market conduct | Bertrand is not the best fit, or its cost gap exceeds 15% |
| Equilibrium certified | KKT certificate fails |
| Rivals' response unique | random restarts disagree |
| Inside the data | recommended prices outside the historical range |
| Robust to uncertainty | the recommendation wins in <70% of draws (warn) or <50% (fail) |
| Ground truth recovered | synthetic data only: true α or σ outside the CI |

Grade: any fail → D; three or more warnings → C; one or two → B; none → A.

## 9. What the model cannot tell you

- Substitution follows the nest structure (grades). Buyers who would switch region rather than grade, and long-term offtake contracts, are not represented.
- The game is static, one quarter at a time: no long-term contracts, retaliation or reputation.
- Costs and capacities come from the inputs. The engine checks them against prices but cannot verify them.
- The conduct test compares three textbook behaviours. Partial collusion or tacit regional splits sit between them.
