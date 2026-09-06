# Dynamic Bankroll Allocation Engine

A from-scratch Python implementation of discrete Kelly betting, fractional
Kelly risk scaling, and constrained multivariate Kelly allocation. It uses only
NumPy, `scipy.optimize.minimize`, and Matplotlib—there are no finance-library
calls or black-box portfolio optimizers.

## Run it

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl python3 bankroll_engine.py \
  --output kelly_equity_curves.png
python3 -m unittest -v test_bankroll_engine.py
```

The first command creates `kelly_equity_curves.png` from 1,000 seeded,
even-money flips with a 55% win probability. All three strategies see the exact
same outcomes, so path differences come from sizing rather than luck.

## Performance analytics

`calculate_performance_metrics(...)` turns any strictly positive equity curve
into transparent, reusable risk/return statistics:

- total return and geometric mean period return;
- sample volatility and Sharpe ratio per observation interval;
- peak-to-trough maximum drawdown; and
- optional annualized return, volatility, and Sharpe ratio when a real calendar
  frequency is supplied.

For consecutive equity values $E_t$, the engine uses simple period returns
$r_t=E_t/E_{t-1}-1$ and calculates the per-period Sharpe ratio as

$$
S=\frac{\mu(r_t-r_f)}{\sigma(r_t)},
$$

where $S$ is the Sharpe ratio, $r_f$ is the simple risk-free return on the
**same interval**, $\mu$ denotes the sample mean, and $\sigma$ denotes the
sample standard deviation (`ddof=1`). A Sharpe ratio is undefined, and reported
as `nan`, when there are fewer than two returns or zero observed volatility.

The bundled demonstration consists of synthetic flips rather than dated market
returns, so the CLI reports metrics **per trade**. For an externally supplied
equity curve at a known cadence, opt into conventional annualization through
the reusable API:

```python
from bankroll_engine import calculate_performance_metrics

metrics = calculate_performance_metrics(
    daily_equity,
    risk_free_rate_per_period=(1.04 ** (1 / 252)) - 1,
    periods_per_year=252,
)
```

Annualized volatility and Sharpe use the standard square-root-of-time
assumption; annualized return compounds the geometric per-period return. Do
not annualize the default synthetic flip results or present them as a backtest.

### Seeded synthetic example — not historical investment performance

With the default seed (`20260906`) and a zero per-trade risk-free rate, the
1,000-trade simulation produces 536 wins (53.6%). Starting from $100, the
reproducible path reports:

| Strategy | Final bankroll | Total return | Volatility/trade | Sharpe (trade returns) | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full Kelly | $901.58 | 801.6% | 9.98% | 0.0722 | 90.5% |
| Half Kelly | $1,050.07 | 950.1% | 4.99% | 0.0722 | 64.4% |
| Fixed stake | $460.00 | 360.0% | 3.13% | 0.0644 | 41.9% |

Those figures demonstrate the reporting pipeline on one controlled sample;
they do not establish a trading strategy's expected real-world returns.

## CV-ready project bullets

Use one or two of these, tailored to the role:

- Built a Python Kelly allocation engine with exact binary sizing and constrained multivariate optimization; generated $2n$ moment-matched return scenarios and used SLSQP to enforce long-only, cash, and position limits.
- Added risk analytics for compound return, volatility, Sharpe ratio, and maximum drawdown to compare 3 sizing strategies across a reproducible 1,000-trial simulation; the reusable API supports frequency-aware annualization for dated equity curves and is validated by 18 unit tests.
- Modeled a correlated 3-asset allocation example in which 0.90 correlation between two tech assets reduced their combined Full-Kelly allocation from about 74% to 50% versus a diagonal-covariance assumption.

## Conventions

`net_odds=b` means that a $1 winning stake earns $b in profit; a losing stake
loses the $1. For decimal/gross odds `d`, pass `b=d-1`. A fraction is always a
fraction of the *current* bankroll, except for the deliberately naive fixed
currency stake in the simulation.

## Exact discrete Kelly derivation

Let the current bankroll be $W$, win probability be $p$, loss probability
be $q=1-p$, and the fraction wagered be $f$. A win and a loss produce

$$
W' = W(1+bf) \quad\text{and}\quad W' = W(1-f),
$$

respectively. The per-bet expected log growth is therefore

$$
g(f)=p\log(1+bf)+q\log(1-f), \qquad 0\le f<1.
$$

Differentiate rather than optimizing expected dollars:

$$
g'(f)=\frac{pb}{1+bf}-\frac{q}{1-f}.
$$

Setting the derivative to zero and cross-multiplying gives

$$
pb(1-f)=q(1+bf),
$$

$$
pb-q=bf(p+q)=bf,
$$

so the stationary point is

$$
\boxed{f^*=\frac{pb-q}{b}=p-\frac{q}{b}}.
$$

It is the unique maximum because

$$
g''(f)=-\frac{pb^2}{(1+bf)^2}-\frac{q}{(1-f)^2}<0.
$$

`kelly_fraction(p, b, fractional_kelly=0.5)` applies the safety multiplier
after enforcing the long-only no-bet rule, $\max(0,f^*)$. Thus an even-money
55% wager gives Full Kelly = 10% and Half-Kelly = 5%.

## Two explicit multi-asset models

Means and covariance alone do **not** uniquely define the one-period
distribution of simple returns. The module makes that modelling choice visible
instead of hiding it.

`optimize_multivariate_kelly(mu, covariance, ...)` first builds six equally
likely scenarios for three assets (in general, `2n` for `n` assets). Their
standardized shocks are $+\sqrt{n}e_i$ and $-\sqrt{n}e_i$. With
$LL^\mathsf{T}=\Sigma$, the states

$$
R_s=\mu+z_sL^\mathsf{T}
$$

have exactly the input mean and covariance. It then maximizes the fully
specified one-period objective

$$
\sum_s\pi_s\log(1+R_s^\mathsf{T}w)
$$

with SLSQP. Any state with a non-positive wealth multiplier returns
$-\infty$, which correctly represents bankruptcy under log utility.

For the requested continuous model, use
`optimize_continuous_multivariate_kelly(mu, covariance, ...)`. Here `mu` is an
arithmetic drift **rate** and `covariance` is an instantaneous covariance
**rate**, both on the same time scale; the output is a log-growth rate on that
scale. It assumes

$$
\frac{dS_i}{S_i}=\mu_i\,dt+(L\,dB)_i, \qquad LL^\mathsf{T}=\Sigma.
$$

With the unused allocation in cash earning $r_f$, Itô's formula yields the
exact diffusion-model expected log-growth rate

$$
G(w)=r_f+w^\mathsf{T}(\mu-r_f\mathbf{1})
-\frac12 w^\mathsf{T}\Sigma w.
$$

The optimizer supplies the analytic gradient
$\nabla G=\mu-r_f\mathbf{1}-\Sigma w$, uses
`scipy.optimize.minimize(method="SLSQP")`, and enforces
$w_i\ge0$, $\sum_i w_i\le1$, and optional individual caps. The cash
buffer and position caps are intentional risk controls, not merely numerical
details.

For example:

```python
import numpy as np
from bankroll_engine import optimize_continuous_multivariate_kelly

mu = np.array([0.07425, 0.0796875, 0.06415625])  # annual drift rates
volatility = np.array([0.20, 0.25, 0.15])
correlation = np.array([
    [1.00, 0.90, 0.20],
    [0.90, 1.00, 0.15],
    [0.20, 0.15, 1.00],
])
sigma = np.outer(volatility, volatility) * correlation

allocation = optimize_continuous_multivariate_kelly(
    mu, sigma, fractional_kelly=0.5, max_total_weight=1.0
)
print(allocation.full_kelly_weights)  # [0.25, 0.25, 0.50]
print(allocation.weights)             # [0.125, 0.125, 0.25]
```

## Why log wealth, not raw expected value?

For one binary wager,

$$
\mathbb E[W'/W]=p(1+bf)+q(1-f)=1+f(pb-q).
$$

That expression is linear in $f$. If the edge $pb-q$ is positive, raw-EV
maximization chooses the largest permitted position, typically all-in
$f=1$. A single loss then produces $W'=0$. Across $N$ independent
bets, literal ruin has probability

$$
\Pr(\text{ruin by }N)=1-p^N,
$$

which approaches one whenever $p<1$, despite a possibly rising arithmetic
mean bankroll. Rare survivors dominate that mean. Expected log wealth assigns
$\log 0=-\infty$, so it maximizes typical compound growth instead. This
does not make Half-Kelly invulnerable: it avoids literal zero in this bounded
binary model but can still have severe drawdowns and practical ruin.

## Why an optimistic probability estimate is dangerous

If the true win probability is $p_0$ and the engine uses $\hat p$, then

$$
f(\hat p)-f(p_0)=\frac{b+1}{b}(\hat p-p_0).
$$

Overstating $p$ directly creates overbetting. The dangerous loss term is
$q_0\log(1-f)$, which tends to $-\infty$ as $f$ approaches one. In
contrast, a long-only under-estimate drives the allocation toward zero, whose
worst usual consequence is a finite missed-growth opportunity. Locally, errors
are not magically one-sided—the exact interior regret is

$$
g_{p_0}(f(\hat p))-g_{p_0}(f(p_0))
=-D_{\mathrm{KL}}(\mathrm{Bern}(p_0)\Vert \mathrm{Bern}(\hat p)).
$$

The practical asymmetry comes from the solvency boundary: optimistic estimates
can push exposure toward all-in, where one ordinary loss is catastrophic.

## How covariance protects against correlated tech exposure

The continuous objective has the risk term

$$
w^\mathsf{T}\Sigma w=
\sum_i\sigma_i^2w_i^2+
2\sum_{i \lt j}\rho_{ij}\sigma_i\sigma_jw_iw_j.
$$

High positive correlation between two tech stocks makes the cross term large.
The marginal log-growth of Tech B falls when Tech A is already held:

$$
\frac{\partial G}{\partial w_i}=
\mu_i-(\Sigma w)_i.
$$

In the included example, the 0.90 tech correlation gives a combined Full-Kelly
tech allocation of 50%; deleting the off-diagonal covariances raises it to
about 74%. The budget constraint prevents assigning 100% to *each* stock, and
the covariance term prevents treating two nearly identical bets as independent
diversifiers. Covariance is still a soft penalty, not a hard sector ban: if
expected returns are estimated high enough, the optimizer can concentrate.
Use an explicit sector cap, factor-risk cap, drawdown limit, and stressed return
scenarios if concentration must be prohibited operationally.

## Important production caveats

- Estimate `p`, `mu`, and `Sigma` conservatively and out of sample; the engine
  optimizes the model, not truth.
- Align the return horizon of `mu` and `Sigma`; annual drift with daily
  covariance is invalid.
- Include costs, slippage, taxes, jump/default risk, liquidity limits, and
  position caps before using real capital.
- Re-estimate and rebalance on a documented schedule. Do not reactively size
  up because a short simulation happened to win.
