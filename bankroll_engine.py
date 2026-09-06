#!/usr/bin/env python3
"""A transparent Dynamic Bankroll Allocation Engine built from first principles.

The single-asset component uses the exact two-outcome Kelly objective.  The
multi-asset component includes both discrete scenarios and a diffusion solver.
For one-period returns, it constructs an equally likely distribution whose
mean and covariance match the supplied inputs exactly, then maximizes expected
log wealth over it with ``scipy.optimize.minimize``. The continuous-time
solver instead states a geometric-Brownian model, under which the mean and
covariance identify expected log growth.

The one-period scenario model uses *simple returns*: ``0.10`` means a 10% gain
and ``-0.10`` means a 10% loss. The continuous solver instead accepts
arithmetic drift and covariance *rates* on one consistent time scale. Odds are
*net odds*: with odds ``b``, a winning stake of $1 earns $b in profit, while a
losing stake loses $1.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize


FloatArray = NDArray[np.float64]


def _validate_probability(value: float, name: str = "probability") -> float:
    """Return a finite probability, rejecting values outside [0, 1]."""
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]; got {value!r}.")
    return value


def _validate_fraction(value: float, name: str = "fraction") -> float:
    """Return a long-only, unlevered bankroll fraction in [0, 1]."""
    value = float(value)
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]; got {value!r}.")
    return value


def _validate_net_odds(net_odds: float) -> float:
    """Validate and return positive net odds ``b``."""
    net_odds = float(net_odds)
    if not np.isfinite(net_odds) or net_odds <= 0.0:
        raise ValueError("net_odds must be a positive finite number.")
    return net_odds


def _clean_constrained_weights(
    raw_weights: Sequence[float] | FloatArray,
    max_total_weight: float,
    max_asset_weight: float,
) -> FloatArray:
    """Defend the cash and position limits against optimizer round-off only."""
    raw = np.asarray(raw_weights, dtype=float)
    tolerance = 1e-8
    if np.any(raw < -tolerance) or np.any(raw > max_asset_weight + tolerance):
        raise RuntimeError("Optimizer returned weights outside the stated asset bounds.")
    if raw.sum() > max_total_weight + tolerance:
        raise RuntimeError("Optimizer returned weights above the stated total-weight cap.")

    cleaned = np.clip(raw, 0.0, max_asset_weight)
    # A violation smaller than tolerance is numerical noise.  Scale it back to
    # the cash budget so a caller never receives an infeasible allocation.
    if cleaned.sum() > max_total_weight:
        cleaned *= max_total_weight / cleaned.sum()
    return cleaned


def _validated_covariance(
    covariance: Sequence[Sequence[float]] | FloatArray, n_assets: int
) -> FloatArray:
    """Validate a covariance matrix and remove negligible PSD round-off noise."""
    sigma = np.asarray(covariance, dtype=float)
    if sigma.shape != (n_assets, n_assets):
        raise ValueError("covariance must have shape (len(mu), len(mu)).")
    if not np.all(np.isfinite(sigma)):
        raise ValueError("covariance must be finite.")
    if not np.allclose(sigma, sigma.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric.")

    # The input passed the symmetry check, so this only removes insignificant
    # floating-point asymmetry before using a symmetric eigensolver.
    sigma = 0.5 * (sigma + sigma.T)
    if np.any(np.diag(sigma) < 0.0):
        raise ValueError("covariance diagonal entries (variances) must be non-negative.")

    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    covariance_scale = max(float(np.linalg.norm(sigma, ord=np.inf)), np.finfo(float).eps)
    psd_tolerance = 1e-10 * covariance_scale
    if np.min(eigenvalues) < -psd_tolerance:
        raise ValueError("covariance must be positive semidefinite.")
    if np.min(eigenvalues) < 0.0:
        # Retain a concave log-growth objective when a theoretically PSD input
        # has a tiny negative eigenvalue from numerical rounding.
        sigma = (eigenvectors * np.clip(eigenvalues, 0.0, None)) @ eigenvectors.T
    return sigma


def binary_return_distribution(
    win_probability: float, net_odds: float
) -> tuple[FloatArray, FloatArray]:
    """Build the explicit two-state distribution for a binary wager.

    Returns
    -------
    probabilities:
        ``[p, 1 - p]``.
    simple_returns:
        A shape ``(2, 1)`` array: the net return is ``b`` on a win and ``-1``
        on a loss.
    """
    p = _validate_probability(win_probability, "win_probability")
    b = _validate_net_odds(net_odds)
    probabilities = np.array([p, 1.0 - p], dtype=float)
    simple_returns = np.array([[b], [-1.0]], dtype=float)
    return probabilities, simple_returns


def expected_log_growth(
    weights: Sequence[float] | FloatArray,
    simple_returns: Sequence[Sequence[float]] | FloatArray,
    probabilities: Sequence[float] | FloatArray,
) -> float:
    """Calculate ``E[log(1 + w.T @ R)]`` for an explicit distribution.

    ``simple_returns`` has one row per state and one column per asset.  The
    function returns ``-np.inf`` when a state with positive probability makes
    the portfolio wealth multiplier non-positive.  That is the correct log
    utility treatment of bankruptcy, rather than a numerical clipping trick.
    """
    w = np.asarray(weights, dtype=float)
    returns = np.asarray(simple_returns, dtype=float)
    probs = np.asarray(probabilities, dtype=float)

    if w.ndim != 1:
        raise ValueError("weights must be a one-dimensional vector.")
    if returns.ndim == 1:
        returns = returns[:, np.newaxis]
    if returns.ndim != 2:
        raise ValueError("simple_returns must be a two-dimensional array.")
    if returns.shape[1] != w.size:
        raise ValueError("weights and simple_returns have incompatible asset dimensions.")
    if probs.ndim != 1 or probs.size != returns.shape[0]:
        raise ValueError("probabilities must have one entry per return state.")
    if not np.all(np.isfinite(returns)) or not np.all(np.isfinite(w)):
        raise ValueError("weights and simple_returns must be finite.")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("probabilities must be finite and non-negative.")
    if not np.isclose(probs.sum(), 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("probabilities must sum to one.")

    wealth_multipliers = 1.0 + returns @ w
    active_states = probs > 0.0
    if np.any(wealth_multipliers[active_states] <= 0.0):
        return float("-inf")
    return float(np.dot(probs[active_states], np.log(wealth_multipliers[active_states])))


def kelly_fraction(
    win_probability: float,
    net_odds: float,
    fractional_kelly: float = 1.0,
) -> float:
    """Return the long-only discrete Kelly fraction, optionally scaled down.

    The full-Kelly stationary point is ``(p * b - (1 - p)) / b``.  A negative
    value means that this side of the wager has no edge, so a long-only engine
    places no wager.  Use ``fractional_kelly=0.5`` for Half-Kelly.
    """
    p = _validate_probability(win_probability, "win_probability")
    b = _validate_net_odds(net_odds)
    scale = _validate_fraction(fractional_kelly, "fractional_kelly")

    full_kelly = max(0.0, (p * b - (1.0 - p)) / b)
    # The upper clip only matters at the degenerate p=1 boundary and protects
    # against round-off slightly above one.
    full_kelly = min(full_kelly, 1.0)
    return scale * full_kelly


def moment_matched_return_distribution(
    mu: Sequence[float] | FloatArray,
    covariance: Sequence[Sequence[float]] | FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """Construct an explicit distribution with exactly the supplied moments.

    For ``n`` assets, the distribution has ``2n`` equally likely states.  Its
    standardized shocks are ``+sqrt(n) e_i`` and ``-sqrt(n) e_i``.  They have
    mean zero and covariance identity.  If ``L @ L.T = Sigma``, the states
    ``mu + z @ L.T`` therefore have mean ``mu`` and covariance ``Sigma``.

    This finite distribution is a modelling choice, not a claim that the first
    two moments uniquely determine return tails.  It makes the expected-log
    objective fully specified without a finance library or a hidden normality
    assumption.
    """
    mean = np.asarray(mu, dtype=float)
    if mean.ndim != 1 or mean.size == 0:
        raise ValueError("mu must be a non-empty one-dimensional vector.")
    n_assets = mean.size
    if not np.all(np.isfinite(mean)):
        raise ValueError("mu must be finite.")
    sigma = _validated_covariance(covariance, n_assets)

    eigenvalues, eigenvectors = np.linalg.eigh(sigma)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    factor = eigenvectors @ np.diag(np.sqrt(eigenvalues))

    unit_shocks = np.eye(n_assets, dtype=float) * np.sqrt(n_assets)
    standardized_shocks = np.vstack((unit_shocks, -unit_shocks))
    simple_returns = mean + standardized_shocks @ factor.T
    probabilities = np.full(2 * n_assets, 1.0 / (2.0 * n_assets), dtype=float)

    # A simple return below -100% is not economically meaningful and could
    # make a long-only portfolio impossible to keep solvent in that state.
    if np.any(simple_returns <= -1.0):
        raise ValueError(
            "The supplied moments create a scenario with a return <= -100%. "
            "Use a shorter period, lower volatility, or custom stress scenarios."
        )
    return probabilities, simple_returns


@dataclass(frozen=True)
class MultiAssetKellyResult:
    """Result of the long-only, unlevered multivariate Kelly optimization."""

    full_kelly_weights: FloatArray
    weights: FloatArray
    full_kelly_growth: float
    expected_log_growth: float
    probabilities: FloatArray
    scenario_returns: FloatArray
    optimization: OptimizeResult


def optimize_multivariate_kelly(
    mu: Sequence[float] | FloatArray,
    covariance: Sequence[Sequence[float]] | FloatArray,
    *,
    fractional_kelly: float = 1.0,
    max_total_weight: float = 1.0,
    max_asset_weight: float = 1.0,
    tolerance: float = 1e-12,
    maxiter: int = 1_000,
) -> MultiAssetKellyResult:
    """Maximize multi-asset expected log wealth using SLSQP.

    The engine uses the explicit moment-matched distribution made from ``mu``
    and ``covariance``.  It solves

    ``max_w sum_s pi_s log(1 + R_s.T @ w)``

    subject to ``0 <= w_i <= max_asset_weight`` and
    ``sum_i w_i <= max_total_weight``.  The unused allocation is cash with a
    zero return.  The default cap of one forbids leverage and leaves no way to
    accidentally put 100% of the bankroll into each of several assets.

    Fractional Kelly is applied *after* solving for the full-Kelly vector.  It
    is a deliberate risk scaling rule, not a claim that the scaled vector is a
    new unconstrained optimum.
    """
    scale = _validate_fraction(fractional_kelly, "fractional_kelly")
    max_total_weight = _validate_fraction(max_total_weight, "max_total_weight")
    max_asset_weight = _validate_fraction(max_asset_weight, "max_asset_weight")
    if max_total_weight == 0.0 or max_asset_weight == 0.0:
        raise ValueError("max_total_weight and max_asset_weight must be positive.")
    if tolerance <= 0.0 or maxiter <= 0:
        raise ValueError("tolerance and maxiter must be positive.")

    probabilities, scenario_returns = moment_matched_return_distribution(mu, covariance)
    n_assets = scenario_returns.shape[1]
    active = probabilities > 0.0
    upper_bound = min(max_asset_weight, max_total_weight)

    def negative_expected_log_growth(weights: FloatArray) -> float:
        growth = expected_log_growth(weights, scenario_returns, probabilities)
        # SLSQP minimizes a finite scalar.  Feasible long-only unlevered inputs
        # with returns > -1 stay inside the log domain, but this preserves the
        # correct ordering if it probes an invalid point while iterating.
        return 1e100 if not np.isfinite(growth) else -growth

    def negative_gradient(weights: FloatArray) -> FloatArray:
        multipliers = 1.0 + scenario_returns @ weights
        if np.any(multipliers[active] <= 0.0):
            return np.zeros(n_assets, dtype=float)
        # d/dw log(1 + R_s.T w) = R_s / (1 + R_s.T w).
        gradient = scenario_returns[active].T @ (
            probabilities[active] / multipliers[active]
        )
        return -gradient

    result = minimize(
        negative_expected_log_growth,
        x0=np.zeros(n_assets, dtype=float),
        jac=negative_gradient,
        method="SLSQP",
        bounds=[(0.0, upper_bound)] * n_assets,
        constraints={
            "type": "ineq",
            "fun": lambda weights: max_total_weight - float(np.sum(weights)),
            "jac": lambda weights: -np.ones_like(weights),
        },
        options={"ftol": tolerance, "maxiter": int(maxiter), "disp": False},
    )
    if not result.success:
        raise RuntimeError(
            "Multivariate Kelly optimization failed: "
            f"{result.message}. Try checking the return moments or constraints."
        )

    full_weights = _clean_constrained_weights(
        result.x, max_total_weight=max_total_weight, max_asset_weight=upper_bound
    )
    scaled_weights = scale * full_weights
    full_growth = expected_log_growth(full_weights, scenario_returns, probabilities)
    scaled_growth = expected_log_growth(scaled_weights, scenario_returns, probabilities)
    return MultiAssetKellyResult(
        full_kelly_weights=full_weights,
        weights=scaled_weights,
        full_kelly_growth=full_growth,
        expected_log_growth=scaled_growth,
        probabilities=probabilities,
        scenario_returns=scenario_returns,
        optimization=result,
    )


@dataclass(frozen=True)
class ContinuousKellyResult:
    """Result under the continuous-time diffusion version of multivariate Kelly."""

    full_kelly_weights: FloatArray
    weights: FloatArray
    full_kelly_growth: float
    expected_log_growth: float
    cash_weight: float
    optimization: OptimizeResult


def continuous_expected_log_growth(
    weights: Sequence[float] | FloatArray,
    mu: Sequence[float] | FloatArray,
    covariance: Sequence[Sequence[float]] | FloatArray,
    risk_free_rate: float = 0.0,
) -> float:
    """Return the diffusion-model expected log-growth rate.

    The continuous model is

    ``dS_i / S_i = mu_i dt + (L dB)_i,  L L.T = covariance``.

    Here ``mu`` is an arithmetic drift *rate* and ``covariance`` is an
    instantaneous covariance *rate*, both expressed on the same time scale.
    The returned value is an expected log-growth rate on that scale.

    With the unallocated balance held in cash at ``risk_free_rate``, Itô's
    formula gives

    ``E[d log(V)] / dt = r_f + w.T @ (mu - r_f) - 0.5 w.T @ Sigma @ w``.

    Thus the mean and covariance do identify the objective *under this stated
    diffusion model*.  This should not be confused with an exact one-period
    simple-return objective, for which the first two moments alone are not
    enough; use ``optimize_multivariate_kelly`` for the explicit scenario
    model above.
    """
    w = np.asarray(weights, dtype=float)
    mean = np.asarray(mu, dtype=float)
    rf = float(risk_free_rate)
    if w.ndim != 1 or mean.ndim != 1 or w.shape != mean.shape:
        raise ValueError("weights and mu must be one-dimensional vectors of equal length.")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(mean)):
        raise ValueError("weights and mu must be finite.")
    if not np.isfinite(rf):
        raise ValueError("risk_free_rate must be finite.")
    sigma = _validated_covariance(covariance, mean.size)
    return float(rf + w @ (mean - rf) - 0.5 * (w @ sigma @ w))


def continuous_log_growth_gradient(
    weights: Sequence[float] | FloatArray,
    mu: Sequence[float] | FloatArray,
    covariance: Sequence[Sequence[float]] | FloatArray,
    risk_free_rate: float = 0.0,
) -> FloatArray:
    """Return ``mu - r_f - Sigma @ w``, the diffusion Kelly gradient."""
    w = np.asarray(weights, dtype=float)
    mean = np.asarray(mu, dtype=float)
    rf = float(risk_free_rate)
    if w.ndim != 1 or mean.ndim != 1 or w.shape != mean.shape:
        raise ValueError("weights and mu must be one-dimensional vectors of equal length.")
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(mean)) or not np.isfinite(rf):
        raise ValueError("weights, mu, and risk_free_rate must be finite.")
    sigma = _validated_covariance(covariance, mean.size)
    return mean - rf - sigma @ w


def optimize_continuous_multivariate_kelly(
    mu: Sequence[float] | FloatArray,
    covariance: Sequence[Sequence[float]] | FloatArray,
    *,
    risk_free_rate: float = 0.0,
    fractional_kelly: float = 1.0,
    max_total_weight: float = 1.0,
    max_asset_weight: float = 1.0,
    tolerance: float = 1e-12,
    maxiter: int = 1_000,
) -> ContinuousKellyResult:
    """Solve the requested continuous, correlated multi-asset Kelly problem.

    ``mu`` contains arithmetic drift rates and ``covariance`` contains
    instantaneous covariance rates on the same time scale. The output is a
    log-growth rate on that scale. The optimizer is long-only and unlevered:
    the cash weight is ``1 - sum(weights)``. The function works for any asset
    count, while the demo below intentionally supplies three assets.
    """
    mean = np.asarray(mu, dtype=float)
    rf = float(risk_free_rate)
    scale = _validate_fraction(fractional_kelly, "fractional_kelly")
    max_total_weight = _validate_fraction(max_total_weight, "max_total_weight")
    max_asset_weight = _validate_fraction(max_asset_weight, "max_asset_weight")
    if mean.ndim != 1 or mean.size == 0:
        raise ValueError("mu must be a non-empty one-dimensional vector.")
    if not np.all(np.isfinite(mean)) or not np.isfinite(rf):
        raise ValueError("mu and risk_free_rate must be finite.")
    sigma = _validated_covariance(covariance, mean.size)
    if max_total_weight == 0.0 or max_asset_weight == 0.0:
        raise ValueError("max_total_weight and max_asset_weight must be positive.")
    if tolerance <= 0.0 or maxiter <= 0:
        raise ValueError("tolerance and maxiter must be positive.")

    n_assets = mean.size
    upper_bound = min(max_asset_weight, max_total_weight)
    result = minimize(
        fun=lambda weights: -continuous_expected_log_growth(weights, mean, sigma, rf),
        x0=np.zeros(n_assets, dtype=float),
        jac=lambda weights: -continuous_log_growth_gradient(weights, mean, sigma, rf),
        method="SLSQP",
        bounds=[(0.0, upper_bound)] * n_assets,
        constraints={
            "type": "ineq",
            "fun": lambda weights: max_total_weight - float(np.sum(weights)),
            "jac": lambda weights: -np.ones_like(weights),
        },
        options={"ftol": tolerance, "maxiter": int(maxiter), "disp": False},
    )
    if not result.success:
        raise RuntimeError(
            "Continuous multivariate Kelly optimization failed: "
            f"{result.message}. Try checking the return moments or constraints."
        )

    full_weights = _clean_constrained_weights(
        result.x, max_total_weight=max_total_weight, max_asset_weight=upper_bound
    )
    scaled_weights = scale * full_weights
    return ContinuousKellyResult(
        full_kelly_weights=full_weights,
        weights=scaled_weights,
        full_kelly_growth=continuous_expected_log_growth(full_weights, mean, sigma, rf),
        expected_log_growth=continuous_expected_log_growth(scaled_weights, mean, sigma, rf),
        cash_weight=float(1.0 - np.sum(scaled_weights)),
        optimization=result,
    )


def simulate_fractional_bankroll(
    outcomes: Sequence[bool] | NDArray[np.bool_],
    fraction: float,
    net_odds: float,
    initial_bankroll: float = 100.0,
) -> FloatArray:
    """Simulate constant-proportion betting over a supplied win/loss sequence."""
    f = _validate_fraction(fraction, "fraction")
    b = _validate_net_odds(net_odds)
    starting_bankroll = float(initial_bankroll)
    if not np.isfinite(starting_bankroll) or starting_bankroll <= 0.0:
        raise ValueError("initial_bankroll must be a positive finite number.")

    wins = np.asarray(outcomes, dtype=bool)
    if wins.ndim != 1:
        raise ValueError("outcomes must be a one-dimensional win/loss sequence.")

    equity = np.empty(wins.size + 1, dtype=float)
    equity[0] = starting_bankroll
    for trade_index, won in enumerate(wins, start=1):
        net_return = b if won else -1.0
        equity[trade_index] = equity[trade_index - 1] * (1.0 + f * net_return)
    return equity


def simulate_fixed_stake_bankroll(
    outcomes: Sequence[bool] | NDArray[np.bool_],
    fixed_stake: float,
    net_odds: float,
    initial_bankroll: float = 100.0,
) -> FloatArray:
    """Simulate a naive constant-currency stake, stopping at literal ruin.

    The stake stays fixed in currency units while the bankroll is positive.  If
    less than one full stake remains, the final wager is reduced to the cash
    left; this prevents the toy simulation from silently borrowing money.
    """
    stake = float(fixed_stake)
    b = _validate_net_odds(net_odds)
    starting_bankroll = float(initial_bankroll)
    if not np.isfinite(stake) or stake <= 0.0:
        raise ValueError("fixed_stake must be a positive finite number.")
    if not np.isfinite(starting_bankroll) or starting_bankroll <= 0.0:
        raise ValueError("initial_bankroll must be a positive finite number.")

    wins = np.asarray(outcomes, dtype=bool)
    if wins.ndim != 1:
        raise ValueError("outcomes must be a one-dimensional win/loss sequence.")

    equity = np.empty(wins.size + 1, dtype=float)
    equity[0] = starting_bankroll
    for trade_index, won in enumerate(wins, start=1):
        previous = equity[trade_index - 1]
        wager = min(stake, previous)
        net_return = b if won else -1.0
        equity[trade_index] = previous + wager * net_return
    return equity


@dataclass(frozen=True)
class KellySimulation:
    """Three strategies evaluated against the exact same binary outcomes."""

    outcomes: NDArray[np.bool_]
    full_kelly_fraction: float
    half_kelly_fraction: float
    full_kelly_equity: FloatArray
    half_kelly_equity: FloatArray
    fixed_stake_equity: FloatArray
    fixed_stake: float


def run_coin_flip_comparison(
    *,
    n_trades: int = 1_000,
    win_probability: float = 0.55,
    net_odds: float = 1.0,
    initial_bankroll: float = 100.0,
    fixed_stake: float = 5.0,
    seed: int = 20260906,
) -> KellySimulation:
    """Create a reproducible 55%-win-rate Kelly-versus-fixed-stake experiment."""
    if int(n_trades) != n_trades or n_trades <= 0:
        raise ValueError("n_trades must be a positive integer.")
    n_trades = int(n_trades)
    p = _validate_probability(win_probability, "win_probability")
    b = _validate_net_odds(net_odds)

    rng = np.random.default_rng(seed)
    outcomes = rng.random(n_trades) < p
    full_fraction = kelly_fraction(p, b, fractional_kelly=1.0)
    half_fraction = kelly_fraction(p, b, fractional_kelly=0.5)
    return KellySimulation(
        outcomes=outcomes,
        full_kelly_fraction=full_fraction,
        half_kelly_fraction=half_fraction,
        full_kelly_equity=simulate_fractional_bankroll(
            outcomes, full_fraction, b, initial_bankroll
        ),
        half_kelly_equity=simulate_fractional_bankroll(
            outcomes, half_fraction, b, initial_bankroll
        ),
        fixed_stake_equity=simulate_fixed_stake_bankroll(
            outcomes, fixed_stake, b, initial_bankroll
        ),
        fixed_stake=float(fixed_stake),
    )


def max_drawdown(equity: Sequence[float] | FloatArray) -> float:
    """Return the largest peak-to-trough drawdown as a fraction of the peak."""
    values = np.asarray(equity, dtype=float)
    if (
        values.ndim != 1
        or values.size == 0
        or not np.all(np.isfinite(values))
        or np.any(values < 0.0)
    ):
        raise ValueError("equity must be a finite, non-empty, non-negative vector.")
    peaks = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = np.where(peaks > 0.0, 1.0 - values / peaks, 0.0)
    return float(np.max(drawdowns))


@dataclass(frozen=True)
class PerformanceMetrics:
    """Risk and return statistics calculated from a positive equity curve.

    The base metrics are expressed per observation interval.  When
    ``periods_per_year`` is supplied to :func:`calculate_performance_metrics`,
    the annualized fields use conventional square-root-of-time scaling for
    volatility and Sharpe ratio.  That assumption should only be used when the
    intervals represent a consistent real-world cadence (for example, daily
    returns), rather than the synthetic coin flips used by the default demo.
    """

    periods: int
    total_return: float
    compound_return_per_period: float
    volatility_per_period: float
    sharpe_ratio_per_period: float
    max_drawdown: float
    annualized_return: float | None
    annualized_volatility: float | None
    annualized_sharpe_ratio: float | None


def calculate_performance_metrics(
    equity: Sequence[float] | FloatArray,
    *,
    risk_free_rate_per_period: float = 0.0,
    periods_per_year: float | None = None,
) -> PerformanceMetrics:
    """Calculate transparent risk/return metrics for consecutive equity values.

    ``risk_free_rate_per_period`` is a simple return on the *same interval* as
    adjacent observations in ``equity``.  Sharpe ratio is calculated as the
    arithmetic mean excess simple return divided by sample return volatility
    (``ddof=1``).  Its value is ``nan`` if fewer than two returns are available
    or every observed return is identical, because a sample volatility-based
    Sharpe ratio is then undefined.

    Provide ``periods_per_year`` only when an interval has a genuine calendar
    frequency.  Annualized return compounds the geometric per-period return;
    annualized volatility and Sharpe ratio use the standard iid
    square-root-of-time convention.  The function requires strictly positive
    equity values so every period return is well-defined; a ruined path should
    be reported separately rather than assigned a misleading Sharpe ratio.
    """
    values = np.asarray(equity, dtype=float)
    if (
        values.ndim != 1
        or values.size < 2
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
    ):
        raise ValueError("equity must be a finite, strictly positive vector with at least two values.")

    risk_free = float(risk_free_rate_per_period)
    if not np.isfinite(risk_free) or risk_free <= -1.0:
        raise ValueError("risk_free_rate_per_period must be finite and greater than -1.")

    annualization: float | None = None
    if periods_per_year is not None:
        annualization = float(periods_per_year)
        if not np.isfinite(annualization) or annualization <= 0.0:
            raise ValueError("periods_per_year must be a positive finite number when provided.")

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        period_returns = values[1:] / values[:-1] - 1.0
    if not np.all(np.isfinite(period_returns)):
        raise ValueError("equity values produce non-finite period returns.")
    periods = int(period_returns.size)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        total_return = float(values[-1] / values[0] - 1.0)
        compound_return = float(np.expm1(np.mean(np.log1p(period_returns))))
    if not np.isfinite(total_return) or not np.isfinite(compound_return):
        raise ValueError("equity values produce non-finite return metrics.")

    if periods < 2:
        volatility = float("nan")
        sharpe_ratio = float("nan")
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            volatility = float(np.std(period_returns, ddof=1))
        if not np.isfinite(volatility):
            raise ValueError("equity values produce non-finite return volatility.")
        sharpe_ratio = (
            float(np.mean(period_returns - risk_free) / volatility)
            if volatility > 0.0
            else float("nan")
        )
        if not np.isfinite(sharpe_ratio) and volatility > 0.0:
            raise ValueError("equity values produce a non-finite Sharpe ratio.")

    if annualization is None:
        annualized_return = None
        annualized_volatility = None
        annualized_sharpe = None
    else:
        with np.errstate(over="ignore", invalid="ignore"):
            annualized_return = float(np.expm1(np.log1p(compound_return) * annualization))
            annualized_volatility = (
                float(volatility * np.sqrt(annualization))
                if np.isfinite(volatility)
                else float("nan")
            )
            annualized_sharpe = (
                float(sharpe_ratio * np.sqrt(annualization))
                if np.isfinite(sharpe_ratio)
                else float("nan")
            )
        if not np.isfinite(annualized_return):
            raise ValueError("annualization produces a non-finite return.")
        if np.isfinite(volatility) and not np.isfinite(annualized_volatility):
            raise ValueError("annualization produces a non-finite volatility.")
        if np.isfinite(sharpe_ratio) and not np.isfinite(annualized_sharpe):
            raise ValueError("annualization produces a non-finite Sharpe ratio.")

    return PerformanceMetrics(
        periods=periods,
        total_return=total_return,
        compound_return_per_period=compound_return,
        volatility_per_period=volatility,
        sharpe_ratio_per_period=sharpe_ratio,
        max_drawdown=max_drawdown(values),
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        annualized_sharpe_ratio=annualized_sharpe,
    )


def plot_equity_curves(simulation: KellySimulation, output_path: str | Path) -> Path:
    """Write the requested three-strategy equity-curve plot to ``output_path``."""
    # Delayed import keeps the mathematical engine usable in non-plotting
    # environments and makes headless test runs straightforward.
    import matplotlib.pyplot as plt

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    trades = np.arange(simulation.outcomes.size + 1)

    figure, axis = plt.subplots(figsize=(11, 6.5), constrained_layout=True)
    axis.plot(
        trades,
        np.maximum(simulation.full_kelly_equity, np.finfo(float).tiny),
        color="#9b2226",
        linewidth=1.8,
        label=(f"100% Kelly ({simulation.full_kelly_fraction:.1%} of bankroll)"),
    )
    axis.plot(
        trades,
        np.maximum(simulation.half_kelly_equity, np.finfo(float).tiny),
        color="#005f73",
        linewidth=2.1,
        label=(f"Half-Kelly ({simulation.half_kelly_fraction:.1%} of bankroll)"),
    )
    axis.plot(
        trades,
        np.maximum(simulation.fixed_stake_equity, np.finfo(float).tiny),
        color="#6c757d",
        linewidth=1.6,
        label=f"Naive fixed stake (${simulation.fixed_stake:g} per trade)",
    )
    axis.set_yscale("log")
    axis.set_title(
        f"Kelly bankroll paths on the same {simulation.outcomes.size:,} flips", weight="bold"
    )
    axis.set_xlabel("Trade number")
    axis.set_ylabel("Bankroll (log scale)")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, loc="upper left")
    axis.text(
        0.99,
        0.02,
        "55% win probability, even-money odds, same seeded outcome sequence",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#495057",
    )
    figure.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return destination


def _demo_multi_asset_problem() -> tuple[FloatArray, FloatArray]:
    """Return three correlated annual drift/covariance rates for the CLI demo."""
    # The first two assets are strongly correlated tech names (rho = 0.90).
    # The third has lower correlation and represents a diversifying asset.
    mu = np.array([0.07425, 0.0796875, 0.06415625], dtype=float)
    volatility = np.array([0.20, 0.25, 0.15], dtype=float)
    correlation = np.array(
        [
            [1.00, 0.90, 0.20],
            [0.90, 1.00, 0.15],
            [0.20, 0.15, 1.00],
        ],
        dtype=float,
    )
    covariance = np.outer(volatility, volatility) * correlation
    return mu, covariance


def _format_metric(value: float, format_spec: str) -> str:
    """Format a finite statistic, using N/A for an undefined sample metric."""
    return format(value, format_spec) if np.isfinite(value) else "N/A"


def main() -> None:
    """Run the reproducible simulation and print the multi-asset allocation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="kelly_equity_curves.png",
        help="PNG destination for the 1,000-trade equity-curve plot.",
    )
    parser.add_argument("--trades", type=int, default=1_000, help="Number of flips to simulate.")
    parser.add_argument("--seed", type=int, default=20260906, help="Random seed for reproducibility.")
    args = parser.parse_args()

    simulation = run_coin_flip_comparison(n_trades=args.trades, seed=args.seed)
    output = plot_equity_curves(simulation, args.output)

    print("Single-asset Kelly (p=55%, even-money odds)")
    print(f"  Full Kelly fraction: {simulation.full_kelly_fraction:.2%}")
    print(f"  Half-Kelly fraction: {simulation.half_kelly_fraction:.2%}")
    print("\nSimulation statistics")
    print(
        f"  Empirical win rate: {np.mean(simulation.outcomes):.1%} "
        f"({np.count_nonzero(simulation.outcomes)}/{simulation.outcomes.size})"
    )
    for label, curve in (
        ("100% Kelly", simulation.full_kelly_equity),
        ("Half-Kelly", simulation.half_kelly_equity),
        ("Fixed stake", simulation.fixed_stake_equity),
    ):
        if np.any(curve <= 0.0):
            print(
                f"  {label:12s} final=${curve[-1]:,.2f}; "
                f"total return={curve[-1] / curve[0] - 1.0:.1%}; "
                "volatility/trade=N/A; Sharpe (trade returns)=N/A (ruined path); "
                f"max drawdown={max_drawdown(curve):.1%}"
            )
            continue
        metrics = calculate_performance_metrics(
            curve,
        )
        print(
            f"  {label:12s} final=${curve[-1]:,.2f}; "
            f"total return={metrics.total_return:.1%}; "
            f"volatility/trade={_format_metric(metrics.volatility_per_period, '.2%')}; "
            f"Sharpe (trade returns)={_format_metric(metrics.sharpe_ratio_per_period, '.4f')}; "
            f"max drawdown={metrics.max_drawdown:.1%}"
        )
    print(f"\nWrote equity plot: {output.resolve()}")

    mu, covariance = _demo_multi_asset_problem()
    allocation = optimize_continuous_multivariate_kelly(
        mu, covariance, fractional_kelly=0.5, max_total_weight=1.0
    )
    print("\nThree-asset continuous multivariate Kelly (Half-Kelly scaled)")
    print(f"  Full-Kelly weights: {np.array2string(allocation.full_kelly_weights, precision=4)}")
    print(f"  Half-Kelly weights: {np.array2string(allocation.weights, precision=4)}")
    print(f"  Cash weight: {allocation.cash_weight:.4f}")
    print(f"  Expected log growth: {allocation.expected_log_growth:.5f} per year")


if __name__ == "__main__":
    main()
