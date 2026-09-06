"""Regression tests for the from-scratch Kelly bankroll engine.

Run with:
    python3 -m unittest -v test_bankroll_engine.py
"""

from __future__ import annotations

import unittest

import numpy as np

from bankroll_engine import (
    binary_return_distribution,
    calculate_performance_metrics,
    continuous_expected_log_growth,
    continuous_log_growth_gradient,
    expected_log_growth,
    kelly_fraction,
    max_drawdown,
    moment_matched_return_distribution,
    optimize_continuous_multivariate_kelly,
    optimize_multivariate_kelly,
    run_coin_flip_comparison,
    simulate_fixed_stake_bankroll,
    simulate_fractional_bankroll,
)


def correlated_three_asset_moments() -> tuple[np.ndarray, np.ndarray]:
    """Three assets: two highly correlated tech names and one diversifier."""
    mu = np.array([0.07425, 0.0796875, 0.06415625])
    volatility = np.array([0.20, 0.25, 0.15])
    correlation = np.array(
        [[1.00, 0.90, 0.20], [0.90, 1.00, 0.15], [0.20, 0.15, 1.00]]
    )
    return mu, np.outer(volatility, volatility) * correlation


class SingleAssetKellyTests(unittest.TestCase):
    def test_even_money_formula_and_half_kelly(self) -> None:
        self.assertAlmostEqual(kelly_fraction(0.55, 1.0), 0.10)
        self.assertAlmostEqual(kelly_fraction(0.55, 1.0, fractional_kelly=0.5), 0.05)

    def test_long_only_engine_skips_negative_edge(self) -> None:
        self.assertEqual(kelly_fraction(0.45, 1.0), 0.0)

    def test_kelly_stationary_point_beats_nearby_fractions(self) -> None:
        probabilities, returns = binary_return_distribution(0.55, 1.0)
        optimum = kelly_fraction(0.55, 1.0)
        optimum_growth = expected_log_growth([optimum], returns, probabilities)
        self.assertGreater(optimum_growth, expected_log_growth([0.05], returns, probabilities))
        self.assertGreater(optimum_growth, expected_log_growth([0.15], returns, probabilities))

    def test_log_objective_treats_bankruptcy_as_negative_infinity(self) -> None:
        probabilities, returns = binary_return_distribution(0.55, 1.0)
        self.assertEqual(expected_log_growth([1.0], returns, probabilities), float("-inf"))


class DistributionAndScenarioTests(unittest.TestCase):
    def test_moment_matched_states_recover_mean_and_covariance(self) -> None:
        mu, covariance = correlated_three_asset_moments()
        probabilities, returns = moment_matched_return_distribution(mu, covariance)
        weighted_mean = probabilities @ returns
        centered = returns - weighted_mean
        weighted_covariance = (centered * probabilities[:, np.newaxis]).T @ centered
        np.testing.assert_allclose(weighted_mean, mu, atol=1e-12)
        np.testing.assert_allclose(weighted_covariance, covariance, atol=1e-12)

    def test_scenario_optimizer_respects_long_only_budget(self) -> None:
        mu, covariance = correlated_three_asset_moments()
        result = optimize_multivariate_kelly(mu, covariance, fractional_kelly=0.5)
        self.assertTrue(np.all(result.full_kelly_weights >= -1e-10))
        self.assertLessEqual(result.full_kelly_weights.sum(), 1.0 + 1e-9)
        np.testing.assert_allclose(result.weights, 0.5 * result.full_kelly_weights, atol=1e-10)


class ContinuousMultiAssetKellyTests(unittest.TestCase):
    def test_public_continuous_objective_rejects_invalid_covariance(self) -> None:
        weights = np.array([0.2, 0.3])
        mu = np.array([0.05, 0.06])
        with self.assertRaises(ValueError):
            continuous_expected_log_growth(weights, mu, np.array([[0.04, 0.01], [0.0, 0.04]]))
        with self.assertRaises(ValueError):
            continuous_expected_log_growth(weights, mu, np.array([[-1e-13, 0.0], [0.0, 0.04]]))

    def test_analytic_gradient_matches_finite_difference(self) -> None:
        mu, covariance = correlated_three_asset_moments()
        weights = np.array([0.20, 0.30, 0.10])
        epsilon = 1e-6
        finite_difference = np.array(
            [
                (
                    continuous_expected_log_growth(
                        weights + epsilon * np.eye(3)[index], mu, covariance
                    )
                    - continuous_expected_log_growth(
                        weights - epsilon * np.eye(3)[index], mu, covariance
                    )
                )
                / (2.0 * epsilon)
                for index in range(3)
            ]
        )
        np.testing.assert_allclose(
            finite_difference,
            continuous_log_growth_gradient(weights, mu, covariance),
            atol=1e-8,
        )

    def test_correlated_demo_solution_and_half_kelly_scaling(self) -> None:
        mu, covariance = correlated_three_asset_moments()
        result = optimize_continuous_multivariate_kelly(mu, covariance, fractional_kelly=0.5)
        np.testing.assert_allclose(result.full_kelly_weights, [0.25, 0.25, 0.50], atol=1e-7)
        np.testing.assert_allclose(result.weights, [0.125, 0.125, 0.25], atol=1e-7)
        self.assertAlmostEqual(result.cash_weight, 0.5, places=7)

    def test_covariance_reduces_false_tech_diversification(self) -> None:
        mu, covariance = correlated_three_asset_moments()
        correlated = optimize_continuous_multivariate_kelly(mu, covariance)
        diagonal_only = optimize_continuous_multivariate_kelly(mu, np.diag(np.diag(covariance)))
        self.assertLess(
            correlated.full_kelly_weights[:2].sum(),
            diagonal_only.full_kelly_weights[:2].sum(),
        )


class SimulationTests(unittest.TestCase):
    def test_fractional_and_fixed_stake_paths(self) -> None:
        outcomes = np.array([True, False])
        np.testing.assert_allclose(
            simulate_fractional_bankroll(outcomes, 0.10, 1.0, initial_bankroll=100.0),
            [100.0, 110.0, 99.0],
        )
        np.testing.assert_allclose(
            simulate_fixed_stake_bankroll(outcomes, 5.0, 1.0, initial_bankroll=100.0),
            [100.0, 105.0, 100.0],
        )

    def test_comparison_is_reproducible_and_uses_one_outcome_sequence(self) -> None:
        first = run_coin_flip_comparison(n_trades=20, seed=9)
        second = run_coin_flip_comparison(n_trades=20, seed=9)
        np.testing.assert_array_equal(first.outcomes, second.outcomes)
        np.testing.assert_allclose(first.full_kelly_equity, second.full_kelly_equity)
        self.assertEqual(first.full_kelly_equity.size, 21)
        self.assertAlmostEqual(first.full_kelly_fraction, 0.10)
        self.assertAlmostEqual(first.half_kelly_fraction, 0.05)


class PerformanceMetricsTests(unittest.TestCase):
    def test_metrics_use_consecutive_returns_and_sample_volatility(self) -> None:
        equity = np.array([100.0, 110.0, 99.0, 108.9])
        period_returns = np.array([0.10, -0.10, 0.10])
        metrics = calculate_performance_metrics(
            equity, risk_free_rate_per_period=0.01, periods_per_year=4
        )

        expected_volatility = np.std(period_returns, ddof=1)
        expected_sharpe = np.mean(period_returns - 0.01) / expected_volatility
        expected_compound_return = np.prod(1.0 + period_returns) ** (1.0 / 3.0) - 1.0

        self.assertEqual(metrics.periods, 3)
        self.assertAlmostEqual(metrics.total_return, 0.089)
        self.assertAlmostEqual(metrics.compound_return_per_period, expected_compound_return)
        self.assertAlmostEqual(metrics.volatility_per_period, expected_volatility)
        self.assertAlmostEqual(metrics.sharpe_ratio_per_period, expected_sharpe)
        self.assertAlmostEqual(metrics.max_drawdown, 0.10)
        self.assertAlmostEqual(metrics.annualized_return, (1.0 + expected_compound_return) ** 4 - 1)
        self.assertAlmostEqual(metrics.annualized_volatility, expected_volatility * 2.0)
        self.assertAlmostEqual(metrics.annualized_sharpe_ratio, expected_sharpe * 2.0)

    def test_metrics_leave_annualized_fields_unset_without_frequency(self) -> None:
        metrics = calculate_performance_metrics([100.0, 110.0, 99.0])
        self.assertIsNone(metrics.annualized_return)
        self.assertIsNone(metrics.annualized_volatility)
        self.assertIsNone(metrics.annualized_sharpe_ratio)

    def test_seeded_demo_metrics_remain_reproducible(self) -> None:
        simulation = run_coin_flip_comparison()
        full_kelly = calculate_performance_metrics(simulation.full_kelly_equity)
        half_kelly = calculate_performance_metrics(simulation.half_kelly_equity)

        self.assertEqual(np.count_nonzero(simulation.outcomes), 536)
        self.assertAlmostEqual(full_kelly.total_return, 8.0157866221986)
        self.assertAlmostEqual(full_kelly.sharpe_ratio_per_period, 0.07215125003865851)
        self.assertAlmostEqual(full_kelly.max_drawdown, 0.9051978986309267)
        self.assertAlmostEqual(half_kelly.total_return, 9.500673510800514)
        self.assertAlmostEqual(half_kelly.max_drawdown, 0.6442108605661658)

    def test_sharpe_is_undefined_for_too_few_or_constant_returns(self) -> None:
        one_period = calculate_performance_metrics([100.0, 110.0], periods_per_year=252)
        constant_returns = calculate_performance_metrics([100.0, 110.0, 121.0])
        self.assertTrue(np.isnan(one_period.volatility_per_period))
        self.assertTrue(np.isnan(one_period.sharpe_ratio_per_period))
        self.assertTrue(np.isclose(one_period.annualized_return, 1.1**252 - 1.0, rtol=1e-12))
        self.assertTrue(np.isnan(one_period.annualized_volatility))
        self.assertTrue(np.isnan(one_period.annualized_sharpe_ratio))
        self.assertTrue(np.isnan(constant_returns.sharpe_ratio_per_period))

    def test_metrics_reject_invalid_equity_and_rate_inputs(self) -> None:
        for equity in ([100.0], [100.0, 0.0], [100.0, np.nan]):
            with self.assertRaises(ValueError):
                calculate_performance_metrics(equity)
        for risk_free_rate in (-1.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                calculate_performance_metrics(
                    [100.0, 110.0], risk_free_rate_per_period=risk_free_rate
                )
        for periods_per_year in (0.0, -1.0, np.nan, np.inf):
            with self.assertRaises(ValueError):
                calculate_performance_metrics([100.0, 110.0], periods_per_year=periods_per_year)

    def test_metrics_reject_overflowing_returns_and_invalid_drawdown_inputs(self) -> None:
        with self.assertRaises(ValueError):
            calculate_performance_metrics([np.finfo(float).tiny, np.finfo(float).max])
        for equity in (
            [],
            [[100.0, 90.0]],
            [100.0, -1.0],
            [100.0, np.nan],
            [100.0, np.inf],
        ):
            with self.assertRaises(ValueError):
                max_drawdown(equity)


if __name__ == "__main__":
    unittest.main(verbosity=2)
