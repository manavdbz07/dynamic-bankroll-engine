"""Regression tests for the from-scratch Kelly bankroll engine.

Run with:
    python3 -m unittest -v test_bankroll_engine.py
"""

from __future__ import annotations

import unittest

import numpy as np

from bankroll_engine import (
    binary_return_distribution,
    continuous_expected_log_growth,
    continuous_log_growth_gradient,
    expected_log_growth,
    kelly_fraction,
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
