"""Focused tests for the Metropolis-Hastings implementation."""

import unittest

import numpy as np

from dmri_project_mh import (
    _wishart_scale,
    effective_sample_size,
    metropolis_hastings,
    posterior_summaries,
)


class MetropolisHastingsTests(unittest.TestCase):
    def test_mean_centered_wishart_scale(self):
        D = np.diag([1.0, 2.0, 3.0])
        nu = 20.0
        np.testing.assert_allclose(nu * _wishart_scale(D, nu, "mean_centered"), D)

    def test_instruction_wishart_scale(self):
        D = np.diag([1.0, 2.0, 3.0])
        np.testing.assert_allclose(_wishart_scale(D, 5.0, "instruction"), D)

    def test_invalid_parameters_fail_before_loading_data(self):
        with self.assertRaises(ValueError):
            metropolis_hastings(1, 0.1, 3.0, force_recompute=True)
        with self.assertRaises(ValueError):
            metropolis_hastings(10, 0.0, 3.0, force_recompute=True)
        with self.assertRaises(ValueError):
            metropolis_hastings(10, 0.1, 2.0, force_recompute=True)

    def test_posterior_summaries(self):
        S0 = np.array([10.0, 11.0, 12.0, 13.0])
        evals = np.tile(np.array([1.0, 2.0, 3.0]), (4, 1))
        evecs = np.tile(np.eye(3), (4, 1, 1))
        summaries = posterior_summaries(S0, evals, evecs, np.array([0, 0, 1]), 1)
        self.assertAlmostEqual(summaries["S0"]["mean"], 12.0)
        self.assertAlmostEqual(summaries["angle_degrees"]["mean"], 0.0)

    def test_effective_sample_size_for_independent_signal(self):
        rng = np.random.default_rng(7)
        values = rng.normal(size=2000)
        ess = effective_sample_size(values)
        self.assertGreater(ess, 1000)
        self.assertLessEqual(ess, len(values))


if __name__ == "__main__":
    unittest.main()
