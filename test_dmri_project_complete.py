"""Regression checks for the integrated runner's reported MH results."""

import unittest
from unittest.mock import patch, Mock

import numpy as np

import dmri_project_complete as project


class IntegratedRunnerTests(unittest.TestCase):
    def test_burn_in_is_applied_without_changing_other_methods(self):
        n = 1004
        S0 = np.r_[np.ones(1000), [10.0, 12.0, 10.0, 12.0]]
        evals = np.tile([1e-4, 2e-4, 3e-4], (n, 1))
        evecs = np.tile(np.eye(3), (n, 1, 1))
        point = (11.0, evals[0, ::-1], evecs[0, :, ::-1])
        diagnostics = {"acceptance_rate": 0.5}
        posterior = Mock()
        posterior.rvs.return_value = (S0, evals, evecs)
        posterior.optimizer_result = None
        posterior.mean = np.zeros(7)
        posterior.hessian_was_regularized = False

        with patch.object(project, "get_preprocessed_data", return_value=(np.ones(3), point, Mock())), \
             patch.object(project, "frozen_prior") as prior, \
             patch.object(project, "frozen_likelihood") as likelihood, \
             patch.object(project, "metropolis_hastings", return_value=(S0, evals, evecs, diagnostics)) as mh, \
             patch.object(project, "posterior_summaries", wraps=project.posterior_summaries) as summaries, \
             patch.object(project, "importance_sampling", return_value=(np.ones(n) / n, S0, evals, evecs, 100.0)) as importance, \
             patch.object(project, "laplace_approximation", return_value=posterior) as laplace, \
             patch.object(project, "variational_inference", return_value=(posterior, [0.0])) as vi, \
             patch("dmri_project_mh.plot_mh_results") as plot_mh, \
             patch.object(project, "plot_results"), \
             patch.object(project, "plot_learning_curve"), \
             patch("builtins.print") as output:
            prior.return_value.logpdf.return_value = 0.0
            likelihood.return_value.logpdf.return_value = 0.0
            project.main()

        self.assertEqual(summaries.call_args.args[4], 1000)
        self.assertEqual(plot_mh.call_args.args[5], 1000)
        self.assertTrue(mh.call_args.kwargs["force_recompute"])
        self.assertFalse(laplace.call_args.kwargs["force_recompute"])
        self.assertTrue(importance.call_args.kwargs["force_recompute"])
        self.assertTrue(vi.call_args.kwargs["force_recompute"])
        printed = [call.args[0] for call in output.call_args_list]
        self.assertIn("MH S0: mean=11, 95% CI=[10, 12]", printed)


if __name__ == "__main__":
    unittest.main()
