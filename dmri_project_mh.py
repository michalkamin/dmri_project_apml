"""Metropolis-Hastings inference for the APML diffusion tensor project.

This module reuses the shared data, prior, likelihood, and tensor helpers from
``MK_dmri_project.py`` and implements the state-dependent proposal specified in
the project instructions.  It can also tune proposal parameters and create the
posterior and trace figures used in the report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.stats import gamma, wishart

from MK_dmri_project import (
    compute_D,
    disk_memoize,
    frozen_likelihood,
    frozen_prior,
    get_preprocessed_data,
)


def _eigendecomposition(D):
    """Return ascending eigenvalues/eigenvectors, matching ``np.linalg.eigh``."""
    evals, evecs = np.linalg.eigh(D)
    return np.asarray(evals, dtype=float), np.asarray(evecs, dtype=float)


def _wishart_scale(D, nu_param, proposal_mode):
    """Return the SciPy Wishart scale matrix for the requested convention."""
    if proposal_mode == "instruction":
        # Equation (13): q(D' | D) = W(D, nu), whose mean in SciPy is nu * D.
        return D
    if proposal_mode == "mean_centered":
        # Useful diagnostic alternative: E[D' | D] = nu * (D / nu) = D.
        return D / nu_param
    raise ValueError("proposal_mode must be 'instruction' or 'mean_centered'.")


def _log_posterior(S0, D, prior, likelihood):
    """Evaluate the unnormalized posterior at a scalar S0 and SPD tensor D."""
    evals, evecs = _eigendecomposition(D)
    if S0 <= 0 or np.any(evals <= 0):
        return -np.inf, evals, evecs

    value = prior.logpdf(S0, evals, evecs)
    value += likelihood.logpdf(S0, evecs, evals)
    value = float(np.asarray(value).item())
    return value, evals, evecs


@disk_memoize(cache_dir="cache/mh")
def metropolis_hastings(
    n_samples,
    gamma_param,
    nu_param,
    random_seed=0,
    proposal_mode="instruction",
):
    """Sample the posterior using the assignment's asymmetric MH proposal.

    Parameters
    ----------
    n_samples : int
        Total chain length, including the DTI initialization.
    gamma_param : float
        Spread parameter for the Gamma proposal for S0.
    nu_param : float
        Degrees of freedom of the 3x3 Wishart proposal; must exceed 2.
    random_seed : int
        Seed used for reproducible proposal and acceptance draws.
    proposal_mode : {"instruction", "mean_centered"}
        ``instruction`` follows Eq. (13) literally with SciPy scale ``D``.
        ``mean_centered`` uses scale ``D / nu`` as a documented diagnostic
        alternative whose conditional mean is the current tensor.

    Returns
    -------
    S0_samples, evals_samples, evecs_samples, diagnostics
        Full chain plus acceptance indicators and log-posterior trace.
    """
    if int(n_samples) != n_samples or n_samples < 2:
        raise ValueError("n_samples must be an integer of at least 2.")
    if gamma_param <= 0:
        raise ValueError("gamma_param must be positive.")
    if nu_param <= 2:
        raise ValueError("nu_param must be greater than 2 for a 3x3 Wishart.")

    n_samples = int(n_samples)
    rng = np.random.default_rng(random_seed)

    y, point_estimate, gtab = get_preprocessed_data(force_recompute=False)
    S0_init, evals_init, evecs_init = point_estimate
    D_init = np.asarray(compute_D(evals_init, evecs_init).squeeze(), dtype=float)

    prior = frozen_prior()
    likelihood = frozen_likelihood(gtab, y)

    current_S0 = float(np.asarray(S0_init).item())
    current_D = D_init
    current_log_post, current_evals, current_evecs = _log_posterior(
        current_S0, current_D, prior, likelihood
    )
    if not np.isfinite(current_log_post):
        raise RuntimeError("The DTI initialization has a non-finite posterior.")

    S0_samples = np.empty(n_samples)
    evals_samples = np.empty((n_samples, 3))
    evecs_samples = np.empty((n_samples, 3, 3))
    log_posterior_trace = np.empty(n_samples)
    accepted = np.zeros(n_samples, dtype=bool)

    S0_samples[0] = current_S0
    evals_samples[0] = current_evals
    evecs_samples[0] = current_evecs
    log_posterior_trace[0] = current_log_post

    gamma_shape = gamma_param ** -2

    for i in range(1, n_samples):
        forward_gamma_scale = gamma_param**2 * current_S0
        proposed_S0 = float(
            gamma.rvs(
                a=gamma_shape,
                scale=forward_gamma_scale,
                random_state=rng,
            )
        )

        forward_wishart_scale = _wishart_scale(
            current_D, nu_param, proposal_mode
        )
        proposed_D = np.asarray(
            wishart.rvs(
                df=nu_param,
                scale=forward_wishart_scale,
                random_state=rng,
            ),
            dtype=float,
        )
        proposed_D = 0.5 * (proposed_D + proposed_D.T)

        proposed_log_post, proposed_evals, proposed_evecs = _log_posterior(
            proposed_S0, proposed_D, prior, likelihood
        )

        if np.isfinite(proposed_log_post):
            reverse_gamma_scale = gamma_param**2 * proposed_S0
            reverse_wishart_scale = _wishart_scale(
                proposed_D, nu_param, proposal_mode
            )

            log_q_forward = gamma.logpdf(
                proposed_S0,
                a=gamma_shape,
                scale=forward_gamma_scale,
            )
            log_q_forward += wishart.logpdf(
                proposed_D,
                df=nu_param,
                scale=forward_wishart_scale,
            )

            log_q_reverse = gamma.logpdf(
                current_S0,
                a=gamma_shape,
                scale=reverse_gamma_scale,
            )
            log_q_reverse += wishart.logpdf(
                current_D,
                df=nu_param,
                scale=reverse_wishart_scale,
            )

            log_acceptance_ratio = (
                proposed_log_post
                - current_log_post
                + log_q_reverse
                - log_q_forward
            )

            if np.log(rng.uniform()) < min(0.0, log_acceptance_ratio):
                current_S0 = proposed_S0
                current_D = proposed_D
                current_log_post = proposed_log_post
                current_evals = proposed_evals
                current_evecs = proposed_evecs
                accepted[i] = True

        # Rejections deliberately repeat the current state in the Markov chain.
        S0_samples[i] = current_S0
        evals_samples[i] = current_evals
        evecs_samples[i] = current_evecs
        log_posterior_trace[i] = current_log_post

    diagnostics = {
        "accepted": accepted,
        "acceptance_rate": float(np.mean(accepted[1:])),
        "log_posterior": log_posterior_trace,
        "gamma": float(gamma_param),
        "nu": float(nu_param),
        "proposal_mode": proposal_mode,
        "random_seed": int(random_seed),
    }
    return S0_samples, evals_samples, evecs_samples, diagnostics


def posterior_summaries(S0, evals, evecs, evec_ref, burn_in):
    """Compute report-ready summaries after discarding burn-in."""
    if burn_in < 0 or burn_in >= len(S0):
        raise ValueError("burn_in must be between 0 and n_samples - 1.")

    S0 = np.asarray(S0[burn_in:])
    evals = np.asarray(evals[burn_in:])
    evecs = np.asarray(evecs[burn_in:])
    md = np.mean(evals, axis=-1)
    denominator = np.sum(evals**2, axis=-1)
    fa = np.sqrt(
        1.5 * np.sum((evals - md[:, None]) ** 2, axis=-1) / denominator
    )
    cosine = np.abs(np.einsum("ni,i->n", evecs[:, :, 2], evec_ref))
    angle = np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))

    def summarize(values):
        return {
            "mean": float(np.mean(values)),
            "sd": float(np.std(values, ddof=1)),
            "q025": float(np.quantile(values, 0.025)),
            "q975": float(np.quantile(values, 0.975)),
        }

    return {
        "S0": summarize(S0),
        "MD": summarize(md),
        "FA": summarize(fa),
        "angle_degrees": summarize(angle),
    }


def effective_sample_size(values):
    """Estimate MCMC effective sample size using an initial positive sequence."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("values must be a one-dimensional array of length >= 3.")
    centered = values - np.mean(values)
    if np.allclose(centered, 0):
        return float(values.size)

    n = values.size
    spectrum = np.fft.rfft(centered, n=2 * n)
    autocorrelation = np.fft.irfft(spectrum * np.conjugate(spectrum))[:n]
    autocorrelation /= autocorrelation[0]
    paired = autocorrelation[1:-1:2] + autocorrelation[2::2]
    first_nonpositive = np.flatnonzero(paired <= 0)
    stop = int(first_nonpositive[0]) if first_nonpositive.size else paired.size
    integrated_time = max(1.0, 1.0 + 2.0 * np.sum(paired[:stop]))
    return float(n / integrated_time)


def plot_mh_results(
    S0,
    evals,
    evecs,
    evec_ref,
    diagnostics,
    burn_in,
    output_dir="APML_report",
):
    """Create posterior and trace plots and return their paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    S0_post = np.asarray(S0[burn_in:])
    evals_post = np.asarray(evals[burn_in:])
    evecs_post = np.asarray(evecs[burn_in:])
    md = np.mean(evals_post, axis=-1)
    fa = np.sqrt(
        1.5
        * np.sum((evals_post - md[:, None]) ** 2, axis=-1)
        / np.sum(evals_post**2, axis=-1)
    )
    cosine = np.abs(np.einsum("ni,i->n", evecs_post[:, :, 2], evec_ref))
    angle = np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))

    values = [S0_post, md, fa, angle]
    labels = [
        r"Baseline signal $S_0$",
        "Mean diffusivity",
        "Fractional anisotropy",
        "Acute angle (degrees)",
    ]
    colors = ["#c44e52", "#55a868", "#4c72b0", "#8172b2"]
    n_bins = max(20, int(np.sqrt(len(S0_post))))

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    for axis, samples, label, color in zip(
        axes.ravel(), values, labels, colors
    ):
        axis.hist(
            samples,
            bins=n_bins,
            density=True,
            color=color,
            alpha=0.8,
            edgecolor="white",
        )
        axis.set_xlabel(label)
        axis.set_ylabel("Density")
        axis.xaxis.set_major_locator(MaxNLocator(5))
        if label == "Mean diffusivity":
            axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
        axis.grid(alpha=0.2)
    fig.suptitle(
        "Metropolis-Hastings posterior "
        f"(acceptance {diagnostics['acceptance_rate']:.1%})"
    )
    fig.tight_layout()
    posterior_path = output_dir / "results_mh.png"
    fig.savefig(posterior_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    full_md = np.mean(evals, axis=-1)
    full_fa = np.sqrt(
        1.5
        * np.sum((evals - full_md[:, None]) ** 2, axis=-1)
        / np.sum(evals**2, axis=-1)
    )
    traces = [S0, full_md, full_fa, diagnostics["log_posterior"]]
    trace_labels = [r"$S_0$", "MD", "FA", "Log posterior"]

    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    for axis, trace, label in zip(axes, traces, trace_labels):
        axis.plot(trace, linewidth=0.55, color="#4c72b0")
        axis.axvline(burn_in, color="#c44e52", linestyle="--", linewidth=1)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Iteration")
    fig.suptitle(
        "Metropolis-Hastings traces "
        f"($\\gamma$={diagnostics['gamma']:g}, "
        f"$\\nu$={diagnostics['nu']:g})"
    )
    fig.tight_layout()
    trace_path = output_dir / "trace_mh.png"
    fig.savefig(trace_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return posterior_path, trace_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--gamma", type=float, default=0.01)
    parser.add_argument("--nu", type=float, default=2000.0)
    parser.add_argument("--burn-in", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--proposal-mode",
        choices=("instruction", "mean_centered"),
        default="mean_centered",
    )
    parser.add_argument("--output-dir", default="APML_report")
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args()

    S0, evals, evecs, diagnostics = metropolis_hastings(
        n_samples=args.samples,
        gamma_param=args.gamma,
        nu_param=args.nu,
        random_seed=args.seed,
        proposal_mode=args.proposal_mode,
        force_recompute=args.force_recompute,
    )
    _, point_estimate, _ = get_preprocessed_data(force_recompute=False)
    evec_ref = point_estimate[2][:, 0]
    summaries = posterior_summaries(S0, evals, evecs, evec_ref, args.burn_in)
    posterior_path, trace_path = plot_mh_results(
        S0,
        evals,
        evecs,
        evec_ref,
        diagnostics,
        args.burn_in,
        args.output_dir,
    )

    print(f"Acceptance rate: {diagnostics['acceptance_rate']:.3f}")
    print(f"Burn-in: {args.burn_in}")
    for name, values in summaries.items():
        print(
            f"{name}: mean={values['mean']:.8g}, sd={values['sd']:.8g}, "
            f"95% CI=[{values['q025']:.8g}, {values['q975']:.8g}]"
        )
    md = np.mean(evals[args.burn_in :], axis=-1)
    print(
        "Approximate ESS: "
        f"S0={effective_sample_size(S0[args.burn_in:]):.1f}, "
        f"MD={effective_sample_size(md):.1f}"
    )
    print(f"Saved {posterior_path}")
    print(f"Saved {trace_path}")


if __name__ == "__main__":
    main()
