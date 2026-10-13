"""
===============================================================================
Bayesian Inference in Diffusion Tensor Imaging - Project Template
===============================================================================

This Python file provides the starter template for the course project in
"Advanced Probabilistic Machine Learning",
Department of Information Technology, Uppsala University.

Authors:
- Jens Sjölund (original author) - jens.sjolund@it.uu.se
- Anton O'Nils (updates & finalization) - anton.o-nils@it.uu.se
- Stina Brunzell (updates & finalization) - stina.brunzell@it.uu.se

------------------------------------------------------------------------------
Purpose
------------------------------------------------------------------------------
The project concerns Bayesian inference in diffusion MRI (dMRI), specifically
the diffusion tensor model (DTI). The goal is to estimate local tissue
properties (baseline signal S0 and diffusion tensor D) from real-world dMRI
measurements, using different Bayesian inference techniques.

Each student/group member will implement one of the following inference methods:
  1. Metropolis-Hastings
  2. Importance Sampling
  3. Variational Inference
  4. Laplace Approximation

The provided code gives:
  - Utilities for loading and preprocessing the "Stanford HARDI dataset".
  - Helper functions for matrix operations, parameterizations and gradients.
  - A skeleton structure for the prior, likelihood, and posterior approx.
  - Placeholders where each inference method should be implemented.
  - Plotting routines to visualize posterior summaries.

------------------------------------------------------------------------------
Dataset
------------------------------------------------------------------------------
The code uses the Stanford HARDI diffusion MRI dataset (Rokem et al., 2015),
accessible via DIPY's "get_fnames('stanford_hardi')".

------------------------------------------------------------------------------
Notes
------------------------------------------------------------------------------
- Several classes and methods are left as "NotImplementedError"; students are
  expected to fill these in.
- Computations are memoized with "disk_memoize" to avoid repeated costly runs.
- Results for each inference method are automatically plotted and saved.

=============================================================================
Imports
=============================================================================
Required libraries: numpy, matplotlib, scipy, dipy
Install with: pip install numpy matplotlib scipy dipy
"""

# Standard library: general utilities
import os
import pickle
import hashlib
from functools import wraps

# NumPy and Matplotlib: math and plotting
import numpy as np
import matplotlib.pyplot as plt

# SciPy: probability distributions, math functions, and optimization
# Hint: these tools might be useful later in the project
from scipy.stats import gamma, norm, wishart, multivariate_normal
from scipy.spatial.transform import Rotation
from scipy.special import logsumexp, digamma
from scipy.optimize import minimize
from scipy.special import multigammaln  # needed for Wishart logpdf for VI

# DIPY: diffusion MRI utilities and models
from dipy.io.image import load_nifti, save_nifti   # for loading / saving imaging datasets
from dipy.io.gradients import read_bvals_bvecs     # for loading / saving our bvals and bvecs
from dipy.core.gradients import gradient_table     # for constructing gradient table from bvals/bvecs
from dipy.data import get_fnames                   # for small datasets that we use in tests and examples
from dipy.segment.mask import median_otsu          # for masking out the background  # pyright: ignore[reportUnknownVariableType]
import dipy.reconst.dti as dti                     # for diffusion tensor model fitting and metrics

import argparse

"""
=============================================================================
Caching Utility (already implemented)
=============================================================================
Provides disk-based memoization to avoid recomputation.
"""

def disk_memoize(cache_dir="cache"):
    """
    Decorator for caching function outputs on disk.

    This utility is already implemented and should not be modified by students.
    It allows expensive computations to be stored and re-used across runs,
    based on the function arguments. If you call the same function again with
    the same inputs, it returns the cached results instead of recomputing.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Optionally force a fresh computation (ignores cache if True)
            force = kwargs.pop("force_recompute", False)

            # Make sure the cache directory exists
            os.makedirs(cache_dir, exist_ok=True)

            # Build a unique hash key from the function name and arguments
            func_name = func.__name__
            key = (func_name, args, kwargs)
            hash_str = hashlib.md5(pickle.dumps(key)).hexdigest()
            cache_path = os.path.join(cache_dir, f"{func_name}_{hash_str}.pkl")

            # Load the cached result if it exists (and recomputation is not forced)
            if not force and os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

            # Otherwise: compute the result, then cache it to disk
            result = func(*args, **kwargs)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)

            return result

        return wrapper
    return decorator



"""
=============================================================================
Data Loading & Preprocessing (already implemented)
=============================================================================
Loads the Stanford HARDI dataset, applies masking/cropping, and
extracts one voxel with a DTI point estimate for testing.
"""

@disk_memoize()
def get_preprocessed_data():
    """
    Load and preprocess a single voxel of diffusion MRI data.

    What it does:
    - Loads the dataset and gradient information (b-values and b-vectors).
    - Fits a diffusion tensor model (DTI) to one voxel.
    - Extracts a point estimate: baseline signal (S0), eigenvalues, eigenvectors.

    Returns
    -------
    y : ndarray
        Observed diffusion MRI signal vector for a single voxel.
    point_estimate : [S0, evals, evecs]
        Estimated baseline signal, eigenvalues, and eigenvectors.
    gtab : GradientTable
        Gradient table with b-values (diffusion weighting strength)
        and b-vectors (gradient directions).
    """

    # Load the masked data, background mask, and gradient information
    data, mask, gtab = get_data()

    # Initialize a diffusion tensor model (DTI) with S0 estimation enabled
    tenmodel = dti.TensorModel(gtab, return_S0_hat=True)

    # Extract the signal for a single voxel (coordinates chosen for this project)
    y = data[35, 35, 30, :]

    # Fit the DTI model to this voxel's signal
    tenfit = tenmodel.fit(y)

    # Extract point estimates: baseline signal, eigenvalues, and eigenvectors
    S0 = tenfit.S0_hat
    evals = tenfit.evals
    evecs = tenfit.evecs
    point_estimate = [S0, evals, evecs]

    # Return the raw voxel signal, point estimate, and gradient table
    return y, point_estimate, gtab


def get_data():
    """
    Load and preprocess the Stanford HARDI diffusion MRI dataset.

    What it does:
    - Downloads the dataset if not already present (via DIPY).
    - Loads the 4D diffusion MRI volume (x, y, z, measurements).
    - Reads b-values (diffusion weighting strength) and b-vectors (gradient directions).
    - Creates a gradient table (gtab) combining this information.
    - Applies a brain mask and cropping to remove background and reduce size.

    Returns
    -------
    maskdata : ndarray
        The masked and cropped diffusion MRI data.
    mask : ndarray (boolean)
        The brain mask used to exclude background voxels.
    gtab : GradientTable
        Gradient information (b-values and b-vectors) for each measurement.
    """

    # Download filenames for the Stanford HARDI dataset if not already cached
    hardi_fname, hardi_bval_fname, hardi_bvec_fname = get_fnames('stanford_hardi')

    # Load the raw 4D dataset: dimensions are (x, y, z, diffusion measurements)
    data, _ = load_nifti(hardi_fname)

    # Read diffusion weighting information (b-values, b-vectors) and build gradient table
    bvals, bvecs = read_bvals_bvecs(hardi_bval_fname, hardi_bvec_fname)
    gtab = gradient_table(bvals, bvecs)

    # Apply brain masking and cropping to remove background and save compute
    maskdata, mask = median_otsu(
        data, vol_idx=range(10, 50), median_radius=3, numpass=1, autocrop=True, dilate=2
    )

    # Print the final data shape for confirmation
    print('Loaded data with shape: (%d, %d, %d, %d)' % maskdata.shape)

    return maskdata, mask, gtab


"""
=============================================================================
Linear Algebra Helpers (already implemented)
=============================================================================
Functions for reconstructing tensors and switching between
parameterizations. Already implemented.
Hint: you will make use of these helpers later in the project,
the ones involving theta are useful for VI and Laplace.
"""

def compute_D(evals, V):
    """
    Reconstruct the diffusion tensor D from eigenvalues and eigenvectors.

    D = V Λ V.T, where Λ is the diagonal matrix of eigenvalues.

    Parameters
    ----------
    evals : ndarray
        Eigenvalues, shape (3,) or batched.
    V : ndarray
        Eigenvectors, shape (3, 3) or batched.

    Returns
    -------
    D : ndarray
        Diffusion tensor(s), shape (..., 3, 3).
    """

    # Ensure inputs have the correct batch dimensions
    if evals.ndim == 1:
        evals = evals[None, None, :]
    elif evals.ndim == 2:
        evals = evals[:, None, :]
    if V.ndim == 2:
        V = V[None, :, :]

    # Compute D = V Λ V.T as V (V @ Λ).T
    V_scaled = V * evals
    D = np.matmul(V, np.transpose(V_scaled, axes=[0, 2, 1]))

    return D


def theta_from_D(D):
    """
    Convert a diffusion tensor D into an unconstrained parameter vector theta.

    Follows Eq. (18): D = L L.T with L from Cholesky factorization.
    Diagonals are log-transformed, off-diagonals kept raw.

    Parameters
    ----------
    D : ndarray (3, 3)
        Symmetric positive-definite diffusion tensor.

    Returns
    -------
    theta : ndarray (6,)
        Unconstrained parameter vector corresponding to the lower-triangular
        entries of L (log of diagonals, raw off-diagonals).
    """

    # Compute Cholesky factor (lower-triangular L) of D
    L = np.linalg.cholesky(D)

    # Indices of lower-triangular entries (including diagonal)
    p = D.shape[0]
    tril_indices = np.tril_indices(p)
    theta = []

    # Store log of diagonal entries, raw off-diagonal entries
    for i, j in zip(*tril_indices):
        if i == j:
            theta.append(np.log(L[i, j]))   # Diagonal: log-transform
        else:
            theta.append(L[i, j])           # Off-diagonal: raw value

    return np.array(theta)


def D_from_theta(theta):
    """
    Convert unconstrained parameter vector theta back into diffusion tensor D.

    Follows Eq. (18): D = L L.T with L constructed from theta.
    Diagonal entries of L are exponentiated to ensure positivity,
    off-diagonals are used as raw values.

    Parameters
    ----------
    theta : ndarray (..., 6)
        Unconstrained parameters corresponding to the lower-triangular
        entries of L (log-diagonals, raw off-diagonals).

    Returns
    -------
    D : ndarray (..., 3, 3)
        Symmetric positive-definite diffusion tensor(s).
    """

    # Ensure theta is an array and check shape
    theta = np.asarray(theta)
    *batch_shape, _ = theta.shape
    assert theta.shape[-1] == 6, "Last dimension must be 6 for 3x3 lower-triangular matrices."

    # Initialize lower-triangular matrix L
    L = np.zeros((*batch_shape, 3, 3), dtype=theta.dtype)

    # Fill L with exponentiated diagonals and raw off-diagonals
    tril_indices = np.tril_indices(3)
    for k, (i, j) in enumerate(zip(*tril_indices)):
        if i == j:
            L[..., i, j] = np.exp(theta[..., k])   # Diagonal
        else:
            L[..., i, j] = theta[..., k]           # Off-diagonal

    # Reconstruct D = L @ L.T (batch-aware matrix multiplication)
    D = L @ np.swapaxes(L, -1, -2)

    return D.squeeze()


def grad_D_wrt_theta_at_D(D):
    """
    Compute nabla_theta D evaluated at D.

    Uses the parameterization in Eq. (18): D = L L.T where L is built from
    theta. Returns the gradient tensor with one (3x3) slice per theta component.

    Parameters
    ----------
    D : ndarray (3, 3)
        Symmetric positive-definite diffusion tensor.

    Returns
    -------
    grad_D : ndarray (3, 3, 6)
        Gradient of D w.r.t. theta, one 3x3 matrix per parameter.
    """

    # Get Cholesky factor of D and set up indices for lower-triangular entries
    p = D.shape[0]
    L = np.linalg.cholesky(D)
    tril_indices = np.tril_indices(p)
    num_params = len(tril_indices[0])

    # Prepare output container
    grad_D = np.zeros((p, p, num_params))

    # Loop over all parameters in theta
    for k, (m, n) in enumerate(zip(*tril_indices)):

        # Build a basis matrix for the effect of this parameter
        E_mn = np.zeros((p, p))
        if m == n:
            # Diagonal: dL_mm/dtheta = L_mm since L_mm = exp(theta)
            factor = L[m, n]
        else:
            # Off-diagonal: dL_mn/dtheta = 1
            factor = 1.0
        E_mn[m, n] = factor

        # Work out the corresponding change in D
        dD_k = E_mn @ L.T + L @ E_mn.T
        grad_D[:, :, k] = dD_k

    return grad_D


"""
=============================================================================
Bayesian Model Components (need to be implemented) - done by MK
=============================================================================
Students: implement all parts in this section (priors, likelihoods, etc.)
These are required before any inference method can be attempted.
"""

class frozen_prior:
    # Placeholder for the prior distribution.
    # Hint: you may want to add input parameters to these methods.

    def __init__(self):
        self.alpha_S = 2
        self.theta_S = 500
        self.alpha_l = 4
        self.theta_l = 2.5e-4


    def rvs(self, size=1):
        # generate samples from the prior
        S0 = gamma.rvs(a=self.alpha_S, scale=self.theta_S, size=size)
        lambdas = gamma.rvs(a=self.alpha_l, scale=self.theta_l, size=(size, 3))
        V = Rotation.random(num=size).as_matrix()  # random rotation matrix

        return S0, lambdas, V

    def logpdf(self, S0, lambdas, V):
        logpdf_S0 = gamma.logpdf(S0, a=self.alpha_S, scale=self.theta_S)
        logpdf_lambdas = np.sum(gamma.logpdf(lambdas, a=self.alpha_l, scale=self.theta_l))
        logpdf_V = 0  # uniform over SO(3), constant logpdf

        return logpdf_S0 + logpdf_lambdas + logpdf_V


class frozen_likelihood:
    # Placeholder for the likelihood (with partial code provided).
    # Hint: you may want to add input parameters to these methods.

    def __init__(self, gtab, y, sigma=29):
        self.gtab = gtab   # store gradient table with b-values and b-vectors
        self.y = y         # store observed data
        self.sigma = sigma # store noise level


    def logpdf(self, S0, evecs, evals):
        S0 = np.atleast_1d(S0)        # ensure S0 is array-like
        D = compute_D(evals, evecs)   # reconstruct diffusion tensor

        # Build q from diffusion gradients (b-values & b-vectors),
        # corresponds to the experimental setting x in the project instructions
        q = np.sqrt(self.gtab.bvals[:, None]) * self.gtab.bvecs

        # Model signal S given tensor D and baseline S0
        S = S0[:, None] * np.exp( - np.einsum('...j, ijk, ...k->i...', q, D, q))

        # Compute log-likelihood under Gaussian noise model
        log_likelihood = np.sum(norm.logpdf(self.y, loc=S, scale=self.sigma))

        return log_likelihood



"""
=============================================================================
Posterior Approximations (need to be implemented)
=============================================================================
Students: implement these approximations, which are only used in the
corresponding inference methods below:
  - variational_posterior: used only for Variational Inference
  - mvn_reparameterized: used only for Laplace Approximation

They are NOT needed for Metropolis-Hastings or Importance Sampling.
"""

class variational_posterior:
    """
    Variational posterior  q(S0, D) = Gamma(S0; alpha, beta) * W(D; Sigma, nu)   (Eqs. 15-17).

    Unconstrained parameter vector theta (9,), same order as the provided score():
        theta[0]   = log alpha                 (Gamma shape)
        theta[1]   = log beta                  (Gamma scale)
        theta[2:8] = Cholesky parameters of Sigma, Eq. (18) / D_from_theta
        theta[8]   = log(nu - 2)               (nu = exp(theta[8]) + 2 > 2)

    Wishart convention (the one assumed by the provided wishart_score):
        W = nu * D ~ Wishart(df=nu, scale=Sigma)   =>   E[D] = Sigma.
    The density of D therefore carries the Jacobian factor nu^(p(p+1)/2).
    Positive-definiteness of every sample is guaranteed by construction
    (Bartlett decomposition), and of Sigma by the Cholesky parameterization.
    The Wishart is implemented by hand because scipy.stats.wishart requires
    df >= 3 while nu in (2, 3) is allowed by the parameterization.
    """

    p = 3  # dimension of D

    def __init__(self, theta):
        theta = np.asarray(theta, dtype=float).ravel()
        if theta.shape != (9,):
            raise ValueError(f"theta must have 9 entries, got shape {theta.shape}")
        self.theta = theta
        self.shape = float(np.exp(theta[0]))          # alpha
        self.scale = float(np.exp(theta[1]))          # beta
        self.Sigma = D_from_theta(theta[2:8])         # (3, 3), SPD
        self.df = float(np.exp(theta[8]) + 2.0)       # nu

    # ------------------------------------------------------------------ #
    @staticmethod
    def theta_from_moments(S0, D, rel_sd_S0=0.05, nu=200.0):
        """
        Build theta from a point estimate: E[S0] = S0 with coefficient of
        variation rel_sd_S0 (Gamma: CV = 1/sqrt(alpha)), E[D] = D, df = nu.
        D is projected onto the SPD cone first (eigenvalues floored at 1e-5),
        since a least-squares tensor fit can have non-positive eigenvalues.
        """
        w, U = np.linalg.eigh(np.asarray(D, dtype=float))
        D_spd = (U * np.maximum(w, 1e-5)) @ U.T
        D_spd = 0.5 * (D_spd + D_spd.T)
        alpha = 1.0 / rel_sd_S0 ** 2
        beta = max(float(S0), 1e-3) / alpha
        return np.concatenate([[np.log(alpha), np.log(beta)],
                               theta_from_D(D_spd),
                               [np.log(nu - 2.0)]])

    # ------------------------------------------------------------------ #
    def _wishart_logpdf(self, D):
        """log density of D = W/nu, W ~ Wishart(nu, Sigma). D: (..., 3, 3)."""
        p, nu = self.p, self.df
        W = nu * np.asarray(D, dtype=float)
        sign, logdet_W = np.linalg.slogdet(W)
        _, logdet_Sigma = np.linalg.slogdet(self.Sigma)
        tr = np.einsum('ij,...ji->...', np.linalg.inv(self.Sigma), W)    # tr(Sigma^-1 W)
        out = (0.5 * (nu - p - 1) * logdet_W - 0.5 * tr
               - 0.5 * nu * p * np.log(2.0) - 0.5 * nu * logdet_Sigma
               - multigammaln(0.5 * nu, p)
               + 0.5 * p * (p + 1) * np.log(nu))                           # dW = nu^6 dD
        return np.where(sign > 0, out, -np.inf)

    def logpdf(self, S0, D):
        """log q(S0, D) = log Gamma(S0; alpha, beta) + log W(D). Batched over leading dims."""
        S0 = np.asarray(S0, dtype=float)
        return gamma.logpdf(S0, a=self.shape, scale=self.scale) + self._wishart_logpdf(D)

    # ------------------------------------------------------------------ #
    def sample(self, size, rng=None):
        """Draw (S0, D): S0 (size,), D (size, 3, 3), D symmetric positive definite."""
        rng = np.random.mtrand._rand if rng is None else rng   # global state -> respects np.random.seed
        S0 = rng.gamma(shape=self.shape, scale=self.scale, size=size)
        # Bartlett decomposition (valid for real nu > p - 1):  W = (Lc B)(Lc B)^T
        B = np.zeros((size, self.p, self.p))
        for i in range(self.p):
            B[:, i, i] = np.sqrt(rng.chisquare(self.df - i, size=size))
        rows, cols = np.tril_indices(self.p, k=-1)
        B[:, rows, cols] = rng.standard_normal((size, len(rows)))
        A = np.linalg.cholesky(self.Sigma) @ B
        D = A @ np.swapaxes(A, -1, -2) / self.df
        return S0, 0.5 * (D + np.swapaxes(D, -1, -2))

    def rvs(self, size, rng=None):
        """
        Samples in the format used by plot_results: S0 (size,), evals (size, 3)
        and evecs (size, 3, 3) from the eigendecomposition D = V diag(evals) V^T.
        np.linalg.eigh returns ascending eigenvalues, so the principal
        eigenvector is evecs[:, :, 2] (what plot_results expects). V is in SO(3).
        """
        S0_samples, D = self.sample(size, rng)
        evals_samples, evecs_samples = np.linalg.eigh(D)
        flip = np.linalg.det(evecs_samples) < 0
        evecs_samples[flip, :, 0] *= -1          # det = +1  ->  V in SO(3)
        return S0_samples, evals_samples, evecs_samples

    # ------------------------------------------------------------------ #
    def score(self, S0, D):
        """grad_theta log q_theta(S0, D) for ONE sample (S0 scalar/(1,), D (3,3)), shape (9,)."""
        # Combine score contributions from gamma and Wishart parts
        score_wrt_log_shape, score_wrt_log_scale = self.gamma_score(S0)
        score_wrt_theta, score_wrt_log_df = self.wishart_score(D)
        # FIX: the gamma / df parts are 0-d for scalar input and cannot be concatenated as-is
        return np.concatenate([
            np.atleast_1d(score_wrt_log_shape), np.atleast_1d(score_wrt_log_scale),
            np.atleast_1d(score_wrt_theta), np.atleast_1d(score_wrt_log_df)])

    def gamma_score(self, x):
        # Score function for gamma distribution
        score_wrt_log_shape = (np.log(x / self.scale) - digamma(self.shape)) * self.shape
        score_wrt_log_scale = (x / self.scale**2 - self.shape / self.scale) * self.scale
        return score_wrt_log_shape, score_wrt_log_scale

    def wishart_score(self, D):
        # Score function for Wishart distribution
        W = self.df * D
        Sigma_inv = np.linalg.inv(self.Sigma)
        score_wrt_Sigma = 0.5 * Sigma_inv @ (W - self.df * self.Sigma) @ Sigma_inv
        score_wrt_theta = np.tensordot(
            score_wrt_Sigma, grad_D_wrt_theta_at_D(self.Sigma), axes=([0,1], [0,1])
        )
        p = W.shape[0]
        _, logdet_W = np.linalg.slogdet(W)
        _, logdet_Sigma = np.linalg.slogdet(self.Sigma)
        digamma_sum = np.sum([digamma((self.df + 1 - j) / 2.0) for j in range(1, p+1)])
        # FIX: the variable of q is D = W/nu (not W), so d/d(nu) at fixed D has the extra
        # term -(tr(Sigma^-1 D) - p)/2 (from W = nu*D and the Jacobian nu^(p(p+1)/2)).
        # Verified against finite differences in vi_sanity_checks().
        score_wrt_log_df = ((self.df - 2) / 2) * (
            logdet_W - p * np.log(2) - logdet_Sigma - digamma_sum
            - (np.trace(Sigma_inv @ D) - p))
        return score_wrt_theta, score_wrt_log_df


class mvn_reparameterized:
    """Gaussian approximation in the unconstrained Laplace parameter space.

    The seven parameters are ``log(S0)`` followed by the six parameters of
    the Cholesky factor of ``D`` used by :func:`D_from_theta`.  Samples are
    transformed back to the physical variables expected by ``plot_results``.
    """

    def __init__(self, mean, cov, optimizer_result=None, hessian=None):
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.asarray(cov, dtype=float)
        if self.mean.shape != (7,):
            raise ValueError("The Laplace mean must contain seven parameters.")
        if self.cov.shape != (7, 7):
            raise ValueError("The Laplace covariance must have shape (7, 7).")

        # Store diagnostics so convergence and curvature can be reported.
        self.optimizer_result = optimizer_result
        self.hessian = hessian
        self._mvn = multivariate_normal(
            mean=self.mean, cov=self.cov, allow_singular=False
        )

    def logpdf(self, theta):
        """Evaluate the Gaussian density in the unconstrained theta space."""
        return self._mvn.logpdf(theta)

    def rvs(self, size=1, random_state=None):
        """Draw samples and transform them to ``S0``, eigenvalues and vectors."""
        theta_samples = self._mvn.rvs(size=size, random_state=random_state)
        theta_samples = np.atleast_2d(theta_samples)

        S0_samples = np.exp(theta_samples[:, 0])
        D_samples = D_from_theta(theta_samples[:, 1:])
        D_samples = np.asarray(D_samples)
        if D_samples.ndim == 2:
            D_samples = D_samples[None, :, :]

        # eigh returns ascending eigenvalues.  Hence column 2 is the principal
        # eigenvector, matching the convention already used by plot_results.
        evals_samples, evecs_samples = np.linalg.eigh(D_samples)
        return S0_samples, evals_samples, evecs_samples




"""
=============================================================================
Inference Methods (need to be implemented)
=============================================================================
Students: implement one method each (MH, IS, VI, or Laplace).
Uses memoization to speed up repeated runs.
"""
# Help functions for Metropolis hastings
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


@disk_memoize()
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



#Help functions for IS
def log_abs_det_z_wrt_theta(theta):
    """
    log |d(S0,D) / d(theta)|
    """

    return (
        np.log(8.0)
        + theta[0]
        + 4.0 * theta[1]
        + 3.0 * theta[3]
        + 2.0 * theta[6]
    )



@disk_memoize()
def importance_sampling(n_samples, gamma_param, nu_param,
                        proposal="original", random_seed=0):
    # Fixed reference point for the proposal
    y, point_estimate, gtab = get_preprocessed_data(force_recompute=False)
    S0_ref, evals_ref, evecs_ref = point_estimate
    D_ref = evecs_ref @ np.diag(evals_ref) @ evecs_ref.T

    D_init = compute_D(evals_ref, evecs_ref).squeeze()
    prior=frozen_prior()
    frozen_prior_instance = frozen_prior()
    likelihood=frozen_likelihood(gtab, y)
    rng = np.random.default_rng(random_seed)

    # Check proposal parameters.
    #if proposal not in ("original", "laplace"):
     #   raise ValueError("proposal must be either 'original' or 'mh'.")
    if gamma_param <= 0:
        raise ValueError("gamma_param must be > 0.")
    if nu_param <= 2:
        raise ValueError("For a 3x3 Wishart distribution, nu_param must be > 2.")
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer.")
    if proposal not in ("original", "laplace"):
        raise ValueError("proposal must be 'original' or 'laplace'.")



    if proposal == "original":
        # Gamma proposal
        gamma_shape = gamma_param ** (-2)
        gamma_scale = gamma_param ** 2 * S0_ref

        # Wishart proposal 
        wishart_scale = D_ref / nu_param

        S0_samples = gamma.rvs(
            a=gamma_shape,
            scale=gamma_scale,
            size=n_samples,
            random_state=rng,
        )

        D_samples = wishart.rvs(
            df=nu_param,
            scale=wishart_scale,
            size=n_samples,
            random_state=rng,
        )

        D_samples = np.asarray(D_samples)

        if D_samples.ndim == 2:
            D_samples = D_samples[None, :, :]

        # q(S0, D) in physical coordinates
        log_q_S0 = gamma.logpdf(S0_samples, a=gamma_shape, scale=gamma_scale)
        log_q_D = np.array([wishart.logpdf(D_samples[i], df=nu_param, scale=wishart_scale) for i in range(n_samples)])
        log_q = log_q_S0 + log_q_D

    else:
        # Laplace Gaussian q_L(theta)
        laplace = laplace_approximation(force_recompute=False)

        # Sample directly in the seven-dimensional theta-space.
        theta_samples = rng.multivariate_normal(mean=laplace.mean, cov=laplace.cov, size=n_samples,)
        theta_samples = np.atleast_2d(theta_samples)

        # Transform theta -> S0, D.
        S0_samples = np.exp(theta_samples[:, 0])

        D_samples = np.asarray(D_from_theta(theta_samples[:, 1:]))

        if D_samples.ndim == 2:
            D_samples = D_samples[None, :, :]

        # Density of the actual proposal, in theta-space.
        log_q = np.asarray(laplace.logpdf(theta_samples)).reshape(-1)

    # Convert D samples to eigenvalues/eigenvectors
    evals_samples, evecs_samples = np.linalg.eigh(D_samples)


    #  Evaluate the original target density
    # log_joint_D includes the original prior, likelihood, and the eigenvalue-to-D density correction.

    # The result is a density in (S0, D) coordinates.
    log_target_z = log_joint_D(S0_samples, D_samples, prior, likelihood, include_jacobian=True)



    # Calculate importance weights

    if proposal == "original":
        # Target and proposal are both densities in (S0, D).
        log_unnormalized_weights = log_target_z - log_q

    else:
        # Convert target density from (S0, D) to theta-space.
        log_jacobian = np.array([
            log_abs_det_z_wrt_theta(theta)
            for theta in theta_samples
        ])
        log_target_theta = log_target_z + log_jacobian
        # Both target and proposal are now densities in theta-space.
        log_unnormalized_weights = log_target_theta - log_q

    # Normalize weights stably
    log_normalizer = logsumexp(log_unnormalized_weights)

    if not np.isfinite(log_normalizer):
        raise RuntimeError(
            "All importance weights are zero or invalid. "
            "Check the proposal and target densities.")

    log_weights = (log_unnormalized_weights - log_normalizer)

    importance_weights = np.exp(log_weights)

    # Effective sample size
    ess = 1 / np.sum(importance_weights ** 2)

    print(
        f"Proposal: {proposal}\n"
        f"ESS: {ess:.1f} / {n_samples} "
        f"({ess / n_samples:.2%})"
    )

    return importance_weights, S0_samples, evals_samples, evecs_samples, ess




# helper functions for VI

def log_joint_D(S0, D, prior, likelihood, include_jacobian=True):
    """
    Vectorized log p(y, S0, D) for a batch of K samples, parameterized by the
    diffusion tensor D itself (needed because q(D) is a distribution over D).

    The model (Eqs. 6-9) places its prior on (S0, lambda, V), not on D.
    Mapping the prior on (lambda, V) to a density w.r.t. Lebesgue measure on
    the symmetric matrix D requires the change-of-variables Jacobian
        dD = prod_{i<j} |lambda_i - lambda_j| dlambda dV,
    hence   log p(D) = sum_i log Gamma(lambda_i) - sum_{i<j} log|lambda_i-lambda_j| + const.
    (V is uniform => constant.) Set include_jacobian=False to drop that term
    (i.e. to treat the Gamma prior as if it were directly a prior on the
    eigenvalues of D); both options are reported in main().

    The sum over measurements is kept per sample, unlike frozen_likelihood.logpdf
    which sums over everything (its value for one sample is checked against this
    function in vi_sanity_checks).

    Parameters
    ----------
    S0 : (K,) or scalar     D : (K, 3, 3) or (3, 3)
    Returns
    -------
    (K,) array (up to an additive constant that does not depend on S0, D).
    """
    S0 = np.atleast_1d(np.asarray(S0, dtype=float))
    D = np.asarray(D, dtype=float)
    if D.ndim == 2:
        D = D[None]
    evals = np.linalg.eigvalsh(D)                                   # (K, 3), ascending

    # Likelihood: same forward model as frozen_likelihood.logpdf
    q = np.sqrt(likelihood.gtab.bvals[:, None]) * likelihood.gtab.bvecs   # (M, 3)
    quad = np.einsum('mi,kij,mj->km', q, D, q)                      # (K, M)  x^T D x
    S = S0[:, None] * np.exp(-quad)
    loglik = np.sum(norm.logpdf(likelihood.y, loc=S, scale=likelihood.sigma), axis=1)

    # Prior
    logprior = gamma.logpdf(S0, a=prior.alpha_S, scale=prior.theta_S)
    logprior = logprior + np.sum(
        gamma.logpdf(evals, a=prior.alpha_l, scale=prior.theta_l), axis=1)
    if include_jacobian:
        gaps = np.stack([evals[:, 1] - evals[:, 0],
                         evals[:, 2] - evals[:, 1],
                         evals[:, 2] - evals[:, 0]], axis=1)
        logprior = logprior - np.sum(np.log(np.maximum(gaps, 1e-300)), axis=1)

    return loglik + logprior






@disk_memoize()
def variational_inference(max_iters, K, learning_rate, y, gtab, S0_init, D_init,
                          seed=0, include_jacobian=True, sigma=29,
                          rel_sd_S0=0.05, nu_init=200.0, avg_frac=0.2, verbose=True):
    """
    Fit q(S0, D) = Gamma x Wishart by maximizing the ELBO with the REINFORCE
    leave-one-out gradient estimator (Eqs. 19-20) and Adam (gradient ASCENT).

    Parameters
    ----------
    max_iters, K, learning_rate : number of Adam steps, MC samples per step, base step size.
    y, gtab, sigma              : voxel signal, gradient table, noise std (likelihood).
    S0_init, D_init             : point estimate used to initialize q (D_init need not be SPD).
    seed                        : seed of the local RNG (results are reproducible).
    include_jacobian            : see log_joint_D.
    rel_sd_S0, nu_init          : initial width of q(S0) (coefficient of variation) and df of q(D).
    avg_frac                    : the returned theta is the average over the last avg_frac of the
                                  iterations (Polyak-Ruppert averaging; REINFORCE is noisy).

    Returns
    -------
    variational_posterior with extra attributes
        .elbo_history  (max_iters,)  per-iteration MC estimate of the ELBO (mean of f over the K samples)
        .theta_history (max_iters, 9)
        .theta_last    parameters at the last iteration (before averaging)
        .final_elbo    ELBO re-estimated with 20000 fresh samples at the returned parameters
    """
    prior = frozen_prior()
    likelihood = frozen_likelihood(gtab, y, sigma=sigma)
    rng = np.random.default_rng(seed)

    theta = variational_posterior.theta_from_moments(S0_init, D_init, rel_sd_S0, nu_init)

    # Adam state. The theta entries live on very different scales: log-parameters are O(1)
    # but the *off-diagonal* Cholesky entries (theta[3], theta[5], theta[6]) have the
    # scale of sqrt(diffusivity) ~ 0.03. A common step size would wreck them, so their
    # step is scaled by the typical Cholesky diagonal (equivalent to optimizing
    # theta_k / scale_k; Adam is otherwise invariant to the gradient scale).
    lr_scale = np.ones(9)
    L0 = np.linalg.cholesky(D_from_theta(theta[2:8]))
    lr_scale[[3, 5, 6]] = np.mean(np.diag(L0))
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m, v = np.zeros(9), np.zeros(9)

    elbo_hist = np.full(max_iters, np.nan)
    theta_hist = np.zeros((max_iters, 9))
    n_skipped = 0

    for t in range(1, max_iters + 1):
        q = variational_posterior(theta)
        S0_k, D_k = q.sample(K, rng)

        # f(z) = log p(D, z) - log q(z)
        f = log_joint_D(S0_k, D_k, prior, likelihood, include_jacobian) - q.logpdf(S0_k, D_k)
        elbo_hist[t - 1] = np.mean(f)
        theta_hist[t - 1] = theta
        if not np.all(np.isfinite(f)):          # should not happen; never let NaNs into Adam
            n_skipped += 1
            continue

        # REINFORCE leave-one-out: f_k - mean_{j != k} f_j   (Eq. 20)
        f_loo = f - (np.sum(f) - f) / (K - 1)
        scores = np.stack([q.score(S0_k[k:k + 1], D_k[k]) for k in range(K)])     # (K, 9)
        grad = np.mean(f_loo[:, None] * scores, axis=0)                          # (9,)

        # Adam ascent step with cosine-decayed learning rate (to 10 % of the base value)
        lr_t = learning_rate * (0.1 + 0.45 * (1 + np.cos(np.pi * t / max_iters)))
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat, v_hat = m / (1 - beta1 ** t), v / (1 - beta2 ** t)
        theta = theta + lr_t * lr_scale * m_hat / (np.sqrt(v_hat) + eps)

        if verbose and (t % max(1, max_iters // 10) == 0):
            print(f"[VI] iter {t:6d}/{max_iters}  ELBO (last 100 it.) = "
                  f"{np.nanmean(elbo_hist[max(0, t - 100):t]):12.2f}")

    # Polyak-Ruppert average of the iterates over the last avg_frac of the run
    n_avg = max(1, int(avg_frac * max_iters))
    posterior = variational_posterior(theta_hist[-n_avg:].mean(axis=0))
    posterior.theta_last = theta
    posterior.elbo_history = elbo_hist
    posterior.theta_history = theta_hist
    posterior.n_skipped = n_skipped

    S0_e, D_e = posterior.sample(20000, rng)
    posterior.final_elbo = float(np.mean(
        log_joint_D(S0_e, D_e, prior, likelihood, include_jacobian) - posterior.logpdf(S0_e, D_e)))
    return posterior 



@disk_memoize()
def laplace_approximation(max_iters=500, hessian_step=1e-4):
    """Fit a Laplace approximation to the posterior in transformed space.

    ``theta[0]`` is ``log(S0)`` and ``theta[1:]`` parameterizes the Cholesky
    factor of the diffusion tensor.  The posterior mode is found with
    L-BFGS-B.  A central finite-difference Hessian of the negative log
    posterior is then inverted to obtain the Gaussian covariance.

    We follow Eq. (21) literally: evaluate log p(z(theta) | data), without
    adding a change-of-variables Jacobian. logpdf on the returned object
    evaluates the fitted Gaussian in theta coordinates.
    """
    y, point_estimate, gtab = get_preprocessed_data(force_recompute=False)
    S0_init, evals_init, evecs_init = point_estimate
    D_init = compute_D(evals_init, evecs_init).squeeze()

    prior = frozen_prior()
    likelihood = frozen_likelihood(gtab, y)

    theta_init = np.concatenate([
        [np.log(float(S0_init))],
        theta_from_D(D_init),
    ])

    def negative_log_posterior(theta):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != (7,) or not np.all(np.isfinite(theta)):
            return 1e100

        # Avoid overflow in exp while leaving a very broad feasible range.
        if theta[0] < -20 or theta[0] > 20:
            return 1e100

        try:
            S0 = np.exp(theta[0])
            D = D_from_theta(theta[1:])
            evals, evecs = np.linalg.eigh(D)
            if np.any(evals <= 0) or not np.all(np.isfinite(evals)):
                return 1e100

            log_post = prior.logpdf(S0, evals, evecs)
            log_post += likelihood.logpdf(S0, evecs, evals)
        except (FloatingPointError, ValueError, np.linalg.LinAlgError):
            return 1e100

        if not np.isfinite(log_post):
            return 1e100
        return -np.asarray(log_post).item()

    # Bound only transformed positive quantities.  Off-diagonal Cholesky
    # entries remain unconstrained as required by equation (18).
    bounds = [(-20, 20)]
    for k in range(6):
        bounds.append((-20, 5) if k in (0, 2, 5) else (None, None))

    result = minimize(
        negative_log_posterior,
        theta_init,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": max_iters, "ftol": 1e-12, "gtol": 1e-7},
    )
    if not result.success or not np.isfinite(result.fun) or result.fun >= 1e99:
        raise RuntimeError(f"Laplace MAP optimization failed: {result.message}")

    theta_map = np.asarray(result.x, dtype=float)
    n_params = theta_map.size
    steps = hessian_step * np.maximum(1.0, np.abs(theta_map))
    hessian = np.zeros((n_params, n_params), dtype=float)
    f0 = negative_log_posterior(theta_map)

    # Central finite differences give the observed information matrix at MAP.
    for i in range(n_params):
        ei = np.zeros(n_params)
        ei[i] = steps[i]
        hessian[i, i] = (
            negative_log_posterior(theta_map + ei)
            - 2.0 * f0
            + negative_log_posterior(theta_map - ei)
        ) / steps[i] ** 2

        for j in range(i + 1, n_params):
            ej = np.zeros(n_params)
            ej[j] = steps[j]
            value = (
                negative_log_posterior(theta_map + ei + ej)
                - negative_log_posterior(theta_map + ei - ej)
                - negative_log_posterior(theta_map - ei + ej)
                + negative_log_posterior(theta_map - ei - ej)
            ) / (4.0 * steps[i] * steps[j])
            hessian[i, j] = value
            hessian[j, i] = value

    hessian = 0.5 * (hessian + hessian.T)
    if not np.all(np.isfinite(hessian)):
        raise RuntimeError("The finite-difference Hessian contains invalid values.")

    # Numerical finite differences can introduce tiny negative eigenvalues.
    # Project the observed information to a well-conditioned SPD matrix before
    # inversion, while retaining the unregularized Hessian for diagnostics.
    curvature, basis = np.linalg.eigh(hessian)
    max_curvature = max(float(np.max(np.abs(curvature))), 1.0)
    curvature_floor = 1e-8 * max_curvature
    if np.min(curvature) < -curvature_floor:
        raise RuntimeError("The mode has negative curvature; cannot fit Laplace.")
    curvature_regularized = np.maximum(curvature, curvature_floor)
    covariance = (basis * (1.0 / curvature_regularized)) @ basis.T
    covariance = 0.5 * (covariance + covariance.T)

    posterior = mvn_reparameterized(
        theta_map,
        covariance,
        optimizer_result=result,
        hessian=hessian,
    )
    posterior.log_posterior_at_mode = -f0
    posterior.hessian_eigenvalues = curvature
    posterior.hessian_was_regularized = bool(np.any(curvature < curvature_floor))
    return posterior






"""
=============================================================================
Visualization & Experiment Runner
=============================================================================
Plotting function and the main() script to run experiments.
"""

def main():
    # Initialize with preprocessed data and DTI point estimate
    # (these values can be used as starting points for inference methods)
    y, point_estimate, gtab = get_preprocessed_data(force_recompute=False)
    S0_init, evals_init, evecs_init = point_estimate
    D_init = compute_D(evals_init, evecs_init).squeeze()

    # Find principal eigenvector from DTI estimate (for plotting)
    evec_principal = evecs_init[:, 0]
    frozen_prior_instance = frozen_prior()

    print("Frozen pror", np.round(frozen_prior_instance.logpdf(S0_init, evals_init, evecs_init),3))

    frozen_likelihood_instance = frozen_likelihood(gtab, y)
    print("Frozen likelihood",np.round(frozen_likelihood_instance.logpdf(S0_init, evecs_init, evals_init),3))

    # Set random seed and number of posterior samples
    np.random.seed(0)
    n_samples = 10000

    # # Run Metropolis–Hastings and plot results
    S0_mh, evals_mh, evecs_mh, diagnostics = metropolis_hastings(
        n_samples=10000,
        gamma_param=0.01,
        nu_param=2000,
        random_seed=0,
        proposal_mode="mean_centered",
        force_recompute=False,
    )
    burn_in = 0
    plot_results(S0_mh[burn_in:], evals_mh[burn_in:], evecs_mh[burn_in:, :, :], evec_principal, method="mh")


    # Run Importance Sampling and plot results
    gamma=[0.01, 0.05,0.1,0.2,0.5]
    nu=[4, 10, 100, 1000, 2000]
    #for g in gamma:
     #   print("Gamma= ", g)
      #  for n in nu:
       #   print("nu= ", n)
        #  w_is, S0_is, evals_is, evecs_is, ess = importance_sampling(n_samples, gamma_param=g, nu_param=n, proposal="original", random_seed=0, force_recompute=True)

    # Original Gamma x Wishart proposal
    w_original, S0_original, evals_original, evecs_original, ess_original = importance_sampling(n_samples=n_samples,
        gamma_param=0.01, nu_param=1000, proposal="original", random_seed=0, force_recompute=True)

    # Laplace Gaussian proposal
    w_laplace, S0_laplace, evals_laplace, evecs_laplace, ess_laplace = importance_sampling(n_samples=n_samples,
        gamma_param=0.01, nu_param=1000, proposal="laplace", random_seed=0, force_recompute=True)

    plot_results(S0_original, evals_original, evecs_original, evec_principal, weights=w_original, method="is")
    plt.savefig("result_importance_original.png")

    plot_results(S0_laplace, evals_laplace, evecs_laplace, evec_principal, weights=w_laplace, method="is")
    plt.savefig("result_importance_laplace.png")



    print("Running Laplace Approximation....")
     # Run the assigned method: Laplace approximation
    posterior_laplace = laplace_approximation(force_recompute=False)
    if posterior_laplace.optimizer_result is not None:
        print("Laplace optimizer success:", posterior_laplace.optimizer_result.success)
        print("Laplace optimizer message:", posterior_laplace.optimizer_result.message)
    print("Laplace MAP theta:", np.round(posterior_laplace.mean, 6))
    print(
        "Hessian regularized:",
        posterior_laplace.hessian_was_regularized,
    )
    S0_laplace, evals_laplace, evecs_laplace = posterior_laplace.rvs(
        size=n_samples, random_state=0
    )

    plot_results(S0_laplace, evals_laplace, evecs_laplace, evec_principal, method="laplace")

     # Run Variational Inference and plot results

    print("Running Variational Inference....")
    posterior_vi = variational_inference(max_iters=2000, K=100, learning_rate=0.01,
                                         y=y, gtab=gtab, S0_init=S0_init, D_init=D_init,
                                         seed=0, include_jacobian=True, sigma=29,
                                         rel_sd_S0=0.05, nu_init=200.0, avg_frac=0.2,
                                         verbose=True)
    S0_vi, evals_vi, evecs_vi = posterior_vi.rvs(size=n_samples)
    plot_results(S0_vi, evals_vi, evecs_vi, evec_principal, method="vi")
    plot_learning_curve(posterior_vi.elbo_history, hyperparams={"max_iters": 2000, "K": 100, "learning_rate": 0.01}, method="vi")


    print("Done.")


def plot_learning_curve(learning_curve, hyperparams, method="vi"):
    """
    Plot the learning curve of the ELBO estimate over iterations.

    Parameters
    ----------
    learning_curve : list or ndarray
        List of ELBO estimates at each iteration.
    method : str
        Name of inference method (used in output filename).
    """
    max_iters = hyperparams["max_iters"]
    K = hyperparams["K"]
    learning_rate = hyperparams["learning_rate"]

    plt.figure(figsize=(8, 6))
    plt.plot(learning_curve, label='ELBO Estimate', color='blue')
    plt.xlabel('Iteration')
    plt.ylabel('ELBO Estimate')
    plt.title(f'Learning Curve for {method.upper()} with max_iters={max_iters}, K={K}, learning_rate={learning_rate}')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"learning_curve_{method}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_results(S0, evals, evecs, evec_ref, weights=None, method=""):
    """
    Plot posterior results as histograms and save to file.

    Creates histograms of baseline signal (S0), mean diffusivity (MD),
    fractional anisotropy (FA), and the angle between estimated and
    reference eigenvectors.

    Parameters
    ----------
    S0 : ndarray
        Sampled baseline signals.
    evals : ndarray
        Sampled eigenvalues of the diffusion tensor.
    evecs : ndarray
        Sampled eigenvectors of the diffusion tensor.
    evec_ref : ndarray
        Reference principal eigenvector (from point estimate).
    weights : ndarray, optional
        Importance weights for samples. Uniform if None.
    method : str
        Name of inference method (used in output filename).
    """

    # Use uniform weights if none provided
    if weights is None:
        weights = np.ones_like(S0)
        weights /= np.sum(weights)

    # Choose number of bins based on sample size
    n_bins = np.floor(np.sqrt(len(weights))).astype(int)

    # Squeeze arrays for plotting
    weights = weights.squeeze()
    S0 = S0.squeeze()
    md = dti.mean_diffusivity(evals).squeeze()
    fa = dti.fractional_anisotropy(evals).squeeze()

    # Compute acute angle between estimated and reference eigenvectors
    angle = 360/(2*np.pi) * np.arccos(np.abs(np.dot(evecs[:, :, 2], evec_ref)))

    # Create 2x2 grid of histograms
    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharey=False)

    axes[0, 0].hist(S0, bins=n_bins, density=True, weights=weights,
                    alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_xlabel("S0")
    axes[0, 0].set_ylabel("Density")

    axes[0, 1].hist(md, bins=n_bins, density=True, weights=weights,
                    alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel("Mean diffusivity")
    axes[0, 1].set_ylabel("Density")

    axes[1, 0].hist(fa, bins=n_bins, density=True, weights=weights,
                     alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_xlabel("Fractional anisotropy")
    axes[1, 0].set_ylabel("Density")

    axes[1, 1].hist(angle, bins=n_bins, density=True, weights=weights,
                    alpha=0.7, color='magenta', edgecolor='black')
    axes[1, 1].set_xlabel("Acute angle")
    axes[1, 1].set_ylabel("Density")

    # Adjust layout and save figure with method name
    plt.tight_layout()
    plt.savefig("results_{}.png".format(method), dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
