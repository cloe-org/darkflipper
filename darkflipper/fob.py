import warnings

import numpy as np


def parameter_bias(samples_unbiased, samples_biased, index, params, verbose=False):
    """
    Calculate the Figure of Bias (FoB) from parameter shifts
    between biased and unbiased MCMC samples using best-fit
    values for two specified parameters.

    Parameters:
        samples_unbiased: MCMC samples for the unbiased model.
        samples_biased: MCMC samples for the biased model.
        index (list): Indices (or names) of the parameters in the chain.
        params (list): Names of the parameters (used for printing only).
        verbose (boolean): print values in screen (default is False).

    Returns:
        float: FoB from parameter shifts.
    """
    # 1. Get parameter means (accept names as well as indices)
    index = [
        samples_unbiased.paramNames.numberOfName(i) if isinstance(i, str) else i
        for i in index
    ]
    means_unbiased = samples_unbiased.getMeans(pars=index)
    means_biased = samples_biased.getMeans(pars=index)
    # 2. Compute parameter shifts (Δθ)
    delta_theta = means_biased - means_unbiased
    # 3. Get the covariance matrix of these parameters from the unbiased model
    cov_matrix = samples_unbiased.cov(pars=index)
    # 4. Calculate the Figure of Bias (FoB): sqrt(Δθᵀ C⁻¹ Δθ)
    fob = np.sqrt(delta_theta @ np.linalg.solve(cov_matrix, delta_theta))
    if verbose:
        print(f"FoB (Parameter Bias) for {params}: {fob}")
    return fob


def chi2_bias(samples_unbiased, samples_biased, verbose=False):
    """
    Calculate the Figure of Bias (FoB) from the chi-squared difference
    between biased and unbiased MCMC samples.

    Parameters:
        samples_unbiased: MCMC samples for the unbiased model.
        samples_biased: MCMC samples for the biased model.
        verbose (boolean): print values in screen (default is False).

    Returns:
        float: FoB from chi-squared difference. NaN (with a warning) if the
        biased samples have a lower mean χ² than the unbiased ones.

    Notes:
        GetDist ``loglikes`` store -log(L), so Δχ² = 2 (⟨-log L_biased⟩ -
        ⟨-log L_unbiased⟩).
    """
    # 1. Compute the average log-likelihoods
    loglike_unbiased = np.mean(samples_unbiased.loglikes)
    loglike_biased = np.mean(samples_biased.loglikes)

    # 2. Calculate Δχ²
    delta_chi2 = 2 * (loglike_biased - loglike_unbiased)

    # 3. Compute FoB from Δχ²
    if delta_chi2 < 0:
        warnings.warn(
            f"Negative Δχ² ({delta_chi2:.3g}): the biased samples fit better "
            "than the unbiased ones, FoB is undefined.",
            stacklevel=2,
        )
        return np.nan
    fob_chi2 = np.sqrt(delta_chi2)
    if verbose:
        print(f"FoB (Chi-squared Bias): {fob_chi2}")
    return fob_chi2
