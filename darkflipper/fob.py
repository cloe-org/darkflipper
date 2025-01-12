import numpy as np
from getdist import MCSamples

def parameter_bias(samples_unbiased, samples_biased, index, params, verbose = False):
    """
    Calculate the Figure of Bias (FoB) from parameter shifts 
    between biased and unbiased MCMC samples using best-fit 
    values for two specified parameters.

    Parameters:
        samples_unbiased: MCMC samples for the unbiased model.
        samples_biased: MCMC samples for the biased model.
        params (list): Names of the parameters.
        index (list): Index in the chain corresponding to parameters.
        verbose (boolean): print values in screen (default is False).

    Returns:
        float: FoB from parameter shifts.
    """
    # 1. Get parameter means
    means_unbiased = samples_unbiased.getMeans(pars = index)
    means_biased = samples_biased.getMeans(pars = index)
    # 2. Compute parameter shifts (Δθ)
    delta_theta = means_biased - means_unbiased
    # 3. Get the covariance matrix from the unbiased model
    cov_matrix = samples_unbiased.cov()
    # 4. Invert the covariance matrix (C⁻¹)
    cov_matrix_inv = np.linalg.inv(cov_matrix)
    # 5. Calculate the Figure of Bias (FoB) from parameter shifts
    fob = np.sqrt(np.dot(delta_theta.T, np.dot(cov_matrix_inv, delta_theta)))
    if verbose:
        print(f"FoB (Parameter Bias): {fob}")
    return fob


def chi2_bias(samples_unbiased, samples_biased, verbose = False):
    """
    Calculate the Figure of Bias (FoB) from the chi-squared difference 
    between biased and unbiased MCMC samples.

    Parameters:
        samples_unbiased: MCMC samples for the unbiased model.
        samples_biased: MCMC samples for the biased model.
        verbose (boolean): print values in screen (default is False).

    Returns:
        float: FoB from chi-squared difference.
    """
    # 1. Compute the average log-likelihoods
    loglike_unbiased = np.mean(samples_unbiased.loglikes)
    loglike_biased = np.mean(samples_biased.loglikes)

    # 2. Calculate Δχ²
    delta_chi2 = 2 * (loglike_biased - loglike_unbiased)

    # 3. Compute FoB from Δχ²
    fob_chi2 = np.sqrt(delta_chi2)
    if verbose:
        print(f"FoB (Chi-squared Bias): {fob_chi2}")
    return fob_chi2

