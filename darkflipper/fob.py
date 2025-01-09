import numpy as np
from getdist import MCSamples

def parameter_bias(samples_unbiased, samples_biased, param1, param2):
    """
    Calculate the Figure of Bias (FoB) from parameter shifts 
    between biased and unbiased MCMC samples using best-fit 
    values for two specified parameters.

    Parameters:
        samples_unbiased: MCMC samples for the unbiased model.
        samples_biased: MCMC samples for the biased model.
        param1 (str): Name of the first parameter.
        param2 (str): Name of the second parameter.

    Returns:
        float: FoB from parameter shifts.
    """
    best_fit_unbiased = samples_unbiased.getBestFit()
    best_fit_biased = samples_biased.getBestFit()

    delta_theta = np.array([
        best_fit_biased.params[param1] - best_fit_unbiased.params[param1],
        best_fit_biased.params[param2] - best_fit_unbiased.params[param2]
    ])

    cov_matrix = samples_unbiased.getCovarianceMatrix([param1, param2]).matrix
    cov_matrix_inv = np.linalg.inv(cov_matrix)

    fob = np.sqrt(np.dot(delta_theta.T, np.dot(cov_matrix_inv, delta_theta)))
    print(f"FoB (Parameter Bias): {fob}")

    return fob

def chi2_bias(samples_unbiased, samples_biased):
    """
    Calculate the Figure of Bias (FoB) from the chi-squared difference 
    between biased and unbiased MCMC samples.

    Parameters:
        samples_unbiased: MCMC samples for the unbiased model.
        samples_biased: MCMC samples for the biased model.

    Returns:
        float: FoB from chi-squared difference.
    """
    loglike_unbiased = samples_unbiased.loglikes.mean()
    loglike_biased = samples_biased.loglikes.mean()

    delta_chi2 = 2 * (loglike_biased - loglike_unbiased)
    fob_chi2 = np.sqrt(delta_chi2)

    print(f"FoB (Chi-squared Bias): {fob_chi2}")

    return fob_chi2
