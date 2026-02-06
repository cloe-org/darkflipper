import numpy as np
from scipy import stats


def pvalue(data, cov, mask=None, mult=1.0):
    """Calculate p-value for chi-squared test."""
    if np.any(mask):
        n_data = len(mask)
    else:
        n_data = len(data)
        mask = np.full(n_data, True)
    chi2 = mult * np.dot(
        data[mask], np.dot(np.linalg.inv(cov[mask, :][:, mask]), data[mask])
    )
    p = stats.chi2.sf(chi2, n_data)
    return p


def mahalanobis_distance(data, mean, cov):
    """Calculate Mahalanobis distance."""
    diff = data - mean
    inv_cov = np.linalg.inv(cov)
    return np.sqrt(np.dot(diff, np.dot(inv_cov, diff)))


def test_statistic(data, cov, mean=None):
    """Calculate chi-squared test statistic."""
    if mean is None:
        mean = np.zeros_like(data)
    diff = data - mean
    inv_cov = np.linalg.inv(cov)
    return np.dot(diff, np.dot(inv_cov, diff))


def critical_value(alpha, df):
    """Get critical value for chi-squared distribution."""
    return stats.chi2.ppf(1 - alpha, df)


def effect_size(chi2, n):
    """Calculate effect size from chi-squared statistic."""
    return np.sqrt(chi2 / n)
