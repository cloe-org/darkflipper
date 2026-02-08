import numpy as np
from scipy import stats


def _chi_squared_statistic(data, theory, cov):
    """
    Calculate the chi-squared statistic for goodness-of-fit testing.

    This function computes the chi-squared test statistic by calculating
    the quadratic form: (data - theory)^T * cov^(-1) * (data - theory),
    which measures the discrepancy between observed data and theoretical
    predictions, weighted by the inverse covariance matrix.

    Parameters
    ----------
    data : array-like
        Observed data values.
    theory : array-like
        Theoretical or model-predicted values.
    cov : array-like
        Covariance matrix of the data. Must be invertible.

    Returns
    -------
    float
        The chi-squared statistic value.

    Raises
    ------
    np.linalg.LinAlgError
        If the covariance matrix is singular and cannot be inverted.

    Notes
    -----
    The chi-squared statistic assumes the covariance matrix is positive definite.
    For better numerical stability with ill-conditioned matrices, consider using
    np.linalg.solve instead of matrix inversion.

    Examples
    --------
    >>> data = np.array([1.0, 2.0, 3.0])
    >>> theory = np.array([1.1, 1.9, 3.1])
    >>> cov = np.eye(3)
    >>> chi2 = _chi_squared_statistic(data, theory, cov)
    """
    diff = data - theory
    inv_cov = np.linalg.inv(cov)
    return diff @ inv_cov @ diff


def pvalue(data, theory, cov):
    """
    Calculate the p-value for a chi-squared test.

    Computes the chi-squared statistic and returns the survival function (1 - CDF)
    to determine the probability of observing a test statistic at least as extreme
    as the one calculated from the data.

    Parameters
    ----------
    data : array-like
        Observed data values.
    theory : array-like
        Theoretical or expected values corresponding to the data.
    cov : array-like
        Covariance matrix for the chi-squared statistic calculation.

    Returns
    -------
    float
        The p-value representing the probability of observing the chi-squared
        statistic or a more extreme value under the null hypothesis.

    Notes
    -----
    The degrees of freedom are determined by the length of the data array.
    The chi-squared statistic is computed using the provided covariance matrix.

    """
    dof = len(data)  # degrees of freedom
    data - theory
    chi2 = _chi_squared_statistic(data, theory, cov)
    p = stats.chi2.sf(chi2, dof)
    return p


def theoretical_prediction(cov, n_samples):
    """
    Calculate the theoretical prediction based on covariance matrix and sample size.
    Computes the trace of the covariance matrix normalized by the number of samples.
    This is commonly used in statistical analysis to estimate the average variance
    or energy of a system.
    Args:
        cov (np.ndarray): A covariance matrix of shape (d, d) where d is the number of features.
        n_samples (int): The number of samples used to compute the covariance matrix.
    Returns:
        float: The normalized trace of the covariance matrix (trace(cov) / n_samples).
    Examples:
        >>> cov = np.array([[2.0, 0.0], [0.0, 2.0]])
        >>> n_samples = 100
        >>> theoretical_prediction(cov, n_samples)
        0.04
    """

    return np.trace(cov) / n_samples


def mahalanobis_distance(data, theory, cov):
    """
    Calculate the Mahalanobis distance between data and theoretical values.
    The Mahalanobis distance is a measure of the distance between a point and a
    distribution, taking into account the correlations of the data set. It is
    computed as the square root of the chi-squared statistic.
    Parameters
    ----------
    data : array-like
        The observed data points.
    theory : array-like
        The theoretical or expected values.
    cov : array-like
        The covariance matrix of the data.
    Returns
    -------
    float or ndarray
        The Mahalanobis distance(s) between the data and theory.
    Notes
    -----
    The Mahalanobis distance is computed as:
        D = sqrt(chi2)
    where chi2 is the chi-squared statistic calculated from the data,
    theory, and covariance matrix.
    Examples
    --------
    >>> data = np.array([1, 2, 3])
    >>> theory = np.array([1.1, 2.1, 2.9])
    >>> cov = np.eye(3)
    >>> distance = mahalanobis_distance(data, theory, cov)
    """

    chi2 = _chi_squared_statistic(data, theory, cov)
    return np.sqrt(chi2)


def effect_size(data, theory, cov):
    """
    Calculate the effect size based on chi-squared statistic.
    Parameters
    ----------
    data : array-like
        Observed data values.
    theory : array-like
        Theoretical or expected values.
    cov : array-like
        Covariance matrix used in chi-squared calculation.
    Returns
    -------
    float
        The effect size computed as the square root of the chi-squared statistic
        divided by the sample size (normalized chi-squared).
    Notes
    -----
    Effect size is calculated using the formula: sqrt(chi2 / n)
    where chi2 is the chi-squared test statistic and n is the number of observations.
    """

    chi2 = _chi_squared_statistic(data, theory, cov)
    n = len(data)
    return np.sqrt(chi2 / n)
