"""Figure of Merit (FoM) estimators for 2D marginalised posteriors."""

import contourpy
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from shapely.geometry import Polygon


def polygon(
    samples,
    param1,
    param2,
    sigma_level=1,
    names=None,
    colors=None,
    verbose=False,
    plot=True,
):
    """
    Calculates the Figure of Merit (FoM) for marginalised
    2-dimensional posterior distributions from the area enclosed by
    the GetDist contour, computed with the Polygon function from Shapely.

    The FoM is normalised so that, for a Gaussian posterior, it equals
    1/sqrt(det(C)) for any contour level, i.e. it agrees with
    ``covariance_matrix``: FoM = pi * Δχ²(p) / area, where p is the
    probability enclosed by the contour and Δχ²(p) its 2-dof quantile.

    Parameters:
        samples (list): List of GetDist MCSamples objects.
        param1 (str or int): Name or index of the first parameter.
        param2 (str or int): Name or index of the second parameter.
        sigma_level (int): Contour level, indexing the GetDist ``contours``
            setting (1 = 68%, 2 = 95%, 3 = 99% with the GetDist defaults).
        names (list): List of names for the samples (default is None).
        colors (list): List of colors for plotting (default is None).
        verbose (boolean): print values in screen (default is False).
        plot (boolean): plot the contours used (default is True).

    Returns:
        list: List of FoM values for each sample.
    """
    if names is None:
        names = [f"Sample {i + 1}" for i in range(len(samples))]
    if colors is None:
        colors = ["red", "blue", "green", "black"]

    foms_array = []

    if plot:
        plt.figure(figsize=(8, 6))
    for sample, name, color in zip(samples, names, colors):
        # Get 2D density grid data; levels are ordered as sample.contours
        density = sample.get2DDensityGridData(
            j=param1, j2=param2, num_plot_contours=sigma_level
        )
        level = density.contours[sigma_level - 1]
        enclosed_prob = sample.contours[sigma_level - 1]

        # Extract the contour lines at the desired level
        lines = contourpy.contour_generator(density.x, density.y, density.P).lines(
            level
        )

        # Sum the areas of all (possibly disjoint) contour regions
        area = sum(Polygon(xy).area for xy in lines if len(xy) >= 3)

        # Calculate FoM
        delta_chi2 = stats.chi2.ppf(enclosed_prob, df=2)
        fom = np.pi * delta_chi2 / area

        if plot:
            for i, xy in enumerate(lines):
                plt.plot(
                    xy[:, 0],
                    xy[:, 1],
                    label=f"{name}: FoM = {fom:.2f}" if i == 0 else None,
                    color=color,
                )
        foms_array.append(fom)
        if verbose:
            print(f"FoM {name}: {fom}")

    if plot:
        plt.xlabel(str(param1), fontsize=16)
        plt.ylabel(str(param2), fontsize=16)
        plt.legend()
        plt.show()

    return foms_array


def covariance_matrix(
    samples, param1, param2, sigma_level=1, names=None, verbose=False
):
    """
    Calculates the Figure of Merit (FoM) for marginalised
    2-dimensional posterior distributions using the determinant of the
    covariance matrix approach.

    Parameters:
        samples (list): List of GetDist MCSamples objects.
        param1 (str): Name of the first parameter.
        param2 (str): Name of the second parameter.
        sigma_level (int): Unused, kept for API symmetry with ``polygon``.
        names (list): List of names for the samples (default is None).
        verbose (boolean): print values in screen (default is False).

    Returns:
        list: List of FoM values for each sample.
    """
    if names is None:
        names = [f"Sample {i + 1}" for i in range(len(samples))]

    foms_array = []

    for sample, name in zip(samples, names):
        # Get the covariance matrix for the specified parameters
        cov_matrix = sample.cov([param1, param2])
        det_cov = np.linalg.det(cov_matrix)

        # Calculate the Figure of Merit (FoM) using det(cov)
        fom = 1 / np.sqrt(det_cov)
        if verbose:
            print(f"FoM {name}: {fom}")
        foms_array.append(fom)

    return foms_array
