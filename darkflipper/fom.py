import numpy as np
from shapely.geometry import Polygon
import matplotlib.pyplot as plt


def polygon(
    samples, param1, param2, sigma_level=1, names=None, colors=None, verbose=False
):
    """
    Calculates the Figure of Merit (FoM) for marginalised
    2-dimensional posterior distributions using the Polygon
    function from Shapely and matplotlib contour extraction.

    Parameters:
        samples (list): List of samples with 2D density grids.
        param1 (int): Index of the first parameter.
        param2 (int): Index of the second parameter.
        sigma_level (int): Contour level (1 = 68%, 2 = 95%, 3 = 99.7%).
        names (list): List of names for the samples (default is None).
        colors (list): List of colors for plotting (default is None).
        verbose (boolean): print values in screen (default is False).

    Returns:
        list: List of FoM values for each sample.
    """
    if names is None:
        names = [f"Sample {i + 1}" for i in range(len(samples))]
    if colors is None:
        colors = ["red", "blue", "green", "black"]

    sigma_lvls = {3: 0, 2: 1, 1: 2}  # Mapping for contour levels
    foms_array = []

    plt.figure(figsize=(8, 6))
    for sample, name, color in zip(samples, names, colors):
        # Get 2D density grid data
        density = sample.get2DDensityGridData(j=param1, j2=param2, num_plot_contours=3)
        contour_levels = density.contours

        # Extract contours using plt.contour
        contours = plt.contour(
            density.x, density.y, density.P, sorted(contour_levels), colors=color
        )
        contour_path = contours.collections[sigma_lvls[sigma_level]].get_paths()[0]

        # Get the vertices of the desired contour
        xy = contour_path.vertices
        poly = Polygon(xy)

        # Calculate area and FoM
        area = poly.area
        fom = (2.3 * np.pi) / area

        # Plot and print FoM
        plt.plot(xy[:, 0], xy[:, 1], label=f"{name}: FoM = {fom:.2f}", color=color)
        foms_array.append(fom)
        if verbose:
            print(f"FoM {name}: {fom}")

    plt.xlabel("Parameter 1", fontsize=16)
    plt.ylabel("Parameter 2", fontsize=16)
    plt.legend()
    plt.show()

    foms_array.append(fom)

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
        sigma_level (int): Contour level (1 = 68%, 2 = 95%, 3 = 99.7%).
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
