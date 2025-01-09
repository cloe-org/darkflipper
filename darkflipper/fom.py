import numpy as np
from shapely.geometry import Polygon
from getdist import MCSamples

def polygon(samples, param1, param2, sigma_level=1, names=None):
    """
    Calculates the Figure of Merit (FoM) for marginalised 
    2-dimensional posterior distributions using Polygon 
    shapely function

    Parameters:
        samples (list): List of samples with 2D density grids.
        param1 (int): Index of the first parameter.
        param2 (int): Index of the second parameter.
        sigma_level (int): Contour level (1 = 68%, 2 = 95%, 3 = 99.7%).
        names (list): List of names for the samples (default is None).

    Returns:
        list: List of FoM values for each sample.
    """
    if names is None:
        names = [f"Sample {i+1}" for i in range(len(samples))]

    sigma_lvls = {3: 0, 2: 1, 1: 2}  # Mapping for contour levels
    foms_array = []

    for sample, name in zip(samples, names):
        # Get 2D density grid data and sorted contour levels
        density = sample.get2DDensityGridData(j=param1, j2=param2, num_plot_contours=3)
        contour_levels = sorted(density.contours)

        # Extract the specified contour level's polygon coordinates
        contour_data = density.getContour(contour_levels[sigma_lvls[sigma_level]])
        xy = contour_data.vertices

        # Calculate the area of the contour and the Figure of Merit (FoM)
        poly = Polygon(xy)
        area = poly.area
        fom = (2.3 * np.pi) / area

        print(f"FoM {name}: {fom}")
        foms_array.append(fom)

    return foms_array


def covariance_matrix(samples, param1, param2, sigma_level=1, names=None):
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

    Returns:
        list: List of FoM values for each sample.
    """
    if names is None:
        names = [f"Sample {i+1}" for i in range(len(samples))]

    sigma_lvls = {3: 0, 2: 1, 1: 2}  # Mapping for contour levels
    foms_array = []

    for sample, name in zip(samples, names):
        # Get the covariance matrix for the specified parameters
        cov_matrix = sample.getCovarianceMatrix([param1, param2]).matrix
        det_cov = np.linalg.det(cov_matrix)

        # Calculate the Figure of Merit (FoM) using det(cov)
        fom = 1 / np.sqrt(det_cov)

        print(f"FoM {name}: {fom}")
        foms_array.append(fom)

    return foms_array
