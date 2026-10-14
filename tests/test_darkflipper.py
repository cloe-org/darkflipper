"""Tests for darkflipper using Gaussian samples with known answers."""

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from getdist import MCSamples  # noqa: E402
from scipy import stats  # noqa: E402

import darkflipper as dfp  # noqa: E402

COV = np.array([[1.0, 0.5, 0.0], [0.5, 2.0, 0.0], [0.0, 0.0, 1.0]])
NAMES = ["a", "b", "c"]


def _samples(mean, cov=COV, seed=0, n=50000, loglikes=None):
    rng = np.random.default_rng(seed)
    x = rng.multivariate_normal(mean, cov, size=n)
    return MCSamples(
        samples=x,
        names=NAMES,
        loglikes=loglikes,
        settings={"ignore_rows": 0},
    )


@pytest.fixture(scope="module")
def unbiased():
    return _samples([0.0, 0.0, 0.0])


@pytest.fixture(scope="module")
def biased():
    return _samples([0.1, 0.2, 0.3], seed=1)


def _true_fom():
    return 1 / np.sqrt(np.linalg.det(COV[:2, :2]))


def test_covariance_matrix_fom(unbiased):
    (fom,) = dfp.covariance_matrix([unbiased], "a", "b")
    assert fom == pytest.approx(_true_fom(), rel=0.02)


@pytest.mark.parametrize("sigma_level", [1, 2, 3])
def test_polygon_fom_matches_gaussian(unbiased, sigma_level):
    foms = dfp.polygon([unbiased], "a", "b", sigma_level=sigma_level, plot=False)
    assert len(foms) == 1
    assert foms[0] == pytest.approx(_true_fom(), rel=0.05)


def test_polygon_one_fom_per_sample(unbiased, biased):
    foms = dfp.polygon([unbiased, biased], "a", "b", plot=True)
    assert len(foms) == 2


def test_parameter_bias_subset(unbiased, biased):
    fob = dfp.parameter_bias(unbiased, biased, [0, 1], ["a", "b"])
    shift = np.array([0.1, 0.2])
    expected = np.sqrt(shift @ np.linalg.solve(COV[:2, :2], shift))
    assert fob == pytest.approx(expected, abs=0.02)
    by_name = dfp.parameter_bias(unbiased, biased, ["a", "b"], ["a", "b"])
    assert by_name == pytest.approx(fob)


def test_chi2_bias():
    n = 10
    unb = _samples([0, 0, 0], n=n, loglikes=np.full(n, 5.0))
    bia = _samples([0, 0, 0], n=n, loglikes=np.full(n, 7.0))
    assert dfp.chi2_bias(unb, bia) == pytest.approx(2.0)
    with pytest.warns(UserWarning, match="Negative"):
        assert np.isnan(dfp.chi2_bias(bia, unb))


def test_chi2_statistics():
    data = np.array([1.0, 2.0, 3.0])
    theory = np.array([0.0, 2.0, 3.0])
    cov = np.diag([1.0, 2.0, 4.0])
    assert dfp.mahalanobis_distance(data, theory, cov) == pytest.approx(1.0)
    assert dfp.effect_size(data, theory, cov) == pytest.approx(np.sqrt(1 / 3))
    assert dfp.pvalue(data, theory, cov) == pytest.approx(stats.chi2.sf(1.0, 3))
    assert dfp.pvalue(data, theory, cov, n_params=1) == pytest.approx(
        stats.chi2.sf(1.0, 2)
    )


def test_theoretical_prediction():
    cov = np.array([[2.0, 0.0], [0.0, 2.0]])
    assert dfp.theoretical_prediction(cov, 100) == pytest.approx(0.04)
