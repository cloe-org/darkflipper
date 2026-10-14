<p align="center">
  <img src="docs/darkflipper-logo-white.png" alt="darkflipper logo" width="400">
</p>

**Flip the dark side of statistics in cosmology.**
Figures of Merit, Bias, and statistical diagnostics such as p-values and much more.

---

## 🐬 What is _darkflipper_?

`darkflipper` is a lightweight Python package delivering **essential statistical tools for cosmology** at ease:

It’s designed for scientists who:

- are investigating systematics in cosmological data
- care about **Figures of Merit** _and_ **Figures of Bias** in cosmological inference
- want results that are robust, transparent, and publication-ready without worrying about definitions

---

## 📦 Installation

One command.

```sh
pip install darkflipper
```

---

## 🧩 Requirements

`darkflipper` plays nicely with the standard packages:

- Python ≥ 3.10
- `numpy`
- `scipy`
- `matplotlib` (≥ 3.6) and `contourpy`
- `getdist`
- `shapely`

---

## 🚀 Usage

Most functions take [GetDist](https://getdist.readthedocs.io) `MCSamples`, so they work with chains from Cobaya, CosmoSIS, MontePython, Nautilus, …

```python
import darkflipper as dfp
from getdist import loadMCSamples

baseline = loadMCSamples("chains/baseline")
systematic = loadMCSamples("chains/with_systematic")

# Figure of Merit in the (w0, wa) plane
dfp.covariance_matrix([baseline, systematic], "w0", "wa")  # 1/sqrt(det C)
dfp.polygon([baseline, systematic], "w0", "wa", plot=False)  # contour area

# Figure of Bias: shift of (w0, wa) in units of the baseline errors
dfp.parameter_bias(baseline, systematic, ["w0", "wa"], ["w0", "wa"])

# Goodness of fit of a theory vector to data
dfp.pvalue(data, theory, cov, n_params=2)
```

| Function                 | What it computes                                                                                                                   |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| `covariance_matrix`      | FoM = 1/√det(C) of two parameters                                                                                                  |
| `polygon`                | FoM from the area of the 2D marginalised contour (68/95/99%), normalised to agree with `covariance_matrix` for Gaussian posteriors |
| `parameter_bias`         | FoB = √(Δθᵀ C⁻¹ Δθ) between the means of two chains                                                                                |
| `chi2_bias`              | FoB = √Δχ² between the mean χ² of two chains                                                                                       |
| `pvalue`                 | χ² probability to exceed, with dof = N<sub>data</sub> − `n_params`                                                                 |
| `mahalanobis_distance`   | √χ² between data and theory                                                                                                        |
| `effect_size`            | √(χ²/N<sub>data</sub>)                                                                                                             |
| `theoretical_prediction` | tr(C)/N<sub>samples</sub>                                                                                                          |

> **Note:** `polygon` relies on GetDist's kernel density estimate. For very degenerate contours (correlation ≳ 0.99), smoothing broadens the contour and the FoM is underestimated, so cross-check with `covariance_matrix`.

A worked example notebook lives in the [cloe-org playground](https://github.com/cloe-org/playground/blob/main/tutorials/statistics/darkflipper.ipynb).

---

## 🤝 Contributing

Want to help shape the dark side? We’d love to have you.

1. **Open an issue** to discuss your idea — new feature, new statistics, or all-dreamy thought
2. **Create a feature branch**
   ```sh
   git checkout -b feature/your-feature-name
   ```
3. **Run the tests and checks**
   ```sh
   pip install -e ".[dev]"
   pytest
   pre-commit run --all-files
   ```
4. **Commit your work**
   ```sh
   git commit -m "Add awesome feature"
   ```
5. **Push it**
   ```sh
   git push origin feature/your-feature-name
   ```
6. **Open a pull request** and let’s flip it together ✨

## 🤝 Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://jclaramunt.github.io"><img src="https://avatars.githubusercontent.com/u/32998547?v=4?s=100" width="100px;" alt="Juan Claramunt"/><br /><sub><b>Juan Claramunt</b></sub></a><br /><a href="#ideas-jclaramunt" title="Ideas, Planning, & Feedback">🤔</a> <a href="#review-jclaramunt" title="Reviewed Pull Requests">👀</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://gcanasherrera.com"><img src="https://avatars.githubusercontent.com/u/13239454?v=4?s=100" width="100px;" alt="Guadalupe Cañas-Herrera"/><br /><sub><b>Guadalupe Cañas-Herrera</b></sub></a><br /><a href="#code-gcanasherrera" title="Code">💻</a> <a href="#bug-gcanasherrera" title="Bug reports">🐛</a> <a href="#doc-gcanasherrera" title="Documentation">📖</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://www.santicasas.xyz"><img src="https://avatars.githubusercontent.com/u/6987716?v=4?s=100" width="100px;" alt="Santiago Casas"/><br /><sub><b>Santiago Casas</b></sub></a><br /><a href="#ideas-santiagocasas" title="Ideas, Planning, & Feedback">🤔</a> <a href="#code-santiagocasas" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
