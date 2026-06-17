# rdacca_hp

Python implementation of hierarchical partitioning and variation partitioning for canonical analyses, inspired by the R package **rdacca.hp**.

`rdacca_hp` provides hierarchical partitioning and variation partitioning for:

* **RDA** (Redundancy Analysis)
* **CCA** (Canonical Correspondence Analysis)
* **dbRDA** (distance-based Redundancy Analysis)

It is designed for users who want a Python workflow similar to **rdacca.hp**, while supporting mixed predictor types such as:

* numeric variables
* unordered categorical variables
* ordered categorical variables
* grouped predictor sets

The package also provides:

* permutation-based significance testing
* plotting utilities for hierarchical partitioning and variation partitioning results

---

## Features

* Hierarchical partitioning (`hier_part`)
* Variation partitioning (`var_part`)
* Support for:

  * numeric predictors
  * unordered factors
  * ordered factors
  * grouped predictors
* Permutation testing with `permu_hp()`
* Plotting utilities for single results and result comparison
* Baseline validation against R outputs for key RDA use cases
* dbRDA support for both:

  * square symmetric distance matrices
  * condensed / dist-style distance input

---

## Current status

This package is currently in an early public release stage.

At the current stage:

* the **RDA** workflow has been checked carefully against the R package **rdacca.hp**
* mixed predictor inputs (numeric + unordered factor + ordered factor) are supported
* permutation testing is available
* dbRDA supports both full distance matrices and condensed distance input
* baseline tests against R outputs are included for selected cases

Notes:

* results for **RDA** are expected to closely match the R implementation in validated scenarios
* **CCA** and **dbRDA** are implemented and tested, and further benchmark expansion is planned in future releases
* permutation p-values may show small Monte Carlo differences relative to R because random permutation sequences differ across platforms

---

## Installation

### Install from local source

```bash
pip install .
```

### Install in editable mode for development

```bash
pip install -e .[dev]
```

### Install optional plotting dependencies

```bash
pip install -e .[plot]
```

### Install from a published package

```bash
pip install rdacca_hp
```

---

## Public API

The main public functions can be imported directly from the package top level:

```python
from rdacca_hp import rdacca_hp, permu_hp, plot_rdaccahp, plot_comparison
```

Main public objects include:

* `rdacca_hp`
* `RdaccaHpResult`
* `calculate_rda`
* `calculate_cca`
* `calculate_dbrda`
* `create_test_data`
* `create_cca_test_data`
* `create_distance_test_data`
* `permu_hp`
* `plot_rdaccahp`
* `plot_comparison`

---

## Quick start

### 1. Numeric predictors only

```python
from rdacca_hp import create_test_data, rdacca_hp

dv, iv = create_test_data()

result = rdacca_hp(
    dv=dv,
    iv=iv,
    method="RDA",
    type="adjR2",
    scale=False,
    var_part=True
)

print(result.total_explained_variation)
print(result.hier_part)
print(result.var_part)
```

### 2. Mixed predictors: numeric + unordered factor + ordered factor

If your predictors contain mixed types, you can explicitly specify factor handling.

```python
import pandas as pd
from rdacca_hp import rdacca_hp

dv = pd.DataFrame({
    "sp1": [2, 3, 5, 4, 6, 7],
    "sp2": [1, 2, 2, 3, 4, 5],
})

iv = pd.DataFrame({
    "WatrCont": [10.1, 9.8, 8.7, 7.5, 6.9, 6.1],
    "Substrate": ["A", "B", "A", "C", "B", "A"],
    "Shrub": ["None", "Few", "Few", "Many", "Many", "Few"],
})

result = rdacca_hp(
    dv=dv,
    iv=iv,
    method="RDA",
    type="adjR2",
    scale=False,
    var_part=True,
    categorical_factors=["Substrate"],
    ordered_factors={"Shrub": ["None", "Few", "Many"]}
)

print(result.hier_part)
print(result.var_part)
```

### 3. Grouped predictors

```python
import pandas as pd
from rdacca_hp import create_test_data, rdacca_hp

dv, iv = create_test_data(n_predictors=4)

groups = {
    "Climate": pd.DataFrame(iv[:, :2], columns=["Temp", "Rain"]),
    "Soil": pd.DataFrame(iv[:, 2:], columns=["N", "C"]),
}

result = rdacca_hp(
    dv=dv,
    iv=groups,
    method="RDA",
    type="R2",
    var_part=True
)

print(result.hier_part)
print(result.var_part)
```

### 4. Permutation test

```python
from rdacca_hp import create_test_data, permu_hp

dv, iv = create_test_data()

perm_result = permu_hp(
    dv=dv,
    iv=iv,
    method="RDA",
    type="adjR2",
    permutations=99,
    scale=False,
    random_state=123,
    verbose=False
)

print(perm_result)
```

### 5. dbRDA with a square distance matrix

```python
from rdacca_hp import create_distance_test_data, create_test_data, rdacca_hp

dv_dist = create_distance_test_data(n_samples=30, n_species=10, seed=123)
_, iv = create_test_data(n_samples=30, n_predictors=3, n_responses=1, seed=123)

result = rdacca_hp(
    dv=dv_dist,
    iv=iv,
    method="dbRDA",
    type="adjR2",
    var_part=True,
    add=True,
    n_axes=5,
)

print(result.hier_part)
```

### 6. dbRDA with condensed / dist-style input

```python
from scipy.spatial.distance import squareform
from rdacca_hp import create_distance_test_data, create_test_data, rdacca_hp

dv_dist = create_distance_test_data(n_samples=30, n_species=10, seed=123)
dv_condensed = squareform(dv_dist)
_, iv = create_test_data(n_samples=30, n_predictors=3, n_responses=1, seed=123)

result = rdacca_hp(
    dv=dv_condensed,
    iv=iv,
    method="dbRDA",
    type="adjR2",
    var_part=True,
)

print(result.hier_part)
```

### 7. Plotting

```python
from rdacca_hp import create_test_data, rdacca_hp, plot_rdaccahp

dv, iv = create_test_data()
result = rdacca_hp(dv=dv, iv=iv, method="RDA", type="R2", var_part=True)

fig = plot_rdaccahp(result, plot_type="bar")
```

You can also use the convenience method on the result object:

```python
fig = result.plot(plot_type="bar")
```

---

## Main functions

### `rdacca_hp()`

Main function for hierarchical partitioning and variation partitioning.

### `permu_hp()`

Permutation test for hierarchical partitioning results.

### `plot_rdaccahp()`

Plot a single hierarchical partitioning result.

### `plot_comparison()`

Compare multiple hierarchical partitioning results in one figure.

---

## Input conventions

### Response matrix (`dv`)

`dv` can be:

* a NumPy array
* a pandas DataFrame

For **RDA**, users often apply Hellinger transformation before analysis when working with community data.

For **dbRDA**, `dv` can be either:

* a square symmetric distance matrix
* a valid condensed / dist-style distance vector

### Predictor matrix (`iv`)

`iv` can be:

* a NumPy array
* a pandas DataFrame
* a grouped structure such as `dict`
* a grouped structure such as `list`

Supported predictor types include:

* continuous numeric columns
* unordered categorical columns
* ordered categorical columns

---

## Predictor handling

`rdacca_hp` supports several predictor formats.

### 1. Numeric matrix or array

If `iv` is given as a numeric array or numeric matrix, all predictors are treated as numeric variables.

```python
result = rdacca_hp(dv=dv, iv=iv_numeric)
```

### 2. pandas DataFrame with mixed predictor types

If `iv` is given as a pandas DataFrame, the package can handle mixed predictor types, including:

* continuous numeric variables
* unordered categorical factors
* ordered factors

Numeric columns are handled directly as numeric predictors.

For non-numeric predictors, users can explicitly specify variable types when needed:

* use `categorical_factors=[...]` for unordered categorical variables
* use `ordered_factors={...}` for ordered variables with a declared level order

```python
result = rdacca_hp(
    dv=dv,
    iv=iv_df,
    categorical_factors=["Substrate"],
    ordered_factors={"Shrub": ["None", "Few", "Many"]},
)
```

For mixed-type DataFrames, explicit specification is recommended, especially when:

* the dataset contains string-based predictors
* factor level order matters
* reproducible encoding behavior is important

In practice:

* numeric variables are supported directly
* unordered factors should be declared with `categorical_factors`
* ordered factors should be declared with `ordered_factors`

This makes the package easy to use for standard numeric analyses, while still allowing precise control over how mixed predictor data are encoded.

### 3. Grouped predictors as a dictionary

```python
result = rdacca_hp(dv=dv, iv={"Climate": climate_df, "Soil": soil_df})
```

### 4. Grouped predictors as a list

```python
result = rdacca_hp(dv=dv, iv=[group1_df, group2_df, group3_df])
```

---

## Returned object

`rdacca_hp()` returns a `RdaccaHpResult` object containing at least:

* `method_type`
* `total_explained_variation`
* `hier_part`

and optionally:

* `var_part`

It also provides:

* `summary()`
* `plot()`

Example:

```python
result = rdacca_hp(dv=dv, iv=iv)
result.summary()
fig = result.plot(plot_type="bar")
```

### `hier_part`

A table containing:

* `Unique`
* `Average.share`
* `Individual`
* `I.perc(%)`

### `var_part`

A table containing:

* `Fractions`
* `% Total`

---

## Running tests

Run all tests:

```bash
pytest -q
```

Run coverage:

```bash
pytest --cov=rdacca_hp --cov-report=term-missing
```

Run only R baseline tests:

```bash
pytest tests/test_r_baselines.py -q
```

Run dbRDA-specific tests:

```bash
pytest tests/test_dbrda.py -q
```

---

## R baseline validation

This project includes a benchmark workflow against R outputs.

### Benchmark directories

* `benchmark/data/`: fixed input data
* `benchmark/expected/`: expected outputs exported from R
* `benchmark/r_scripts/`: scripts used to generate expected R outputs

### Current validated RDA baselines

* `rda_numeric_2vars`
* `rda_unordered_factor`
* `rda_mite_full_mixed`
* `rda_ordered_factor_mixed`

These baselines are used to check that Python results remain aligned with the corresponding R workflow for validated RDA scenarios.

---

## Important notes

### 1. Small p-value differences are normal

Permutation p-values may differ slightly from R because:

* permutation sequences differ
* random seeds differ across platforms
* permutation p-values are Monte Carlo estimates

### 2. Ordered factors matter

Ordered factors should not be treated the same way as ordinary categorical variables.
If you have ordered predictor levels, specify them explicitly.

### 3. CSV reading and `"None"`

If a valid category level is literally `"None"`, make sure it is not accidentally parsed as missing data when reading CSV files.

For example:

```python
import pandas as pd
pd.read_csv("file.csv", keep_default_na=False)
```

### 4. dbRDA distance input

For dbRDA, the response can now be supplied either as:

* a full square symmetric distance matrix, or
* a condensed / dist-style distance vector

This is intended to make the Python workflow closer to the flexibility of R-style distance input.

---

## Limitations

* RDA is currently the most thoroughly validated workflow
* CCA and dbRDA are available, but more benchmark expansion is still desirable
* very large permutation jobs may be slow in pure Python workflows

---

## Recommended usage for reproducibility

For the most reproducible results:

1. keep benchmark datasets fixed
2. explicitly specify unordered and ordered factors when needed
3. use baseline tests against R outputs
4. report the package version and analysis settings

---

## Package structure

```text
rdacca_hp/
├── rdacca_hp/
│   ├── __init__.py
│   ├── core.py
│   ├── utils.py
│   ├── permutation.py
│   └── plotting.py
│
├── tests/
│   ├── test_r_baselines.py
│   ├── test_assertions.py
│   ├── test_core.py
│   ├── test_cca.py
│   ├── test_dbrda.py
│   ├── test_permutation.py
│   ├── test_plotting.py
│   └── test_public_api.py
│
├── benchmark/
│   ├── data/
│   ├── expected/
│   └── r_scripts/
│
├── scripts/
│   └── test_time.py
│
├── README.md
├── pyproject.toml
└── LICENSE
```

---

## Citation / inspiration

This Python project is inspired by the R package **rdacca.hp** and its hierarchical partitioning framework for canonical analyses.

If you use this package in academic work, you should also cite the original methodological and/or R package sources as appropriate.

---

## License

This project is licensed under the **MIT License**.

---

## Contact

Author: **Jiangshan Lai**
Email: **[lai@njfu.edu.cn](mailto:lai@njfu.edu.cn)**

Repository: `https://github.com/peony-peo/rdacca_hp`
