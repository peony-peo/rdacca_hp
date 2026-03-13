import matplotlib.pyplot as plt
import pytest

from rdacca_hp.core import rdacca_hp
from rdacca_hp.plotting import plot_comparison, plot_rdaccahp
from rdacca_hp.utils import create_test_data


def _make_result(n_predictors=3, var_part=True, seed=42):
    dv, iv = create_test_data(
        n_samples=50,
        n_predictors=n_predictors,
        n_responses=2,
        seed=seed,
    )
    return rdacca_hp(
        dv=dv,
        iv=iv,
        method="RDA",
        type="R2",
        var_part=var_part,
    )


def test_plot_rdaccahp_bar_returns_figure():
    result = _make_result(n_predictors=3, var_part=True, seed=1)
    fig = plot_rdaccahp(result, plot_type="bar", plot_perc=False)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_rdaccahp_venn_returns_figure_for_three_predictors():
    result = _make_result(n_predictors=3, var_part=True, seed=2)
    fig = plot_rdaccahp(result, plot_type="venn")

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_rdaccahp_venn_rejects_missing_var_part():
    result = _make_result(n_predictors=3, var_part=False, seed=3)

    with pytest.raises(ValueError, match="Variation partitioning results not available"):
        plot_rdaccahp(result, plot_type="venn")


def test_plot_rdaccahp_venn_rejects_more_than_four_predictors():
    result = _make_result(n_predictors=5, var_part=True, seed=4)

    with pytest.raises(ValueError, match="supports only 2-4 variables"):
        plot_rdaccahp(result, plot_type="venn")


def test_plot_rdaccahp_invalid_plot_type_raises():
    result = _make_result(n_predictors=3, var_part=True, seed=5)

    with pytest.raises(ValueError, match="plot_type must be"):
        plot_rdaccahp(result, plot_type="bad")


def test_result_plot_convenience_method_works():
    result = _make_result(n_predictors=3, var_part=True, seed=6)
    fig = result.plot(plot_type="bar")

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_comparison_returns_figure():
    r1 = _make_result(n_predictors=3, var_part=False, seed=7)
    r2 = _make_result(n_predictors=3, var_part=False, seed=8)

    fig = plot_comparison([r1, r2], plot_perc=True)

    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_comparison_rejects_empty_input():
    with pytest.raises(ValueError, match="At least one result is required"):
        plot_comparison([])