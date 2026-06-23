from pathlib import Path
import importlib

import pandas as pd
import pytest

from rdacca_hp import rdacca_hp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "benchmark" / "dbrda_reference"
OPTIONS_DIR = PROJECT_ROOT / "benchmark" / "dbrda_options_reference"

SETTINGS = {
    "dbrda_default": {
        "dbrdatype": "dbrda",
        "add": False,
        "sqrt_dist": False,
    },
    "dbrda_sqrt": {
        "dbrdatype": "dbrda",
        "add": False,
        "sqrt_dist": True,
    },
    "dbrda_lingoes": {
        "dbrdatype": "dbrda",
        "add": "lingoes",
        "sqrt_dist": False,
    },
    "dbrda_cailliez": {
        "dbrdatype": "dbrda",
        "add": "cailliez",
        "sqrt_dist": False,
    },
    "capscale_default": {
        "dbrdatype": "capscale",
        "add": False,
        "sqrt_dist": False,
    },
}


def _read_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, index_col=0)
    table.index = table.index.astype(str).str.strip()
    table.columns = table.columns.astype(str).str.strip()
    return table


@pytest.fixture(scope="module")
def bray_and_xy():
    bray = pd.read_csv(DATA_DIR / "distance_bray.csv", index_col=0).to_numpy(float)
    xy = pd.read_csv(DATA_DIR / "mite_xy.csv", index_col=0)
    return bray, xy


@pytest.mark.parametrize("setting_name", tuple(SETTINGS))
@pytest.mark.parametrize("r2_type", ("R2", "adjR2"))
def test_dbrda_options_match_r(setting_name, r2_type, bray_and_xy):
    bray, xy = bray_and_xy
    prefix = f"{setting_name}_{r2_type.lower()}"

    result = rdacca_hp(
        dv=bray,
        iv=xy,
        method="dbRDA",
        type=r2_type,
        var_part=True,
        **SETTINGS[setting_name],
    )

    expected_hier = _read_table(OPTIONS_DIR / f"{prefix}_hier_part.csv")
    expected_var = _read_table(OPTIONS_DIR / f"{prefix}_var_part.csv")
    expected_total = float(
        pd.read_csv(OPTIONS_DIR / f"{prefix}_total.csv")["total"].iloc[0]
    )

    actual_hier = result.hier_part.copy()
    actual_hier.index = actual_hier.index.astype(str).str.strip()
    actual_hier.columns = actual_hier.columns.astype(str).str.strip()

    actual_var = result.var_part.copy()
    actual_var.index = actual_var.index.astype(str).str.strip()
    actual_var.columns = actual_var.columns.astype(str).str.strip()

    pd.testing.assert_frame_equal(
        actual_hier,
        expected_hier,
        check_exact=False,
        rtol=0,
        atol=1e-4,
    )
    pd.testing.assert_frame_equal(
        actual_var,
        expected_var,
        check_exact=False,
        rtol=0,
        atol=1e-4,
    )
    assert result.total_explained_variation == pytest.approx(
        expected_total,
        abs=5e-4,
    )


def test_capscale_failure_is_not_silently_replaced(monkeypatch, bray_and_xy):
    core_module = importlib.import_module("rdacca_hp.core")
    bray, xy = bray_and_xy

    def fail_pcoa(*args, **kwargs):
        raise ValueError("deliberate PCoA failure")

    monkeypatch.setattr(core_module, "calculate_pcoa", fail_pcoa)

    with pytest.raises(ValueError, match="deliberate PCoA failure"):
        rdacca_hp(
            dv=bray,
            iv=xy,
            method="dbRDA",
            type="R2",
            dbrdatype="capscale",
        )
