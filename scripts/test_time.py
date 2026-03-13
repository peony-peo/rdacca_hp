import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdacca_hp.core import rdacca_hp
from rdacca_hp.permutation import permu_hp
from rdacca_hp.utils import remove_index_like_columns


def hellinger_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Equivalent to vegan::decostand(x, 'hellinger')."""
    row_sums = df.sum(axis=1).replace(0, np.nan)
    rel = df.div(row_sums, axis=0).fillna(0.0)
    return np.sqrt(rel)


def load_csv_safely(path: Path) -> pd.DataFrame:
    """Load CSV while preserving string levels such as 'None'."""
    df = pd.read_csv(path, keep_default_na=False)
    return remove_index_like_columns(df, warn=True)


def print_section(title: str):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main():
    mite_path = ROOT / "benchmark" / "data" / "mite.csv"
    mite_env_path = ROOT / "benchmark" / "data" / "mite_env.csv"

    if not mite_path.exists() or not mite_env_path.exists():
        raise FileNotFoundError(
            f"Missing required files: {mite_path.name} and/or {mite_env_path.name}. "
            "Please export mite and mite.env from R into the project root."
        )

    mite = load_csv_safely(mite_path)
    mite_env = load_csv_safely(mite_env_path)

    print_section("Input overview")
    print(f"mite shape: {mite.shape}")
    print(f"mite.env shape: {mite_env.shape}")
    print("mite.env columns:", list(mite_env.columns))
    print(mite_env.head())

    mite_hel = hellinger_transform(mite)

    ordered_factors = {"Shrub": ["None", "Few", "Many"]}
    categorical_factors = ["Substrate", "Topo"]

    print_section("Observed rdacca_hp()")
    t0 = time.perf_counter()
    obs = rdacca_hp(
        dv=mite_hel,
        iv=mite_env,
        method="RDA",
        type="adjR2",
        scale=False,
        var_part=True,
        ordered_factors=ordered_factors,
        categorical_factors=categorical_factors,
    )
    t1 = time.perf_counter()

    print(f"Observed runtime: {t1 - t0:.4f} seconds")
    print(f"Total explained variation: {obs.total_explained_variation:.6f}")
    print("\nHier.part:")
    print(obs.hier_part.to_string(float_format="%.4f"))

    if obs.var_part is not None:
        print("\nVar.part:")
        print(obs.var_part.to_string(float_format="%.4f"))

    print_section("Permutation permu_hp()")
    np.random.seed(123)
    t2 = time.perf_counter()
    perm = permu_hp(
        dv=mite_hel,
        iv=mite_env,
        method="RDA",
        type="adjR2",
        permutations=1000,
        scale=False,
        verbose=True,
        ordered_factors=ordered_factors,
        categorical_factors=categorical_factors,
    )
    t3 = time.perf_counter()

    print(f"Permutation runtime: {t3 - t2:.4f} seconds")
    print("\nPermutation result:")
    print(perm.to_string(float_format="%.4f"))


if __name__ == "__main__":
    main()
