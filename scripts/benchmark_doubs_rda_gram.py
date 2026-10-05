import time
import numpy as np
import pandas as pd

from rdacca_hp import rdacca_hp, permu_hp


def hellinger_transform(df: pd.DataFrame) -> pd.DataFrame:
    row_sums = df.sum(axis=1)
    rel = df.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)
    return np.sqrt(rel)


spe = pd.read_csv("doubs_fish.csv", index_col=0).apply(pd.to_numeric, errors="raise")
env = pd.read_csv("doubs_env.csv", index_col=0).apply(pd.to_numeric, errors="raise")

# Match the R workflow used in the project.
spe = spe.drop(spe.index[7])
env = env.drop(env.index[7])
env = env.drop(columns=["dfs"], errors="raise")

if not spe.index.equals(env.index):
    raise ValueError("spe and env row indices do not match after preprocessing.")

spe_hel = hellinger_transform(spe)

print("=== observed rdacca_hp ===")
t0 = time.perf_counter()
obs = rdacca_hp(
    dv=spe_hel,
    iv=env,
    method="RDA",
    type="adjR2",
    scale=False,
    var_part=True,
)
t1 = time.perf_counter()

print(f"Observed runtime: {t1 - t0:.6f} s")
print(f"Total explained variation: {obs.total_explained_variation}")
print(obs.hier_part)

if round(float(obs.total_explained_variation), 3) != 0.556:
    print("WARNING: total explained variation differs from the established Doubs 0.556 benchmark.")
else:
    print("Doubs total explained variation check: PASS (0.556)")

print("\n=== permu_hp: 1000 total values / 999 randomized runs ===")
t2 = time.perf_counter()
perm = permu_hp(
    dv=spe_hel,
    iv=env,
    method="RDA",
    type="adjR2",
    permutations=1000,
    scale=False,
    verbose=True,
    random_state=123,
)
t3 = time.perf_counter()

print(f"Permutation runtime: {t3 - t2:.6f} s")
print(perm)
