import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from rdacca_hp import rdacca_hp, permu_hp

# Same Doubs setup and fixed RNG seed as the Round-1 benchmark so that
# Round-1 and Round-2 permutation tables can be compared exactly.
spe = pd.read_csv("doubs_fish.csv", index_col=0)
env = pd.read_csv("doubs_env.csv", index_col=0)

spe = spe.apply(pd.to_numeric, errors="raise")
env = env.apply(pd.to_numeric, errors="raise")

spe = spe.drop(spe.index[7])
env = env.drop(env.index[7])
env = env.drop(columns=["dfs"], errors="raise")

if not spe.index.equals(env.index):
    raise ValueError("spe and env row indices do not match after preprocessing.")

spe_dist = pdist(spe.to_numpy(dtype=float), metric="braycurtis")

print("=== observed rdacca_hp dbRDA ===")
start = time.perf_counter()
observed = rdacca_hp(
    dv=spe_dist,
    iv=env,
    method="dbRDA",
    type="adjR2",
    var_part=True,
)
elapsed = time.perf_counter() - start

print(f"Observed runtime: {elapsed:.6f} s")
print(f"Total explained variation: {observed.total_explained_variation}")
print(observed.hier_part)

if observed.total_explained_variation != 0.645:
    raise AssertionError(
        f"Unexpected Doubs total explained variation: {observed.total_explained_variation}"
    )
print("Doubs observed check: PASS (0.645)")

print("\n=== dbRDA permutation: 1000 total values / 999 randomized runs ===")
start = time.perf_counter()
perm = permu_hp(
    dv=spe_dist,
    iv=env,
    method="dbRDA",
    type="adjR2",
    permutations=1000,
    verbose=True,
    random_state=123,
)
elapsed = time.perf_counter() - start

print(f"Permutation runtime: {elapsed:.6f} s")
print(perm)

expected_individual = np.array([
    0.0889, 0.0766, 0.1006, -0.0064, 0.0325,
    0.0425, 0.0714, 0.0363, 0.1433, 0.0590,
])
expected_p = [
    "0.025   *",
    "0.029   *",
    "0.013   *",
    "0.466    ",
    "0.134    ",
    "0.058    ",
    "0.03   *",
    "0.114    ",
    "0.002  **",
    "0.061    ",
]

np.testing.assert_array_equal(
    perm["Individual"].to_numpy(dtype=float), expected_individual
)
if perm["Pr(>I)"].tolist() != expected_p:
    raise AssertionError(
        "Fixed-seed permutation table differs from the verified Round-1 result.\n"
        f"Observed: {perm['Pr(>I)'].tolist()}"
    )
print("Fixed-seed Round-1 permutation result check: PASS")
