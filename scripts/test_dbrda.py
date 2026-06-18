import time
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist

from rdacca_hp import rdacca_hp, permu_hp


# ------------------------------------------------------------
# Python equivalent of:
#   spe <- doubs$fish
#   env <- doubs$env
#   spe <- spe[-8, ]
#   env <- env[-8, ]
#   env <- env[, -1]
#   spe.dist <- vegdist(spe, method = "bray")
#   rdacca.hp(spe.dist, env, method = "dbRDA", type = "adjR2")
#   permu.hp(spe.dist, env, method = "dbRDA")
# ------------------------------------------------------------

spe = pd.read_csv("doubs_fish.csv", index_col=0)
env = pd.read_csv("doubs_env.csv", index_col=0)

spe = spe.apply(pd.to_numeric, errors="raise")
env = env.apply(pd.to_numeric, errors="raise")

# R is 1-based: spe[-8, ]; Python position 7 is the 8th row.
spe = spe.drop(spe.index[7])
env = env.drop(env.index[7])

# R: env <- env[, -1]  # remove dfs
env = env.drop(columns=["dfs"], errors="raise")

if not spe.index.equals(env.index):
    raise ValueError("spe and env row indices do not match after preprocessing.")

# R: vegdist(spe, method = "bray") returns a condensed distance vector.
spe_dist = pdist(spe.to_numpy(dtype=float), metric="braycurtis")

print("=== Bray-Curtis condensed distance vector ===")
print("n sites:", spe.shape[0])
print("vector length:", len(spe_dist))
print("expected length:", spe.shape[0] * (spe.shape[0] - 1) // 2)

start = time.perf_counter()
spe_hp_dbrda = rdacca_hp(
    dv=spe_dist,
    iv=env,
    method="dbRDA",
    type="adjR2",
    scale=False,
    var_part=True,
)
mid = time.perf_counter()

print("\n=== dbRDA rdacca_hp result ===")
print("Total explained variation:")
print(spe_hp_dbrda.total_explained_variation)
print("\nHierarchical partitioning:")
print(spe_hp_dbrda.hier_part)
print("\nVariation partitioning:")
print(spe_hp_dbrda.var_part)
print(f"\nrdacca_hp dbRDA time: {mid - start:.3f} seconds")

perm_start = time.perf_counter()
perm_dbrda = permu_hp(
    dv=spe_dist,
    iv=env,
    method="dbRDA",
    type="adjR2",
    permutations=1000,  # R-style: 999 randomized runs + 1 observed value
    scale=False,
    verbose=True,
    random_state=None,
)
perm_end = time.perf_counter()

print("\n=== dbRDA permutation test result ===")
print(perm_dbrda)
print(f"\npermu_hp dbRDA time: {perm_end - perm_start:.3f} seconds")
