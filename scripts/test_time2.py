import time
import numpy as np
import pandas as pd

from rdacca_hp import rdacca_hp, permu_hp


def hellinger_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hellinger transformation:
    sqrt(x_ij / row_sum_i)
    """
    df = df.apply(pd.to_numeric, errors="raise")
    row_sums = df.sum(axis=1)
    rel = df.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)
    return np.sqrt(rel)


# -------------------------------------------------------------------
# Read data exported from R
# Example files:
# - doubs_fish.csv
# - doubs_env.csv
# -------------------------------------------------------------------
total_start = time.perf_counter()

spe = pd.read_csv("doubs_fish.csv", index_col=0)
env = pd.read_csv("doubs_env.csv", index_col=0)

# Ensure numeric
spe = spe.apply(pd.to_numeric, errors="raise")
env = env.apply(pd.to_numeric, errors="raise")

# -------------------------------------------------------------------
# Fish species as response variables
# Environmental factors as explanatory variables
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# Remove empty site 8 without species
# R: spe <- spe[-8,]
#    env <- env[-8,]
# Python is 0-based, so the 8th row is position 7
# -------------------------------------------------------------------
spe = spe.drop(spe.index[7])
env = env.drop(env.index[7])

# -------------------------------------------------------------------
# Apart from being a spatial position variable rather than
# environmental variables, remove the 'dfs' variable
# from the 'env' data frame
# -------------------------------------------------------------------
env = env.drop(columns=["dfs"], errors="raise")

# Check row alignment
if not spe.index.equals(env.index):
    raise ValueError("Row indices of spe and env do not match after preprocessing.")

# -------------------------------------------------------------------
# Hellinger-transformation of the species dataset for RDA
# to deal with the 'double zero' problem
# -------------------------------------------------------------------
t1 = time.perf_counter()
spe_hel = hellinger_transform(spe)
t2 = time.perf_counter()

# -------------------------------------------------------------------
# rdacca.hp equivalent
# R: (spe.hp <- rdacca.hp(spe.hel, env, method="RDA", type="adjR2"))
# -------------------------------------------------------------------
t3 = time.perf_counter()
spe_hp = rdacca_hp(
    dv=spe_hel,
    iv=env,
    method="RDA",
    type="adjR2",
    scale=False,
    var_part=True,
)
t4 = time.perf_counter()

print("=== rdacca_hp result ===")
print("Total explained variation:")
print(spe_hp.total_explained_variation)

print("\nHierarchical partitioning:")
print(spe_hp.hier_part)

print("\nVariation partitioning:")
print(spe_hp.var_part)

# -------------------------------------------------------------------
# permu.hp equivalent
# R: permu.hp(spe.hel, env)
# -------------------------------------------------------------------
t5 = time.perf_counter()
perm_result = permu_hp(
    dv=spe_hel,
    iv=env,
    method="RDA",
    type="adjR2",
    permutations=1000,
    scale=False,
    verbose=False,
)
t6 = time.perf_counter()

print("\n=== permutation test result ===")
print(perm_result)

total_end = time.perf_counter()

print("\n=== runtime ===")
print(f"Hellinger transform time: {t2 - t1:.6f} seconds")
print(f"rdacca_hp time:           {t4 - t3:.6f} seconds")
print(f"permu_hp time:            {t6 - t5:.6f} seconds")
print(f"Total runtime:            {total_end - total_start:.6f} seconds")