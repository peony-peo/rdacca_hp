import cProfile
import pstats
import io
import pandas as pd
import numpy as np
import rdacca_hp
print(rdacca_hp.__file__)

from rdacca_hp import rdacca_hp, permu_hp


def hellinger_transform(df: pd.DataFrame) -> pd.DataFrame:
    row_sums = df.sum(axis=1)
    rel = df.div(row_sums.replace(0, np.nan), axis=0).fillna(0.0)
    return np.sqrt(rel)


spe = pd.read_csv("doubs_fish.csv", index_col=0)
env = pd.read_csv("doubs_env.csv", index_col=0)

spe = spe.drop(spe.index[7])
env = env.drop(env.index[7])

env = env.drop(columns=["dfs"], errors="raise")
spe_hel = hellinger_transform(spe)

profiler = cProfile.Profile()
profiler.enable()

perm_result = permu_hp(
    dv=spe_hel,
    iv=env,
    method="RDA",
    type="adjR2",
    permutations=199,   # 先别用 999，先缩小方便看热点
    scale=False,
    verbose=False,
)

profiler.disable()

s = io.StringIO()
stats = pstats.Stats(profiler, stream=s).sort_stats("cumtime")
stats.print_stats(40)   # 看前 40 个最耗时函数
print(s.getvalue())