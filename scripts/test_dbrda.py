import pandas as pd
from sklearn.metrics import pairwise_distances

import rdacca_hp
from rdacca_hp import rdacca_hp as run_rdacca_hp
from rdacca_hp.core import calculate_dbrda


def main():
    print("=== rdacca_hp package path ===")
    print(rdacca_hp.__file__)

    # =========================
    # 对应 R:
    # data(doubs)
    # spe <- doubs$fish
    # env <- doubs$env
    # =========================
    spe = pd.read_csv("doubs_fish.csv", index_col=0)
    env = pd.read_csv("doubs_env.csv", index_col=0)

    print("\n=== original shapes ===")
    print("spe:", spe.shape)
    print("env:", env.shape)

    # =========================
    # 对应 R:
    # spe <- spe[-8, ]
    # env <- env[-8, ]
    # =========================
    spe = spe.drop(spe.index[7])
    env = env.drop(env.index[7])

    # =========================
    # 对应 R:
    # env <- env[, -1]
    # 删除 dfs
    # =========================
    env = env.iloc[:, 1:].copy()

    print("\n=== processed shapes ===")
    print("spe:", spe.shape)
    print("env:", env.shape)
    print("\n=== env columns ===")
    print(env.columns.tolist())

    # =========================
    # 对应 R:
    # spe.dist <- vegdist(spe, method = "bray")
    # =========================
    spe_dist = pairwise_distances(spe, metric="braycurtis")

    spe_dist_df = pd.DataFrame(
        spe_dist,
        index=spe.index,
        columns=spe.index,
    )

    # =========================
    # 对应 R:
    # spe.hp.dbrda <- rdacca.hp(
    #   spe.dist,
    #   env,
    #   method = "dbRDA",
    #   type = "adjR2"
    # )
    # print(spe.hp.dbrda)
    # =========================
    result = run_rdacca_hp(
        dv=spe_dist_df,
        iv=env,
        method="dbRDA",
        type="adjR2",
        scale=False,
        var_part=True,
    )

    print("\n==============================")
    print("Python rdacca_hp dbRDA result")
    print("==============================")

    print("\n=== Method Type ===")
    print(result.method_type)

    print("\n=== Total Explained Variation ===")
    print(result.total_explained_variation)

    print("\n=== Hierarchical Partitioning ===")
    print(result.hier_part)

    print("\n=== Variation Partitioning ===")
    print(result.var_part)

    # =========================
    # 额外诊断：
    # 直接计算全模型 dbRDA adjR2
    # 对应 vegan::capscale + RsquareAdj 的全模型 adjR2
    # =========================
    direct_full_adjR2 = calculate_dbrda(
        dv_dist=spe_dist,
        iv=env.to_numpy(dtype=float),
        type="adjR2",
        add=False,
        sqrt_dist=False,
        n_axes=None,
    )

    direct_full_R2 = calculate_dbrda(
        dv_dist=spe_dist,
        iv=env.to_numpy(dtype=float),
        type="R2",
        add=False,
        sqrt_dist=False,
        n_axes=None,
    )

    print("\n==============================")
    print("Direct full-model dbRDA check")
    print("==============================")
    print("Direct full-model adjR2:", direct_full_adjR2)
    print("Direct full-model R2:    ", direct_full_R2)


if __name__ == "__main__":
    main()