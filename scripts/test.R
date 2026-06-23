library(ade4)
data(doubs)

write.csv(doubs$fish, "doubs_fish.csv")
write.csv(doubs$env, "doubs_env.csv")

library(vegan)

data(mite)
data(mite.env)

write.csv(mite, "mite.csv", row.names = TRUE)
write.csv(mite.env, "mite_env.csv", row.names = TRUE)

##############################################
library(vegan)
library(utils)

# =========================
# 1. 读 doubs 原始 csv
# =========================
spe <- read.csv("doubs_fish.csv", row.names = 1, check.names = FALSE)
env <- read.csv("doubs_env.csv", row.names = 1, check.names = FALSE)

# 删除第 8 个空样点
spe <- spe[-8, ]
env <- env[-8, ]

# 删除 dfs
env <- env[, -1, drop = FALSE]

# Bray-Curtis 距离
spe.dist <- vegdist(spe, method = "bray")

vars <- colnames(env)

# 先测这些子集
subsets <- list(
  c("alt"),
  c("slo"),
  c("flo"),
  c("pH"),
  c("har"),
  c("pho"),
  c("nit"),
  c("amm"),
  c("oxy"),
  c("bdo"),
  c("alt", "slo"),
  c("alt", "flo"),
  c("slo", "flo"),
  c("alt", "slo", "flo"),
  vars
)

results <- data.frame(
  subset = character(),
  n_vars = integer(),
  r_adjR2 = numeric(),
  r_R2 = numeric(),
  stringsAsFactors = FALSE
)

for (s in subsets) {
  env_sub <- env[, s, drop = FALSE]
  
  cap <- capscale(spe.dist ~ ., data = env_sub)
  adj <- RsquareAdj(cap)
  
  row <- data.frame(
    subset = paste(s, collapse = "+"),
    n_vars = length(s),
    r_adjR2 = adj$adj.r.squared,
    r_R2 = adj$r.squared,
    stringsAsFactors = FALSE
  )
  
  results <- rbind(results, row)
}

print(results)
write.csv(results, "dbrda_totals_r.csv", row.names = FALSE)
cat("R totals saved to dbrda_totals_r.csv\n")

####################################################
library(ade4)
library(vegan)
library(rdacca.hp)

cat("rdacca.hp package version:\n")
print(packageVersion("rdacca.hp"))

cat("\nrdacca.hp function source location:\n")
print(getAnywhere(rdacca.hp))

data(doubs)

spe <- doubs$fish
env <- doubs$env

spe <- spe[-8, ]
env <- env[-8, ]
env <- env[, -1]

spe.dist <- vegdist(spe, method = "bray")

subsets <- list(
  c("alt", "slo"),
  c("alt", "flo"),
  c("slo", "flo"),
  c("alt", "slo", "flo"),
  colnames(env)
)

for (s in subsets) {
  cat("\n=============================\n")
  cat("subset:", paste(s, collapse = "+"), "\n")
  cat("=============================\n")
  
  env_sub <- env[, s, drop = FALSE]
  
  hp <- rdacca.hp(
    spe.dist,
    env_sub,
    method = "dbRDA",
    type = "adjR2"
  )
  
  cap <- RsquareAdj(
    capscale(spe.dist ~ ., data = env_sub)
  )
  
  db <- RsquareAdj(
    dbrda(spe.dist ~ ., data = env_sub)
  )
  
  cat("rdacca.hp Total_explained_variation:\n")
  print(hp$Total_explained_variation)
  
  cat("capscale adjR2:\n")
  print(cap$adj.r.squared)
  
  cat("dbrda adjR2:\n")
  print(db$adj.r.squared)
  
  cat("rdacca.hp Hier.part:\n")
  print(hp$Hier.part)
}

###########################################
library(ade4)
library(vegan)
library(rdacca.hp)

data(doubs)

spe <- doubs$fish
env <- doubs$env

# 删除第 8 个空样点
spe <- spe[-8, ]
env <- env[-8, ]

# 删除 dfs
env <- env[, -1]

# dist 对象（压缩距离形式）
spe.dist <- vegdist(spe, method = "bray")

# dbRDA
spe.hp.dbrda <- rdacca.hp(spe.dist, env, method = "dbRDA", type = "adjR2")
print(spe.hp.dbrda)

# permutation test
perm.dbrda <- permu.hp(spe.dist, env, method = "dbRDA")
print(perm.dbrda)
