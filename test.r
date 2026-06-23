library(vegan)
library(rdacca.hp)

data(mite)
data(mite.env)
data(mite.xy)
data(mite.pcnm)

out <- "benchmark/dbrda_reference"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

write.csv(mite, file.path(out, "mite.csv"))
write.csv(mite.env, file.path(out, "mite_env.csv"))
write.csv(mite.xy, file.path(out, "mite_xy.csv"))
write.csv(mite.pcnm, file.path(out, "mite_pcnm.csv"))

# 保存 vegan 距离矩阵
methods <- c(
  "bray", "euclidean", "manhattan", "canberra",
  "jaccard", "kulczynski", "gower", "hellinger", "chord"
)

for (m in methods) {
  d <- vegdist(mite, method = m)
  write.csv(as.matrix(d), file.path(out, paste0("distance_", m, ".csv")))
}

save_result <- function(result, name) {
  write.csv(result$Hier.part,
            file.path(out, paste0(name, "_hier_part.csv")))
  write.csv(result$Var.part,
            file.path(out, paste0(name, "_var_part.csv")))
  write.csv(
    data.frame(total = result$Total_explained_variation),
    file.path(out, paste0(name, "_total.csv")),
    row.names = FALSE
  )
}

# 普通 dbRDA
d_bray <- vegdist(mite, method = "bray")
xy <- as.data.frame(mite.xy)

res_single <- rdacca.hp(
  d_bray, xy,
  method = "dbRDA",
  type = "adjR2",
  dbrdatype = "dbrda",
  var.part = TRUE
)
save_result(res_single, "dbrda_bray_xy")

# 分组 dbRDA
groups <- list(
  Environment = mite.env,
  Spatial = as.data.frame(mite.xy),
  PCNM = as.data.frame(mite.pcnm[, 1:3, drop = FALSE])
)

res_group <- rdacca.hp(
  d_bray, groups,
  method = "dbRDA",
  type = "adjR2",
  dbrdatype = "dbrda",
  var.part = TRUE
)
save_result(res_group, "dbrda_bray_groups")

capture.output(sessionInfo(),
               file = file.path(out, "R_session_info.txt"))

########################################################
library(vegan)
library(rdacca.hp)

data(mite)
data(mite.xy)

out <- "benchmark/dbrda_options_reference"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

dv <- vegdist(mite, method = "bray")
iv <- as.data.frame(mite.xy)

save_result <- function(result, name) {
  write.csv(
    result$Hier.part,
    file.path(out, paste0(name, "_hier_part.csv"))
  )
  
  write.csv(
    result$Var.part,
    file.path(out, paste0(name, "_var_part.csv"))
  )
  
  write.csv(
    data.frame(total = result$Total_explained_variation),
    file.path(out, paste0(name, "_total.csv")),
    row.names = FALSE
  )
}

settings <- list(
  dbrda_default = list(
    dbrdatype = "dbrda",
    add = FALSE,
    sqrt.dist = FALSE
  ),
  dbrda_sqrt = list(
    dbrdatype = "dbrda",
    add = FALSE,
    sqrt.dist = TRUE
  ),
  dbrda_lingoes = list(
    dbrdatype = "dbrda",
    add = "lingoes",
    sqrt.dist = FALSE
  ),
  dbrda_cailliez = list(
    dbrdatype = "dbrda",
    add = "cailliez",
    sqrt.dist = FALSE
  ),
  capscale_default = list(
    dbrdatype = "capscale",
    add = FALSE,
    sqrt.dist = FALSE
  )
)

for (setting_name in names(settings)) {
  setting <- settings[[setting_name]]
  
  for (r2_type in c("R2", "adjR2")) {
    result <- rdacca.hp(
      dv = dv,
      iv = iv,
      method = "dbRDA",
      type = r2_type,
      dbrdatype = setting$dbrdatype,
      add = setting$add,
      sqrt.dist = setting$sqrt.dist,
      var.part = TRUE
    )
    
    output_name <- paste(
      setting_name,
      tolower(r2_type),
      sep = "_"
    )
    
    save_result(result, output_name)
  }
}

capture.output(
  sessionInfo(),
  file = file.path(out, "R_session_info.txt")
)


########################################################
library(vegan)
library(rdacca.hp)

data(mite)
data(mite.env)
data(mite.xy)
data(mite.pcnm)

out <- "benchmark/cca_r2_reference"
dir.create(out, recursive = TRUE, showWarnings = FALSE)

save_result <- function(result, name) {
  write.csv(
    result$Hier.part,
    file.path(out, paste0(name, "_hier_part.csv"))
  )
  
  write.csv(
    result$Var.part,
    file.path(out, paste0(name, "_var_part.csv"))
  )
  
  write.csv(
    data.frame(total = result$Total_explained_variation),
    file.path(out, paste0(name, "_total.csv")),
    row.names = FALSE
  )
}

# 1. 两个连续解释变量
xy <- as.data.frame(mite.xy)

cca_xy <- rdacca.hp(
  dv = mite,
  iv = xy,
  method = "CCA",
  type = "R2",
  var.part = TRUE
)

save_result(cca_xy, "cca_xy_r2")

# 2. 包含分类变量和有序因子的解释变量表
cca_env <- rdacca.hp(
  dv = mite,
  iv = mite.env,
  method = "CCA",
  type = "R2",
  var.part = TRUE
)

save_result(cca_env, "cca_env_r2")

# 3. 分组解释变量
groups <- list(
  Environment = mite.env,
  Spatial = as.data.frame(mite.xy),
  PCNM = as.data.frame(mite.pcnm[, 1:3, drop = FALSE])
)

cca_groups <- rdacca.hp(
  dv = mite,
  iv = groups,
  method = "CCA",
  type = "R2",
  var.part = TRUE
)

save_result(cca_groups, "cca_groups_r2")

capture.output(
  sessionInfo(),
  file = file.path(out, "R_session_info.txt")
)
