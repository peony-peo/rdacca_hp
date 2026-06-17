library(vegan)
library(rdacca.hp)
library(jsonlite)

data(mite)
data(mite.env)

mite.hel <- decostand(mite, "hellinger")

iv <- mite.env[, c("WatrCont", "Substrate", "Shrub")]

# 明确把 Shrub 设为有序因子
iv$Shrub <- ordered(iv$Shrub, levels = c("None", "Few", "Many"))

res <- rdacca.hp(
  dv = mite.hel,
  iv = iv,
  method = "RDA",
  type = "adjR2",
  scale = FALSE,
  var.part = TRUE
)

var_part_names <- trimws(rownames(res$Var.part))
hier_part_names <- trimws(rownames(res$Hier.part))

# 统一名称格式，避免控制台与 JSON 格式差异
var_part_names <- gsub(", and ", " and ", var_part_names, fixed = TRUE)
var_part_names <- gsub(" ,", ",", var_part_names, fixed = TRUE)

out <- list(
  case = "rda_ordered_factor_mixed",
  total_explained_variation = round(unname(res$Total_explained_variation), 3),
  var_part = list(
    rownames = var_part_names,
    values = round(unname(res$Var.part[, 1]), 4),
    perc = round(unname(res$Var.part[, 2]), 2)
  ),
  hier_part = list(
    rownames = hier_part_names,
    unique = round(unname(res$Hier.part[, "Unique"]), 4),
    average_share = round(unname(res$Hier.part[, "Average.share"]), 4),
    individual = round(unname(res$Hier.part[, "Individual"]), 4),
    perc = round(unname(res$Hier.part[, "I.perc(%)"]), 2)
  )
)

dir.create("benchmark/expected", recursive = TRUE, showWarnings = FALSE)

write_json(
  out,
  "benchmark/expected/rda_ordered_factor_mixed.json",
  pretty = TRUE,
  auto_unbox = TRUE
)

print(res$Var.part)
print(res$Hier.part)
print(out)
