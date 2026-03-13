library(vegan)
library(rdacca.hp)
library(jsonlite)

data(mite)
data(mite.env)

# Hellinger transform
mite.hel <- decostand(mite, "hellinger")

# Two numeric predictors
iv <- mite.env[, c("SubsDens", "WatrCont")]

res <- rdacca.hp(
  dv = mite.hel,
  iv = iv,
  method = "RDA",
  type = "adjR2",
  scale = FALSE,
  var.part = TRUE
)

# 去掉尾部空格
var_part_names <- trimws(rownames(res$Var.part))
hier_part_names <- trimws(rownames(res$Hier.part))

# 同时把 " , and " 这种怪格式修正掉
var_part_names <- gsub(", and ", " and ", var_part_names, fixed = TRUE)
var_part_names <- gsub(" ,", ",", var_part_names, fixed = TRUE)

out <- list(
  case = "rda_numeric_2vars",
  total_explained_variation = unname(res$Total_explained_variation),
  var_part = list(
    rownames = var_part_names,
    values = unname(res$Var.part[, 1]),
    perc = unname(res$Var.part[, 2])
  ),
  hier_part = list(
    rownames = hier_part_names,
    unique = unname(res$Hier.part[, "Unique"]),
    average_share = unname(res$Hier.part[, "Average.share"]),
    individual = unname(res$Hier.part[, "Individual"]),
    perc = unname(res$Hier.part[, "I.perc(%)"])
  )
)

# 保存输入数据
dir.create("benchmark/data", recursive = TRUE, showWarnings = FALSE)
dir.create("benchmark/expected", recursive = TRUE, showWarnings = FALSE)

write.csv(mite, "benchmark/data/mite.csv", row.names = FALSE)
write.csv(mite.env, "benchmark/data/mite_env.csv", row.names = FALSE)

# 保存 JSON
write_json(
  out,
  "benchmark/expected/rda_numeric_2vars.json",
  pretty = TRUE,
  auto_unbox = TRUE
)

print(res$Var.part)
print(res$Hier.part)
print(out)

fit1 <- rda(mite.hel ~ SubsDens, data = mite.env[, c("SubsDens","WatrCont")])
fit2 <- rda(mite.hel ~ WatrCont, data = mite.env[, c("SubsDens","WatrCont")])
fit12 <- rda(mite.hel ~ SubsDens + WatrCont, data = mite.env[, c("SubsDens","WatrCont")])

RsquareAdj(fit1)
RsquareAdj(fit2)
RsquareAdj(fit12)

res <- rdacca.hp(
  dv = mite.hel,
  iv = mite.env[, c("SubsDens","WatrCont")],
  method = "RDA",
  type = "adjR2",
  scale = FALSE,
  var.part = TRUE
)

res$Var.part
res$Hier.part
