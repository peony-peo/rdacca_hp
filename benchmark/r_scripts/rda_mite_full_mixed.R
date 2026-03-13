library(vegan)
library(rdacca.hp)
library(jsonlite)

data(mite)
data(mite.env)

mite.hel <- decostand(mite, "hellinger")

res <- rdacca.hp(
  dv = mite.hel,
  iv = mite.env,
  method = "RDA",
  type = "adjR2",
  scale = FALSE,
  var.part = TRUE
)

var_part_names <- trimws(rownames(res$Var.part))
hier_part_names <- trimws(rownames(res$Hier.part))

var_part_names <- gsub(", and ", " and ", var_part_names, fixed = TRUE)
var_part_names <- gsub(" ,", ",", var_part_names, fixed = TRUE)

out <- list(
  case = "rda_mite_full_mixed",
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

dir.create("benchmark/expected", recursive = TRUE, showWarnings = FALSE)

write_json(
  out,
  "benchmark/expected/rda_mite_full_mixed.json",
  pretty = TRUE,
  auto_unbox = TRUE
)

print(res$Var.part)
print(res$Hier.part)
print(out)
