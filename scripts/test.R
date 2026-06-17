library(ade4)
data(doubs)

write.csv(doubs$fish, "doubs_fish.csv")
write.csv(doubs$env, "doubs_env.csv")

library(vegan)

data(mite)
data(mite.env)

write.csv(mite, "mite.csv", row.names = TRUE)
write.csv(mite.env, "mite_env.csv", row.names = TRUE)
