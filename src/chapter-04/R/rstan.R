# 実行手順
# $ R
# > setwd('/workspace/src/chapter-04/R/')
# > source("rstan.R")

library(rstan)

d <- read.csv(file='../data/data-salary.txt')
data <- list(N=nrow(d), X=d$X, Y=d$Y)
fit <- stan(file='model4-5.stan', data=data, seed=1234)

save.image(file='result-model4-5.RData')