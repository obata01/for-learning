workdir = paste(getwd(), "/workspace", sep="")
setwd(workdir)
getwd()

# P15 データのロードと確認
load("./data/data2.RData")
data
length(data)
summary(data)

# P16 ヒストグラム
table(data)
hist(data, breaks=seq(-0.5, 9.5, 1))
var(data)
sd(data)
sqrt(var(data))

# P19 ポアソン分布
y <- 0:9
prob <- dpois(y, lambda = 3.56)
plot(y, prob, type="b", lty=2)
cbind(y, prob)

hist(data, breaks=seq(-0.5, 9.5, 1))
lines(y, 50*prob, type='b', lty=2)

# 最尤推定
logL <- function(m) sum(dpois(data, m, log=TRUE))
lambda <- seq(2, 5, 0.1)
plot(lambda, sapply(lambda, logL), type="l")
