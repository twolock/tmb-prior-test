library(TMB)
library(mvnfast)
library(Matrix)

set.seed(10)

compile('tmb_prior_test.cpp', flags='-w')
dyn.load(dynlib('tmb_prior_test'))

N <- 1

## IID
p1 <- 10

s1 <- 2
Q1 <- as(diag(1, p1), 'dgCMatrix')
S1 <- Q1 * s1
m1 <- rep(0, p1)

Y1 <- as.vector(rmvn(1, m1, S1))

dens1 <- dmvn(Y1, m1, S1, log=T)
dens1 <- dens1 + dnorm(s1, 0, 1, T)

## RW1
p2 <- 10
s2 <- 2
Y2 <- cumsum(c(0, rnorm(p2-1, 0, s2)))

Y2_scaled <- Y2 / s2
dens2 <- dnorm(Y2_scaled[1], 0, 1, T)
for (i in 2:p2) {
  dens2 <- dens2 + dnorm(Y2_scaled[i], Y2_scaled[i-1], 1, T)
}
dens2 <- dens2 + dnorm(s2, 0, 1, T)

## AR1
p3 <- 100
s3 <- s2
Y3 <- rep(0, p3)
phi3 <- 0.5
eps <- rnorm(p3-1, 0, s3)
for (i in 2:p3) {
  Y3[i] <- phi * Y3[i-1] + eps[i-1]
}

Y3_adj <- Y3 / s3
dens3 <- dnorm(Y3_adj[1], 0, 1, T)
s3_ss <- sqrt(1-phi3^2)
for (i in 2:p3) {
  dens3 <- dens3 + dnorm((Y3_adj[i]-phi3*Y3_adj[i-1])/s3_ss,0, 1, T)
}
dens3 <- dens3 - (p3-1) * log(s3_ss)

## IID x AR1

in_dat <- list(
  N = N,
  Y1 = Y1,
  m1 = m1,
  Q1 = Q1,  
  log_s1 = log(s1),
  
  Y2 = Y2,
  log_s2 = log(s2),
  
  Y3 = Y3,
  log_s3 = log(s3),
  logit_phi3 = log(phi3/(1-phi3))
  
)
in_par <- list (
  junk = 0
)

obj <- MakeADFun(in_dat, in_par)
cbind(dens1, obj$report()$dens1)
cbind(dens2, obj$report()$dens2)
cbind(dens3, obj$report()$dens3)
