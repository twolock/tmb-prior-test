library(TMB)
library(mvnfast)
library(Matrix)
library(ar.matrix)

source('make_D_matrix.R')

# set.seed(1000)

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
  Y3[i] <- phi3 * Y3[i-1] + eps[i-1]
}

Y3_adj <- Y3 / s3
dens3 <- dnorm(Y3_adj[1], 0, 1, T)
s3_ss <- sqrt(1-phi3^2)
for (i in 2:p3) {
  dens3 <- dens3 + dnorm((Y3_adj[i]-phi3*Y3_adj[i-1])/s3_ss,0, 1, T)
}
dens3 <- dens3 - (p3-1) * log(s3_ss)

## ARIMA(1, d, 0)
p4 <- 100
s4 <- 2
d4 <- 3
D4 <- make_D_matrix(p4, d4)
Y4 <- rnorm(p4)
phi4 <- 0.5

Y4_diff <- (D4 %*% Y4) / s4
dens4 <- dnorm(Y4_diff[1], 0, 1, T)
s4_ss <- sqrt(1-phi4^2)
for (i in 2:length(Y4_diff)) {
  print(dens4)
  dens4 <- dens4 + dnorm((Y4_diff[i]-phi4*Y4_diff[i-1])/s4_ss,0, 1, T)
}
dens4 <- dens4 - (length(Y4_diff)-1) * log(s4_ss)

## IID x IID
p5_1 <- 10
p5_2 <- 20
s5_1 <- 0.2
s5_2 <- 2
S5_1 <- diag(s5_1, p5_1, p5_1)
S5_2 <- diag(s5_2, p5_2, p5_2)
S5 <- kronecker(S5_2, S5_1)

Q5_1 <- as(diag(1, p5_1, p5_1), 'dgCMatrix')
Q5_2 <- as(diag(1, p5_2, p5_2), 'dgCMatrix')
Y5_v <- 1:(p5_1 * p5_2)
Y5 <- matrix(Y5_v, nrow=p5_1, ncol=p5_2)
dens5 <- dmvn(as.vector(Y5), rep(0, p5_1 * p5_2), S5, log = T)

## IID x AR1
p6_1 <- 10 ## IID dimension
p6_2 <- 50 ## AR1 dimension
s6_1 <- 2
s6_2 <- 1
phi6_2 <- 0.5

Q6_1 <- as(diag(1, p6_1, p6_1), 'dgCMatrix')

## IID is primary index, AR1 is secondary
## Y6=(x_{11}, x_{12}, ... x_{1T}, x_{21}, ..., x_{2T}, ..., x{NT})
## Confusingly in model.matrix language, this should be:
## model.matrix(~ 0 + time:space)
Y6 <- rep(1:p6_1, p6_2)

S6_1 <- solve(Q6_1)
S6_2 <- solve(Q.AR1(p6_2, 1, phi6_2))

S6 <- kronecker(S6_2, S6_1)
s6_ss <- sqrt(1-phi6_2^2)
dens6 <- dmvn(Y6/(s6_1*s6_ss), rep(0, p6_1 * p6_2), S6, log = T)
dens6 <- dens6 - p6_2 * p6_1 * log(s6_ss)

## GMRF x ARIMA(1, d, 0)
p7_1 <- 10 ## IID dimension
p7_2 <- 50 ## AR1 dimension
s7_1 <- 2
s7_2 <- 1
phi7_2 <- 0.5

D7 <- as(make_D_matrix_hier(p7_1, p7_2, 2), 'dgCMatrix')
N_diff <- nrow(D7)/p7_1
Q7_1 <- as(diag(1, p7_1, p7_1), 'dgCMatrix')
Y7 <- rnorm(p7_1 * p7_2)
Y7_diff <- D7 %*% Y7
S7_1 <- solve(Q7_1)
S7_2 <- solve(Q.AR1(N_diff, 1, phi7_2))

S7 <- kronecker(S7_2, S7_1)
s7_ss <- sqrt(1-phi7_2^2)
dens7 <- dmvn(Y7_diff/(s7_1*s7_ss), rep(0, p7_1 * N_diff), S7, log = T)
dens7 <- dens7 - N_diff * p7_1 * log(s7_ss)

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
  logit_phi3 = log(phi3/(1-phi3)),
  
  Y4 = Y4,
  D4 = as(D4, 'dgCMatrix'),
  log_s4 = log(s4),
  logit_phi4 = log(phi4/(1-phi4)),
  
  Y5 = Y5,
  Y5_v = Y5_v,
  log_s5_1 = log(s5_1),
  log_s5_2 = log(s5_2),
  Q5_1 = Q5_1,
  Q5_2 = Q5_2,
  
  Y6 = Y6,
  log_s6_1 = log(s6_1),
  log_s6_2 = log(s6_2),
  logit_phi6_2 = log(phi6_2/(1-phi6_2)),
  Q6_1 = Q6_1,
  
  Y7 = Y7,
  log_s7_1 = log(s7_1),
  logit_phi7_2 = log(phi7_2/(1-phi7_2)),
  Q7_1 = Q7_1,
  D7 = D7
)
in_par <- list (
  junk = 0
)

obj <- MakeADFun(in_dat, in_par)
cbind(dens1, obj$report()$dens1)
cbind(dens2, obj$report()$dens2)
cbind(dens3, obj$report()$dens3)
cbind(dens4, obj$report()$dens4)
cbind(dens5, obj$report()$dens5)
cbind(dens6, obj$report()$dens6)
cbind(dens6, obj$report()$dens6)
cbind(dens7, obj$report()$dens7)
