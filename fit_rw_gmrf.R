library(TMB)
library(mvnfast)
library(Matrix)
library(ar.matrix)
library(data.table)
library(ggplot2)

set.seed(959595)

## Compile and load TMB model
compile('fit_rw_gmrf.cpp', flags='-w')
dyn.load(dynlib('fit_rw_gmrf'))

## Number of MVN samples to take
N_sample <- 1

## Number of time steps
N_t <- 100
## Number of regions
N_r <- 100

## Hyperparameters (poorly identified in this model)
sigma_Y <- 1
sigma_X <- 0.5
rho_X <- 0.99

## AR1 precision matrix
Q_AR <- Q.AR1(N_t, 1, rho_X)
## IID precision matrix
Q_IID <- as(diag(1, N_r), 'dgCMatrix')

## Joint precision matrix
## NB: first matrix is primary index
Q_joint <- kronecker(Q_IID, Q_AR)
## Run with small this to verify order of indexing
## Should be N_r dense blocks
## Will be very slow if N is large!
# image(solve(Q_joint))

## Sample underlying X
X <- t(rmvn(N_sample, rep(0, N_t * N_r), solve(Q_joint))) * sigma_X

## Build data.table
full_dt <- CJ(r_i = 1:N_r, t_i = 1:N_t)
full_dt[, X := X]

## Sample y given X and sigma_Y
full_dt[, y := rnorm(.N, X, sigma_Y)]

## TMB prep
in_dat <- list(
  y = full_dt[, y],
  Q_GMRF = Q_IID
)
in_par <- with(in_dat, {list(
  X = full_dt[, r_i],
  log_sigma_X = 0,
  logit_rho_X = 0,
  log_sigma_y = 0
)})
## Marginalise w.r.t. X
in_ran <- c('X')

## Make TMB model
obj <- MakeADFun(data = in_dat,
                 parameters = in_par,
                 random = in_ran,
                 method = 'L-BFGS-B',
                 DLL = 'fit_rw_gmrf')

## Find post. mode
opt <- do.call(optim, obj)
## Get uncertainty estimates
sd_obj <- sdreport(obj, getJointPrecision = T)

## Take conditional posterior samples of X
X_post <- rmvn(100, sd_obj$par.random, solve(sd_obj$jointPrecision[1:(N_t * N_r), 1:(N_t * N_r)]))
## Summarise posterior samples of X
X_summary <- t(apply(X_post, 2, quantile, probs=c(0.025, 0.5, 0.975)))
full_dt <- cbind(full_dt, X_summary)

## Plot posterior means and data
plot_i <- c(1, sample(2:N_r, pmin(N_r - 1, 5)))
ggplot(full_dt[r_i %in% plot_i], aes(x=t_i, color=factor(r_i), fill=factor(r_i))) + 
  geom_ribbon(aes(ymin=`2.5%`, ymax=`97.5%`), alpha=0.3, color=NA) +
  geom_line(aes(y=`50%`)) +
  geom_line(aes(y=X), linetype=2, color='black', alpha=0.3) +
  geom_point(aes(y=y), shape=3) +
  coord_cartesian(expand = 0) +
  theme_bw() + theme(legend.position = 'none') +
  facet_wrap('r_i') +
  labs(x='Time', y='Value')
