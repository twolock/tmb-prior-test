#include <TMB.hpp>

using namespace Eigen;
using namespace density;

template<class Type>
array<Type> reshape(vector<Type> x, int N, int k) {
  matrix<Type> tmp_y = x;
  tmp_y.resize(k, N);
  matrix<Type> y = tmp_y.transpose();
  array<Type> a(N, k);
  for (size_t i = 0; i < a.cols(); i++) {
    a.col(i) = y.col(i);
  }
  return a;
}

template<class Type>
Type GMRF_AR1(vector<Type> x, SparseMatrix<Type> Q,
  Type log_sigma, Type logit_phi)
{
  int N = Q.rows();
  int k = x.size()/N;
  array<Type> x_arr = reshape(x, N, k);

  Type sigma = exp(log_sigma);
  Type phi = 2 / (1 + exp(-logit_phi)) - 1;

  Type res = -SEPARABLE(AR1(phi), GMRF(Q))(x_arr/sigma);
  res += dnorm(sigma, Type(0), Type(2), true);
  res += dnorm(logit_phi, Type(0), Type(1), true);
  res += log_sigma;

  matrix<Type> tmp_m = x_arr;

  for (size_t i = 0; i < N; i++)
  {
    res += dnorm(tmp_m.row(i).sum(), Type(0), Type(0.1) * k, true);
  }
  

  return res;
}

template<class Type>
Type objective_function<Type>::operator() ()
{

  DATA_VECTOR(y);
  DATA_SPARSE_MATRIX(Q_GMRF);

  PARAMETER_VECTOR(X);
  PARAMETER(log_sigma_X);
  PARAMETER(logit_rho_X);
  PARAMETER(log_sigma_y);

  Type nll = 0;

  nll -= GMRF_AR1(X, Q_GMRF, log_sigma_X, logit_rho_X);
  
  nll -= sum(dnorm(y, X, exp(log_sigma_y), true));
  nll -= dnorm(exp(log_sigma_y), Type(0), Type(2), true);
  nll -= log_sigma_y;

//   array<Type> x_arr = reshape(X, Q_GMRF.rows(), X.size()/Q_GMRF.rows());
//   REPORT(x_arr);

//   matrix<Type> tmp_reshaped = X;
//   tmp_reshaped.resize(X.size()/Q_GMRF.rows(), Q_GMRF.rows());
//   matrix<Type> X_reshaped = tmp_reshaped.transpose();
//   REPORT(X_reshaped);

  Type sigma_X = exp(log_sigma_X);
  Type rho = 2/(1+exp(-logit_rho_X))-1;
  Type sigma_y = exp(log_sigma_y);
  
  ADREPORT(sigma_X);
  ADREPORT(rho);
  ADREPORT(sigma_y);

  return nll;
}
