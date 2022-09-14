#include <TMB.hpp>

using namespace Eigen;
using namespace density;

template<class Type>
array<Type> reshape(vector<Type> x, int N, int k) {
  matrix<Type> y = x;
  y.resize(N, k);
  array<Type> a(N, k);
  for (size_t i = 0; i < a.cols(); i++) {
    a.col(i) = y.col(i);
  }
  return a;
}

template<class Type>
Type IID(vector<Type> x, Type log_sigma, SparseMatrix<Type> Q) {
  Type sigma = exp(log_sigma);
  SparseMatrix<Type> Q_adj = (1/sigma) * Q;

  GMRF_t<Type> d(Q_adj);
  Type res = 0;
  res += dnorm(sigma, Type(0), Type(1), true);
  res -= d(x);
  return res;
}

template<class Type>
Type RW(vector<Type> x, Type log_sigma) {
  Type sigma = exp(log_sigma);
  Type res = dnorm(x[0] / sigma, Type(0), Type(1), true);
  for (size_t i = 1; i < x.size(); i++) {
    res += dnorm((x[i]-x[i-1]) / sigma, Type(0), Type(1), true);
  }
  res += dnorm(sigma, Type(0), Type(1), true);

  return res;
}

template<class Type>
Type AR(vector<Type> x, Type log_sigma, Type logit_phi) {
  Type sigma = exp(log_sigma);
  Type phi = 1 / (1 + exp(-logit_phi));

  Type res = -AR1(phi)(x/sigma);

  return res;
}

template<class Type>
Type ARIMA_1d0(vector<Type> x, SparseMatrix<Type> D, Type log_sigma, Type logit_phi) {
  vector<Type> x_diff = (D * x);

  return AR(x_diff, log_sigma, logit_phi);
}

template<class Type>
Type GMRF_GMRF(vector<Type> x, SparseMatrix<Type> Q1, SparseMatrix<Type> Q2,
  Type log_sigma1, Type log_sigma2)
{
  array<Type> x_arr = reshape(x, Q1.rows(), Q2.cols());

  Type sigma1 = exp(log_sigma1);
  Type sigma2 = exp(log_sigma2);

  SparseMatrix<Type> Q1_adj = (1/sigma1) * Q1;
  SparseMatrix<Type> Q2_adj = (1/sigma2) * Q2;

  Type res = -SEPARABLE(GMRF(Q2_adj), GMRF(Q1_adj))(x_arr);
  return res;
}

template<class Type>
Type GMRF_AR1(vector<Type> x, SparseMatrix<Type> Q1,
  Type log_sigma, Type logit_phi2)
{
  array<Type> x_arr = reshape(x, Q1.rows(), x.size()/Q1.rows());

  Type sigma = exp(log_sigma);
  Type phi2 = 1 / (1 + exp(-logit_phi2));

  Type res = -SEPARABLE(AR1(phi2), GMRF(Q1))(x_arr/sigma);
  return res;
}

template<class Type>
Type GMRF_ARIMA(vector<Type> x, SparseMatrix<Type> Q1,
  SparseMatrix<Type> D, Type log_sigma, Type logit_phi2)
{
  vector<Type> x_diff = (D * x);
  array<Type> x_arr = reshape(x_diff, Q1.rows(), x_diff.size()/Q1.rows());

  Type sigma = exp(log_sigma);
  Type phi2 = 1 / (1 + exp(-logit_phi2));

  Type res = -SEPARABLE(AR1(phi2), GMRF(Q1))(x_arr/sigma);
  return res;
}

template<class Type>
Type objective_function<Type>::operator() ()
{
  DATA_VECTOR(Y1);
  DATA_VECTOR(m1);
  DATA_SPARSE_MATRIX(Q1);
  DATA_SCALAR(log_s1);

  DATA_VECTOR(Y2);
  DATA_SCALAR(log_s2);

  DATA_VECTOR(Y3);
  DATA_SCALAR(log_s3);
  DATA_SCALAR(logit_phi3);

  DATA_VECTOR(Y4);
  DATA_SPARSE_MATRIX(D4);
  DATA_SCALAR(log_s4);
  DATA_SCALAR(logit_phi4);

  DATA_ARRAY(Y5);
  DATA_VECTOR(Y5_v);
  DATA_SCALAR(log_s5_1);
  DATA_SCALAR(log_s5_2);
  DATA_SPARSE_MATRIX(Q5_1);
  DATA_SPARSE_MATRIX(Q5_2);

  DATA_VECTOR(Y6);
  DATA_SCALAR(log_s6_1);
  DATA_SCALAR(logit_phi6_2);
  DATA_SPARSE_MATRIX(Q6_1);

  DATA_VECTOR(Y7);
  DATA_SCALAR(log_s7_1);
  DATA_SCALAR(logit_phi7_2);
  DATA_SPARSE_MATRIX(Q7_1);
  DATA_SPARSE_MATRIX(D7);

  PARAMETER(junk);

  Type nll = 0;

  // IID
  Type dens1 = IID(Y1, log_s1, Q1);
  REPORT(dens1);

  // RW1
  Type dens2 = RW(Y2, log_s2);
  REPORT(dens2);

  // AR1
  Type dens3 = AR(Y3, log_s3, logit_phi3);
  REPORT(dens3);

  // ARIMA(1, d, 0)
  Type dens4 = ARIMA_1d0(Y4, D4, log_s4, logit_phi4);
  REPORT(dens4);

  Type dens5 = GMRF_GMRF(Y5_v, Q5_1, Q5_2, log_s5_1, log_s5_2);
  REPORT(dens5);

  Type dens6 = GMRF_AR1(Y6, Q6_1, log_s6_1, logit_phi6_2);
  REPORT(dens6);

  Type dens7 = GMRF_ARIMA(Y7, Q7_1, D7, log_s7_1, logit_phi7_2);
  REPORT(dens7);

  return nll;
}
