#include <TMB.hpp>

using namespace Eigen;
using namespace density;

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

  return nll;
}
