#include <iostream>
#include <tuple>
#include <armadillo>

#include "cpp-rbf/rbf.hpp"

using namespace std;
using namespace arma;
using namespace mc-rbf;

tuple <mat,vec> create_test_data(double a, double b, double sd, int n){

  // Uniformly sampled points
  mat X = uniform(a,b,n,1);
  // Sort
  X = sort(X);
  // Sin with Gaussian noise
  vec y = arma::sin(X) + (norm(n,1) * sd);

  return make_tuple (X,y);
}


tuple <mat,vec> create_test_data_3d(double a, double b, double sd, int n){

  // Uniformly sampled points
  mat X = uniform(a,b,n,2);
  // Sin
  vec y = arma::sin(X.col(0)) % arma::sin(X.col(1)); // % is element-wise multiplication
  // Add noise
  y = y + (norm(n,1) * sd);
  return make_tuple (X,y);
}

int main(int argc, char *argv[])
{
  // arma_rng::set_seed_random();
  mat X;
  vec y;
  double a = -3.5;
  double b = 3.5;
  int ncentroids = 4;
  double gamma = 0.1;

  // // One state variable
  // tie(X,y) = create_test_data(a,b,0.1,50);
  // vec centroids = linspace<vec>(a, b, ncentroids);
  // shared_ptr<ParametricApproximator> approx (new GaussianRBF(centroids,gamma));
  // approx->fit(X,y);
  // vec yhat = approx->predict(X);
  // plot_approx(X,y,yhat);

  // tie(X,y) = create_test_data_3d(a,b,0.001);
  // plot3d(X,y);

  // Create centroids for 2-dim state space
  // vec c = linspace<vec>(a, b, ncentroids);
  // mat centroids = permutations(c,c);
  // double gamma = 0.1;
  // shared_ptr<ParametricApproximator> approx (new GaussianRBF(centroids,gamma));
  // approx->fit(X,y);
  // vec yhat = approx->predict(X);
  // plot3d(X,yhat);
  // yhat.print();


  


  return 0;
}
