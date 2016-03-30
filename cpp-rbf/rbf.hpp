/* Gaussian Radial Basis Function approximator
 *
 * Copyright (C) 2016  Jarno Kiviaho <jarkki@kapsi.fi>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <armadillo>
#include <math.h>

using namespace std;
using namespace arma;

namespace cpp-rbf{

  class GaussianRBF{
  public:

    size_t dim;
    mat    centroids;
    mat    domain;
    size_t nbasis;
    double gamma;
    vec    weights;

    GaussianRBF(const mat centroids, const double gamma, mat domain){
      this->dim = size(centroids)[1]; // Dimension of the inputs
      this->centroids = centroids;
      this->domain = domain;
      this->nbasis = size(centroids)[0];
      this->gamma = gamma;
      this->weights = uniform(-1.0, 1.0,size(centroids)[0]);
    }

    void GaussianRBF::fit(const mat & X, const vec & y){
      // Solve for w^hat with linear system
      // phi * w = y, or
      // y = phi * w
      // w^hat = (phi^T * phi)^-1 * phi^T * y
      vec what(this->weights.size());
      // First calculate the basis matrix phi ()
      mat phi(X.n_rows, this->centroids.n_rows);
      for(int i=0; i < X.n_rows; ++i){

        // Apply the basis
        vec phi_i = this->basis(X.row(i));

        // // Normalize so that sum(phi_i) == 1
        // phi_i = phi_i/sum(phi_i);

        phi.row(i) = phi_i.t();
      }

      // Solve the system
      what = solve(phi,y);
      this->weights = what;

    }
    vec GaussianRBF::predict(const mat & X) const{

      if (size(X)[1] != size(this->centroids)[1]){
        throw invalid_argument("X and centroids must have same number of columns!");
      }
      vector<mat> phi(X.n_rows);
      vec y(X.n_rows);

      // For each input
      for(int i=0; i < X.n_rows; ++i){

        // Apply the basis
        vec phi_i = this->basis(X.row(i));

        // // Normalize so that sum(phi_i) == 1
        // phi_i = phi_i/sum(phi_i);

        // Apply weights
        vec pred = this->weights.t() * phi_i; // * gives the dot product
        y(i) = pred(0);
      }

      return y;
    }

    vec GaussianRBF::basis(const rowvec & x) const{
      if (x.size() != this->centroids.n_cols){
        throw invalid_argument("x and centroids must have same dimensions!");
      }

      // // Check if x is within the domain and put it inside the domain (on the edge) if necessary
      // rowvec x_(x.size());
      // for (int i = 0; i < x.size(); ++i) {
      //   if (x(i) < this->domain(i,0)) {
      //     x_(i) = this->domain(i,0);
      //   }else if (x(i) > this->domain(i,1)){
      //     x_(i) = this->domain(i,1);
      //   }else{
      //     x_(i) = x(i);
      //   }
      // }

      // Precalc the diff x-c
      mat cdiff(size(this->centroids));
      for (int i = 0; i < this->centroids.n_rows; ++i) {
        cdiff.row(i) = x - this->centroids.row(i);
      }

      // Multidimensional normal density
      vec phi = arma::exp(-1.0 * this->gamma * arma::sum(cdiff % cdiff,1)); // dim=1 means row sums

      // Normalize
      phi = phi/sum(phi);

      return phi;
    }
  }

}
