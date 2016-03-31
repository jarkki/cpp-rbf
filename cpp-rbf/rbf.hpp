/* Linearly parameterized Gaussian Radial Basis Function approximator
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
#pragma once

#include <armadillo>
#include <math.h>

using namespace std;
using namespace arma;

namespace rbf{

  /*!  Linearly parameterized Gaussian Radial Basis Function approximator
   *
   *
   *                        y = phi(x) * w,
   *
   *   where x is the input, y is the target output, phi(x) is the basis
   *   function (Gaussian) and w is the weights.
   *
   *   The weights are solved for with least squares approximation.
   *
   *
   */
  class GaussianRBF{
  public:

    size_t dim;          // Number of variables (input dimension)
    mat    centroids;    // Centroids
    size_t nbasis;       // Number of centroids (the basis is of same size without constant)
    double gamma;        // Gaussian scale parameter
    vec    weights;      // Vector of weights
    bool   normalize;    // Whether we should normalize the basis to sum to one
    bool   add_constant; // Whether we should add constant 1 in front of the basis vector

    /*!  Constructor
     *
     *  @param centroids    : Matrix of size ncentroids x nvariables
     *  @param gamma        : Scales the Gaussian density
     *  @param normalize    : Whether to normalize the basis values so they sum to one
     *  @param add_constant : Whether to add constant 1 to the basis vector
     */
    GaussianRBF(const mat centroids, const double gamma, bool normalize=false, bool add_constant=false){
      this->dim = size(centroids)[1];
      this->centroids = centroids;
      this->nbasis = size(centroids)[0];
      this->gamma = gamma;
      this->weights = randu(size(centroids)[0]);
      this->normalize = normalize;
      this->add_constant = add_constant;
    }

    /*! Solve for the linear system for weights w:
     *
     *  y = phi(x) * w
     *
     *  @param X  : Input matrix of size ninputs x nvariables
     *  @param y  : Vector of target values
    */
    void fit(const mat & X, const vec & y){

      // Check that the input and target have the same length
      if (X.n_rows != y.n_rows){
        throw invalid_argument("X y must have same number of rows!");
      }

      vec what, phi_i;
      size_t ninputs= X.n_rows;
      size_t phi_cols = add_constant ? this->centroids.n_rows + 1 : this->centroids.n_rows;
      mat phi(ninputs, phi_cols);

      // For each input
      for(int i=0; i < ninputs; ++i){

        // Apply the basis to inputs
        phi_i = this->basis(X.row(i));

        // Add phi to phi-matriz
        phi.row(i) = phi_i.t();
      }

      // Solve the system
      what = solve(phi,y);
      this->weights = what;

    }

    /*! Calculate y = phi(x) * w
     *
     *
     *  @param X  : Input matrix of size ninputs x nvariables
     *
     *  @retval   : vector of outputs y = phi(x) * w
     */
    vec predict(const mat & X) const{

      if (size(X)[1] != size(this->centroids)[1]){
        throw invalid_argument("X and centroids must have same number of columns!");
      }
      vector<mat> phi(X.n_rows);
      vec y(X.n_rows);

      // For each input
      for(int i=0; i < X.n_rows; ++i){

        // Apply the basis
        vec phi_i = this->basis(X.row(i));

        // Apply weights
        vec pred = this->weights.t() * phi_i; // * gives the dot product
        y(i) = pred(0);
      }
      return y;
    }

    vec operator()(const mat & X){
      return this->predict(X);
    }

    /*! Multivariate Gaussian density
     *
     *   f(x)  = e^{-\gamma * sum(x-c)}
     *
     *  @param X  : Input vector of size nvariables
     *
     *  @retval   : Gaussian density vector
     */
    vec basis(const rowvec & x) const{
      if (x.size() != this->centroids.n_cols){
        throw invalid_argument("length of x must equal the number centroid columns!");
      }

      // Precalc the diff x-c
      mat cdiff(size(this->centroids));
      for (int i = 0; i < this->centroids.n_rows; ++i) {
        cdiff.row(i) = x - this->centroids.row(i);
      }

      // Multidimensional normal density
      vec phi = arma::exp(-1.0 * this->gamma * arma::sum(cdiff % cdiff,1)); // % gives element-wise product

      // Normalize to sum to one?
      if (this->normalize){
        phi = phi/sum(phi);
      }

      // Add constant of 1 to the front of the basis vector?
      if (add_constant){
        vec phi_padded = ones(phi.size()+1);
        phi_padded(span(1,phi.size()),0) = phi;
        phi = phi_padded;
      }

      return phi;
    }
  };

}
