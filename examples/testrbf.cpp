/* Test the RBF with noisy 2d and 3d data. Plot with python/matplotlib.
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

#include <iostream>
#include <tuple>
#include <armadillo>

#include "cpp-rbf/rbf.hpp"
#include "utils.hpp"

using namespace std;
using namespace arma;
using namespace rbf;


/*!    Approximates 2d and 3d noisy sin functions.
 *     Plots the results with Python.
 *
 *
 */
int main(int argc, char *argv[])
{
  // Seed the rng
  // arma_rng::set_seed_random();

  mat X,X2,X3;
  vec y,yhat,yhat2;

  // Gaussian scale parameter
  double gamma = 0.1;

  // Create test data in interval [a,b]
  double a = -3.0;
  double b = 3.0;
  double sd = 0.3; // Standard deviation of Gaussian noise for the test data

  // Create centroids
  size_t ncentroids = 4;
  vec centroids = linspace<vec>(a, b, ncentroids);
  mat centroids3d = combinations(centroids, centroids);

  // Prediction data
  X2 = linspace(a,b, 50);

  // For 3d test
  X3 = combinations(X2,X2);

  /*
   *  Test 2d
   *
   */
  // Test data for one input variable
  tie(X,y) = create_test_data_2d(a,b,0.3,50);

  // Instantiate rbf
  GaussianRBF rbf2d(centroids,gamma,false, true);

  // Fit to the test data
  rbf2d.fit(X,y);

  // Predict
  yhat = rbf2d.predict(X2);

  // Plot
  plot_2d(X,y,X2,yhat);

  /*
   *  Test 3d
   *
   */
  // 3d test data
  tie(X,y) = create_test_data_3d(a,b,sd,300);

  // 3d rbf
  GaussianRBF rbf3d(centroids3d, gamma);

  // Fit
  rbf3d.fit(X,y);

  // Predict
  yhat2 = rbf3d.predict(X3);

  // Plot
  plot3d(X,y,X3,yhat2,X2.size());


  return 0;
}
