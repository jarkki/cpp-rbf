/* Utility functions for testing and plotting the Gaussian RBF approximator
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

#include <iostream>
#include <tuple>
#include <armadillo>

#include "cpp-rbf/rbf.hpp"

using namespace std;
using namespace arma;

mat uniform(const double a, const double b, const int n_rows, const size_t n_cols){
  // Matrix of x ~ U(a,b) size (n_rows,n_cols)
  return a + (b-a)*randu(n_rows,n_cols);

}
vec uniform(const double a, const double b, const size_t n){
  // Vector of x ~ U(a,b) size n
  return a + (b-a)*randu(n);

}

tuple <mat,vec> create_test_data_2d(double a, double b, double sd, int n){

  // Uniformly sampled points
  mat X = uniform(a,b,n,1);
  // Sort
  X = sort(X);
  // Sin with Gaussian noise
  vec y = arma::sin(X) + (randn(n,1) * sd);

  return make_tuple (X,y);
}


tuple <mat,vec> create_test_data_3d(double a, double b, double sd, int n){

  // Uniformly sampled points
  mat X = uniform(a,b,n,2);
  // Sin
  vec y = arma::sin(X.col(0)) % arma::sin(X.col(1)); // % is element-wise multiplication
  // Add noise
  y = y + (randn(n,1) * sd);
  return make_tuple (X,y);
}

mat combinations(vec x1, vec x2){

  size_t n = x1.size() * x2.size();
  mat grid(n,2);
  int k = 0;
  for(int i = 0; i < x1.size() ; ++i){
    for(int j = 0; j < x2.size(); ++j){
      grid(j+k,0) = x1(i);
      grid(j+k,1) = x2(j);
    }
    k += x1.size();
  }
  return grid;
}

void plot_2d(const mat & X, const vec & y, const mat & X2, const vec & yhat){
  ofstream file;
  file.open ("plot.py");
  file << "import matplotlib.pyplot as plt" << endl;
  file << "plt.style.use('ggplot')" << endl;

  // Target
  file << "x = []" << endl;
  file << "y = []" << endl;
  file << "yhat = []" << endl;
  for (int i = 0; i < X.n_rows; i++) {
    file << "x.append(" << X(i,0) << ")" << endl;
    file << "y.append(" << y(i) << ")" << endl;
  }

  // Prediction
  file << "x2 = []" << endl;
  for (int i = 0; i < X2.n_rows; i++) {
    file << "x2.append(" << X2(i,0) << ")" << endl;
    file << "yhat.append(" << yhat(i) << ")" << endl;
  }

  file << "fig,ax = plt.subplots()" << endl;
  file << "ax.plot(x,y,'o',color='#8C7EEE')" << endl;
  file << "ax.plot(x2,yhat,lw=2,color='#34CEA4')" << endl;
  // file << "plt.xlabel('x')" << endl;
  // file << "plt.ylabel('y')" << endl;
  //  file << "plt.title('Sin with noise')" << endl;

  file << "ax.grid(b=False)" << endl;
  file << "ax.set_axis_bgcolor('#E5E5E5')" << endl;
  file << "ax.patch.set_facecolor('#E5E5E5')" << endl;
  file << "plt.axis('off')" << endl;
  file << "fig.patch.set_facecolor('#E5E5E5')" << endl;
  file << "plt.show()" << endl;
  file.close();

  system ("python plot.py");
}


void plot3d(const mat & X, const vec & y, const mat & X2, const vec & yhat, size_t axlen){

  ofstream file;
  file.open ("plot.py");

  file << "import matplotlib.pyplot as plt" << endl;
  file << "from mpl_toolkits.mplot3d import Axes3D, art3d" << endl;
  file << "from matplotlib import cbook" << endl;
  file << "from matplotlib import cm" << endl;
  file << "from matplotlib.colors import LightSource" << endl;
  file << "import numpy as np" << endl;

  // ggplot style!
  file << "plt.style.use('ggplot')" << endl;

  // Surface for the approximator
  mat xx = X2.col(0);
  xx.reshape(axlen,axlen);
  mat yy = X2.col(1);
  yy.reshape(axlen,axlen);
  mat zz = yhat;
  zz.reshape(axlen,axlen);

  file << "xx = np.zeros((" << axlen <<  ","<< axlen <<"))" << endl;
  file << "yy = np.zeros((" << axlen <<  ","<< axlen <<"))" << endl;
  file << "zz = np.zeros((" << axlen <<  ","<< axlen <<"))" << endl;

  for (size_t i=0; i < (axlen); ++i) {
    for (size_t j=0; j < (axlen); ++j) {
      file << "xx[" << i << "," << j << "] = " << xx(i,j) << endl;
      file << "yy[" << i << "," << j << "] = " << yy(i,j) << endl;
      file << "zz[" << i << "," << j << "] = " << zz(i,j) << endl;
    }
  }

  // Input data
  file << "x = np.zeros(" << X.n_rows << ")" << endl;
  file << "y = np.zeros(" << X.n_rows << ")" << endl;
  file << "z = np.zeros(" << X.n_rows << ")" << endl;

  for (size_t i=0; i < X.n_rows; ++i) {
    file << "x["<< i << "] = " << X(i,0) << endl;
    file << "y["<< i << "] = " << X(i,1) << endl;
    file << "z["<< i << "] = " << y(i) << endl;
  }

  file << "fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))" << endl;

  // Scatter
  file << "ax.scatter(x,y,z, c=z, cmap=cm.winter)" << endl;

  // Wireframe
  file << "wire = ax.plot_wireframe(xx, yy, zz, rstride=5, cstride=5)" << endl;

  // Contour
  file << "ax.contour(xx,yy,zz, zdir='z', offset=-2.0, cmap=cm.winter)" << endl;

  // Fix the colors on wireframe
  file << "# Retrive data from internal storage of plot_wireframe, then delete i" << endl;
  file << "nx, ny, _  = np.shape(wire._segments3d)      " << endl;
  file << "wire_x = np.array(wire._segments3d)[:, :, 0].ravel() " << endl;
  file << "wire_y = np.array(wire._segments3d)[:, :, 1].ravel() " << endl;
  file << "wire_z = np.array(wire._segments3d)[:, :, 2].ravel() " << endl;
  file << "wire.remove()                                        " << endl;

  file << "# create data for a LineCollection                       " << endl;
  file << "wire_x1 = np.vstack([wire_x, np.roll(wire_x, 1)])    " << endl;
  file << "wire_y1 = np.vstack([wire_y, np.roll(wire_y, 1)])    " << endl;
  file << "wire_z1 = np.vstack([wire_z, np.roll(wire_z, 1)])    " << endl;
  file << "to_delete = np.arange(0, nx*ny, ny)                  " << endl;
  file << "wire_x1 = np.delete(wire_x1, to_delete, axis=1)      " << endl;
  file << "wire_y1 = np.delete(wire_y1, to_delete, axis=1)      " << endl;
  file << "wire_z1 = np.delete(wire_z1, to_delete, axis=1)      " << endl;
  file << "scalars = np.delete(wire_z, to_delete)               " << endl;
  file << "                                                     " << endl;
  file << "segs = [list(zip(xl, yl, zl)) for xl, yl, zl in zip(wire_x1.T, wire_y1.T, wire_z1.T)]" << endl;

  file << "# Plots the wireframe by a  a line3DCollection        " << endl;
  file << "my_wire = art3d.Line3DCollection(segs, cmap='winter') " << endl;
  file << "my_wire.set_array(scalars)                            " << endl;
  file << "ax.add_collection(my_wire)                            " << endl;

  file << "plt.axis('off')" << endl;
  file << "plt.show()" << endl;

  file.close();

  system ("ipython plot.py");
}


