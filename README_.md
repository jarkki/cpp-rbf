**cpp-rbf** is linearly parameterized Gaussian Radial Basis Function approximator written in C++.

Example approximation to sin with Gaussian noise (sd=0.3):

![](figures/2d_ex.png)

Example approximation to two-dimensional sin with Gaussian noise (sd=0.3) with 300 uniform random samples:

![](figures/3d_ex2.png)

# Introduction

# Installation
`cpp-rbf` is a header-only library and uses some c++11 features.

## Dependencies
* [Armadillo](http://arma.sourceforge.net) for linear algebra and random number generation.

For plotting the example plots, you also need

* Python + numpy + [matplotlib](http://matplotlib.org/)

## Compilation
To run the example [examples/testrbf.cpp](examples/testrbf.cpp) that approximates 2d and 3d sin(),

1. Edit the [makefile](Makefile) to include the location of Armadillo headers and library.
2. Run `make` in the project root directory.

# Example
See [examples/testrbf.cpp](examples/testrbf.cpp).

# License

**mc-control** is made available under the terms of the GPLv3.

See the LICENSE file that accompanies this distribution for the full text of the license.
