CXX := clang++
CXXFLAGS := -DNDEBUG -O2 -std=c++11
DEBUGFLAGS := -Wall -g -std=c++11

# Header directories
PROJECT_INCLUDE_DIR := ./
ARMADILLO_INCLUDE_DIR := /usr/local/Cellar/armadillo/6.400.3_1/include
INCLUDE_DIRS := -I$(PROJECT_INCLUDE_DIR) -I$(ARMADILLO_INCLUDE_DIR)

# Libraries to link and library directories
LDLIBS := -larmadillo
ARMADILLO_LIB_DIR := /usr/local/Cellar/armadillo/6.400.3_1/lib
LDFLAGS := -L$(ARMADILLO_LIB_DIR)

# This is a header only library
DEPS := cpp-rbf/rbf.hpp examples/utils.hpp

all: testrbf

testrbf: $(DEPS)
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) $(LDFLAGS) -o testrbf examples/testrbf.cpp $(LDLIBS)

debug: $(DEPS)
	$(CXX) $(DEBUGFLAGS) $(INCLUDE_DIRS) $(LDFLAGS) -o testrbf examples/testrbf.cpp $(LDLIBS)

clean:
	rm testrbf
	rm -rf testrbf.dSYM
