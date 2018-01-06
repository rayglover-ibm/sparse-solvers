# Sparse Solvers &nbsp; [![Build Status](https://travis-ci.org/rayglover-ibm/sparse-solvers.svg?branch=master)](https://travis-ci.org/rayglover-ibm/sparse-solvers)
_High performance ℓ₁-minimization solvers for sparse sensing and signal recovery problems._

## Releases

__Python__ – The python binding is available as a package [on pypi](https://pypi.python.org/pypi/sparsesolvers) via pip install:

```bash
pip install sparsesolvers
```

Here is a toy example:

```python
import sparsesolvers as ss
import numpy as np

N = 10

# Create an example sensing matrix
A = np.random.normal(loc=0.025, scale=0.025, size=(N, N)) + np.identity(N)

# An incoming signal
signal = np.zeros(N)
signal[2] = 1

# Use the homotopy solver to produce sparse solution, x.
x, info = ss.Homotopy(A).solve(signal, tolerance=0.1)

# Example output: error=0.064195, sparsity=0.9, argmax=2
print("error=%f, sparsity=%f, argmax=%i" % (
    info.solution_error, 1 - np.count_nonzero(x) / np.double(N),
    np.argmax(x)))
```

## References

1. _A. Y. Yang, Z. Zhou, A. Ganesh, S. S. Sastry, and Y. Ma_ – __Fast ℓ₁-minimization Algorithms For Robust Face Recognition__ – IEEE Trans. Image Processing, vol. 22, pp. 3234–3246, Aug 2013.

2. _R. Chartrand, W. Yin_ – __Iteratively Reweighted Algorithms For Compressive Sensing__ – Acoustics Speech and Signal Processing 2008. ICASSP 2008. IEEE International Conference, pp. 3869-3872, March 2008.

3. _D. O’Leary_ – __Robust Regression Computation Using Iteratively Reweighted Least Squares__ – Society for Industrial and Applied Mathematics, 1990

<br>

## Setup, Build & Test

Sparse solvers is also a c++14 library for your own projects. The python binding is a good example of how you can incorporate the solvers in to your own c++ projects with minimal effort.

### Requirements

At a minimum, you will need:

- CMake 3.2
- A reasonably compliant C++14 compiler, e.g.:

    | Windows    | Linux                 | Mac       |
    |:----------:|:----------------------|:----------|
    | VS 2015    | gcc 5.3 / clang 3.6   | XCode 7.3 |


### Setup

First, clone the repository and its submodules:

```bash
git clone --recurse-submodules https://github.com/rayglover-ibm/sparse-solvers
cd sparse-solvers
```

### Build

Configure and build using CMake in the typical way:

```bash
mkdir build && cd build
cmake ..
cmake --build . [--config Release]
```

Run the test suite and/or (if you've enabled them) benchmarks:

```bash
ctest -VV . [--config <config>]
```

### Build – _Options_

There are a number of _sparse solvers_ specific CMake options:

| CMake option                 | Description            | Default |
|:-----------------------------|:-----------------------|:--------|
| `sparsesolvers_WITH_TESTS`   | Enable unit tests      | ON      |
| `sparsesolvers_WITH_BENCHES` | Enable benchmarks      | OFF     |
| `sparsesolvers_WITH_PYTHON`  | Enable python binding  | OFF     |

### Build – _Python Package_

To build the python package (`.whl`) you will need the relevant Python development package, such as `python-dev` for Debian/Ubuntu. For Windows/Mac I recommend [Conda](https://conda.io/miniconda.html). To build the wheel:

```bash
mkdir build-py && cd build-py
cmake -Dsparsesolvers_WITH_PYTHON=ON ..
cmake --build . --target bdist_wheel [--config Release]
```

Once the wheel has been created (usually in `build-py/bindings/python/dist`) you can install it with `pip` locally in the usual way:

```bash
pip install <path/to/sparsesolvers.whl>
```

<br>

---

__Copyright 2017 International Business Machines Corporation__

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
