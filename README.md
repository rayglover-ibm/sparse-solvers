# Sparse Solvers
_High performance L1 minimization solvers for Python and Tensorflow. (Work in progress.)_

## Releases

_(TODO)_

## Setup, Build & test

First, clone the repository and its submodules:

    > git clone --recurse-submodules https://github.com/rayglover-ibm/sparse-solvers
    > cd sparse-solvers

### Requirements

At a minimum, you will need:

- CMake 3.3
- A reasonably compliant C++14 compiler, e.g.:

    | Windows    | Linux                 | Mac       |
    |:----------:|:----------------------|:----------|
    | VS 2015    | gcc 5.3 / clang 3.6   | XCode 7.3 |

### Python Package (optional)

To build the python package you will need Python 2.7/3, along with the relevant Python development package, such as `python-dev` for Debian/Ubuntu. For Windows/Mac I recommend [Conda](https://conda.io/miniconda.html).

### Build

Build using CMake in the typical way:

```bash
> mkdir build && cd build
> cmake [-G <Generator>] [<Options>] ..
> cmake --build . --config Release
```

There are a number of CMake options available:

| CMake option               | Description            | Default |
|----------------------------|:-----------------------|:--------|
| `sparsesolvers_WITH_TESTS` | Enable unit tests      | ON      |
| `sparsesolvers_WITH_PYTHON`| Enable python binding  | OFF     |

### Test

All tests can be run via CMake:

```bash
>  ctest -C Release -VV
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
