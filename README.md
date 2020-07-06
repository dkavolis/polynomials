# polynomials

[WIP] Polynomial python wrappers for [boost](https://boost.org) polynomials.

## Description

Exposed polynomials:

* `Legendre`
* `Laguerre`
* `Hermite`
* `Chebyshev`
* `LegendreStieltjes`

Every polynomials has additional wrappers for:

* `<Polynomial>Sequence` - polynomial value generator at a specified abscissa value `x` up to a specified `order`
* `<Polynomial>Series` - dense 1D polynomial series with mutable coefficients
* `<Polynomial>Product` - N-dimensional polynomial product
* `<Polynomial>ProductSet` - list of N-dimensional weighed polynomial products

## Building

Cmake, [pybind11](https://github.com/pybind/pybind11) and [boost](https://boost.org) are required for building. Tested with pybind11 2.4.3 and boost 1.73.0 using MSVC 19.27.28826.0 but should compile with GCC and Clang.

To build:

```bash
python setup.py build
```

Additional cmake options can be passed as

```bash
python setup.py build --cmake-options="-DCMAKE_CXX_COMPILER=/usr/bin/g++"
```

or in [setup.cfg](setup.cfg):

```bash
[build_ext]
vcpkg_dir = ~/vcpkg
vcpkg_triplet =
cxx_compiler = /usr/bin/g++
cmake_generator = Ninja
cmake-options =
```

## Installation

```bash
python setup.py install --user
```

## Note

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
