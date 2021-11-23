/**
 * @file bindings.cpp
 * @author Daumantas Kavolis <dkavolis>
 * @brief Python module implementation
 * @date 12-Jun-2020
 *
 * Copyright (c) 2020 <Daumantas Kavolis>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#include <bindings.hpp>

// NOLINTNEXTLINE
PYBIND11_MODULE(polynomials_cpp, m) {
  m.def_submodule("views", "Internal submodule for bound view types");

  // poly::bind_number<poly::Quad>(m, "Real");

  poly::bind_sobol<poly::Quad>(m, "Sobol");
  poly::bind_all_polynomials<poly::LegendreImpl<poly::Quad>>(m, "Legendre");
  poly::bind_all_polynomials<poly::LaguerreImpl<poly::Quad>>(m, "Laguerre");
  poly::bind_all_polynomials<poly::HermiteImpl<poly::Quad>>(m, "Hermite");
  poly::bind_all_polynomials<poly::ChebyshevImpl<poly::Quad>>(m, "Chebyshev");
  poly::bind_all_polynomials<poly::LegendreStieltjesImpl<poly::Quad>>(m, "LegendreStieltjes");

  // NOLINTNEXTLINE(performance-unnecessary-value-param)
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) { std::rethrow_exception(p); }
    } catch (poly::narrowing_error const& e) {
      const char* msg = e.what();
      if (std ::strlen(msg) == 0) {
        PyErr_SetString(PyExc_ValueError, "Narrowing error");
      } else {
        PyErr_SetString(PyExc_ValueError, msg);
      }
    }
  });
}
