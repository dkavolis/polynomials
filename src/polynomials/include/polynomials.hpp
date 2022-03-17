/**
 * Copyright (c) 2022 <Daumantas Kavolis>
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

#pragma once

#include "boost_polynomials.hpp"
#include "config.hpp"
#include "polynomial.hpp"
#include "product.hpp"
#include "product_set.hpp"
#include "quadrature.hpp"
#include "sequence.hpp"
#include "series.hpp"
#include "sobol.hpp"
#include "traits.hpp"
#include "utils.hpp"

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define POLY_POLYNOMIALS_DECLARE(name, nname)                                                 \
  namespace poly {                                                                            \
  template <typename T>                                                                       \
  using name##Polynomial = Polynomial<name##Impl<T>>;                                         \
                                                                                              \
  template <typename T>                                                                       \
  using name##PolynomialSeries = PolynomialSeries<name##Impl<T>>;                             \
                                                                                              \
  using name = name##Polynomial<Quad>;                                                        \
  using name##Series = name##PolynomialSeries<Quad>;                                          \
                                                                                              \
  template <class Real>                                                                       \
  using name##PolynomialProduct = PolynomialProduct<name##Polynomial<Real>>;                  \
  template <class Real, template <class> class Allocator = std::allocator>                    \
  using name##PolynomialProductSet = PolynomialProductSet<name##Polynomial<Real>, Allocator>; \
  using name##Product = name##PolynomialProduct<Quad>;                                        \
  using name##ProductSet = name##PolynomialProductSet<Quad>;                                  \
                                                                                              \
  template <class Real>                                                                       \
  auto nname##_sequence(typename name##Polynomial<Real>::Traits::OrderType end_order, Real x) \
      -> SequenceRange<name##Polynomial<Real>> {                                              \
    return polynomial_sequence<name##Polynomial<Real>>(end_order, x);                         \
  }                                                                                           \
  } /* namespace poly */

POLY_POLYNOMIALS_DECLARE(Legendre, legendre)
POLY_POLYNOMIALS_DECLARE(LegendreStieltjes, legendre_stieltjes)
POLY_POLYNOMIALS_DECLARE(Laguerre, laguerre)
POLY_POLYNOMIALS_DECLARE(Hermite, hermite)
POLY_POLYNOMIALS_DECLARE(Chebyshev, chebyshev)
#undef POLY_POLYNOMIALS_DECLARE

#if !defined(POLY_HEADER_ONLY) || !POLY_HEADER_ONLY
namespace poly {

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#  define POLY_EXTERN_POLYNOMIAL(name)                               \
    extern template struct name##Impl<Quad>;                         \
    extern template class PolynomialProduct<name##Polynomial<Quad>>; \
    extern template class PolynomialProductSet<name##Polynomial<Quad>>;

POLY_EXTERN_POLYNOMIAL(Legendre)
POLY_EXTERN_POLYNOMIAL(LegendreStieltjes)
POLY_EXTERN_POLYNOMIAL(Laguerre)
POLY_EXTERN_POLYNOMIAL(Hermite)
POLY_EXTERN_POLYNOMIAL(Chebyshev)

#  undef POLY_EXTERN_POLYNOMIAL

extern template class SobolItemView<Quad const>;
extern template class SobolItemView<Quad>;
extern template class SobolIterator<Quad const>;
extern template class SobolIterator<Quad>;
extern template class Sobol<Quad>;
}  // namespace poly
#endif
