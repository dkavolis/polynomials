/**
 * @file boost.cpp
 * @author Daumantas Kavolis <dkavolis>
 * @brief
 * @date 14-Jun-2020
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

#include "include/polynomials.hpp"

#if !defined(POLY_HEADER_ONLY) || !POLY_HEADER_ONLY
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#  define EXTERN_TEMPLATE(name)                                        \
    namespace poly {                                                   \
    template struct name##Impl<Quad>;                                  \
    template class PolynomialProduct<name##Polynomial<Quad>>;    \
    template class PolynomialProductSet<name##Polynomial<Quad>>; \
    } /* namespace poly */

EXTERN_TEMPLATE(Legendre);
EXTERN_TEMPLATE(LegendreStieltjes);
EXTERN_TEMPLATE(Laguerre);
EXTERN_TEMPLATE(Hermite);
EXTERN_TEMPLATE(Chebyshev);

namespace poly {
template class SobolItemView<Quad const>;
template class SobolItemView<Quad>;
template class SobolIterator<Quad const>;
template class SobolIterator<Quad>;
template class Sobol<Quad>;
}  // namespace poly

#endif
