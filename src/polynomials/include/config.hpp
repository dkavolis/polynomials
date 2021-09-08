/**
 * @file config.hpp
 * @author Daumantas Kavolis <dkavolis>
 * @brief Constant values for package
 * @date 20-Jun-2020
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

#ifndef SRC_POLYNOMIALS_CONFIG_HPP_
#define SRC_POLYNOMIALS_CONFIG_HPP_

#include <boost/multiprecision/cpp_bin_float.hpp>

#ifndef POLY_HEADER_ONLY
#  define POLY_HEADER_ONLY 1
#endif

#define DO_PRAGMA(x) _Pragma(#x)
#if defined(__GNUC__) && !defined(__clang__)
#  define GCC_DIAGNOSTIC_IGNORED(wrn) \
    DO_PRAGMA(GCC diagnostic push)    \
    DO_PRAGMA(GCC diagnostic ignored wrn)
#  define GCC_DIAGNOSTIC_POP() DO_PRAGMA(GCC diagnostic pop)
#else
#  define GCC_DIAGNOSTIC_IGNORED(wrn)
#  define GCC_DIAGNOSTIC_POP()
#endif

#if defined(__clang__)
#  define CLANG_DIAGNOSTIC_IGNORED(wrn) \
    DO_PRAGMA(clang diagnostic push)    \
    DO_PRAGMA(clang diagnostic ignored wrn)
#  define CLANG_DIAGNOSTIC_POP() DO_PRAGMA(clang diagnostic pop)
#else
#  define CLANG_DIAGNOSTIC_IGNORED(wrn)
#  define CLANG_DIAGNOSTIC_POP()
#endif

#if defined(_MSC_VER)
#  define MSVC_WARNING_DISABLE(...) __pragma(warning(push)) __pragma(warning(disable : __VA_ARGS__))
#  define MSVC_WARNING_POP() __pragma(warning(pop))
#else
#  define MSVC_WARNING_DISABLE(wrn)
#  define MSVC_WARNING_POP()
#endif

namespace poly {
using Quad = boost::multiprecision::cpp_bin_float_quad;

constexpr static unsigned QuadraturePoints = 128;
constexpr static std::size_t SmallStorageSize = 10;

template <bool check = false>
struct bounds_check : std::bool_constant<check> {};
constexpr static inline bounds_check<true> check_bounds{};
constexpr static inline bounds_check<false> no_bounds_check{};

template <bool check = false>
struct narrowing_check : std::bool_constant<check> {};
constexpr static inline narrowing_check<true> check_narrowing{};
constexpr static inline narrowing_check<false> no_narrowing_check{};

}  // namespace poly

#define POLY_TEMPLATE_RANGE(tmp)                     \
  namespace boost {                                  \
  template <class... T>                              \
  struct range_iterator<tmp<T...>> {                 \
    using type = typename tmp<T...>::iterator;       \
  };                                                 \
                                                     \
  template <class... T>                              \
  struct range_iterator<tmp<T...> const> {           \
    using type = typename tmp<T...>::const_iterator; \
  };                                                 \
                                                     \
  }  // namespace boost

#endif  // SRC_POLYNOMIALS_CONFIG_HPP_
