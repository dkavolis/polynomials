/**
 * @file traits.hpp
 * @author Daumantas Kavolis <dkavolis>
 * @brief Type traits for polynomials
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

#ifndef SRC_POLYNOMIALS_TRAITS_HPP_
#define SRC_POLYNOMIALS_TRAITS_HPP_

#include <type_traits>
#include <utility>

#include "config.hpp"

MSVC_WARNING_DISABLE(4619)
#include <boost/range/iterator.hpp>
MSVC_WARNING_POP()

namespace poly {

namespace detail {
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define POLY_MAKE_OPERATOR_MEMBER_DETECTOR(U, name) \
  template <typename T, typename = int>             \
  struct has_##name##_member : std::false_type {};  \
  template <typename T>                             \
  struct has_##name##_member<T, decltype((void)&T::U, 0)> : std::true_type {}

// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define POLY_MAKE_MEMBER_DETECTOR(U) POLY_MAKE_OPERATOR_MEMBER_DETECTOR(U, U)

POLY_MAKE_MEMBER_DETECTOR(eval);
POLY_MAKE_MEMBER_DETECTOR(prime);
POLY_MAKE_MEMBER_DETECTOR(zeros);
POLY_MAKE_MEMBER_DETECTOR(next);
POLY_MAKE_MEMBER_DETECTOR(weights);
POLY_MAKE_MEMBER_DETECTOR(abscissa);

template <typename T, typename = int>
struct is_trivial : std::false_type {};

template <typename T>
struct is_trivial<T, decltype((void)T::is_trivial, 0)> : std::bool_constant<T::is_trivial> {};

template <typename T, typename = int>
struct is_orthonormal : std::false_type {};

template <typename T>
struct is_orthonormal<T, decltype((void)T::is_orthonormal, 0)>
    : std::bool_constant<T::is_orthonormal> {};

template <typename T, typename = int>
struct is_orthogonal_impl : std::false_type {};

template <typename T>
struct is_orthogonal_impl<T, decltype((void)T::is_orthogonal, 0)>
    : std::bool_constant<T::is_orthogonal> {};

// orthonormal implies orthogonal
template <typename T>
struct is_orthogonal : std::disjunction<is_orthonormal<T>, is_orthogonal_impl<T>> {};

template <typename T, typename = void>
struct is_range : std::false_type {};

template <typename T>
struct is_range<T, std::void_t<typename boost::range_iterator<T>::type>> : std::true_type {};

}  // namespace detail

template <typename Impl>
struct PolynomialTraits {
  using Real = typename Impl::Real;
  using Storage = typename Impl::Storage;
  using OrderType = typename Impl::OrderType;

  // default member checker
  constexpr static inline bool has_eval = detail::has_eval_member<Impl>::value;
  constexpr static inline bool has_prime = detail::has_prime_member<Impl>::value;
  constexpr static inline bool has_zeros = detail::has_zeros_member<Impl>::value;
  constexpr static inline bool has_next = detail::has_next_member<Impl>::value;

  // quadrature
  constexpr static inline bool has_weights = detail::has_weights_member<Impl>::value;
  constexpr static inline bool has_abscissa = detail::has_abscissa_member<Impl>::value;
  constexpr static inline bool has_quadrature = has_weights && has_abscissa;

  // additional
  constexpr static inline bool is_trivial = detail::is_trivial<Impl>::value;
  constexpr static inline bool is_orthonormal = detail::is_orthonormal<Impl>::value;
  constexpr static inline bool is_orthogonal = detail::is_orthogonal<Impl>::value;
};
}  // namespace poly

#endif  // SRC_POLYNOMIALS_TRAITS_HPP_
