/**
 * @file boost_polynomials.hpp
 * @author Daumantas Kavolis <dkavolis>
 * @brief Wrapped implementations of boost polynomials
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

#ifndef SRC_POLYNOMIALS_BOOST_POLYNOMIALS_HPP_
#define SRC_POLYNOMIALS_BOOST_POLYNOMIALS_HPP_

#include "config.hpp"

MSVC_WARNING_DISABLE(4619)
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/chebyshev.hpp>
#include <boost/math/special_functions/hermite.hpp>
#include <boost/math/special_functions/laguerre.hpp>
#include <boost/math/special_functions/legendre.hpp>
#include <boost/math/special_functions/legendre_stieltjes.hpp>
MSVC_WARNING_POP()

#include "quadrature.hpp"

namespace poly {
namespace detail {

template <typename Integral>
struct BasicImpl {
  using OrderType = Integral;
  using Storage = Integral;

  constexpr static inline bool is_trivial = true;

  static auto make_storage(OrderType order = 0) noexcept -> Storage { return order; }
  static auto get_order(Storage const& item) noexcept -> OrderType { return item; }
  static void set_order(Storage& item, OrderType const& value) noexcept { item = value; }
};

}  // namespace detail

template <typename Real_, unsigned N = QuadraturePoints>
struct LegendreImpl : detail::BasicImpl<std::int16_t> {
  using Real = Real_;
  constexpr static inline bool is_orthonormal = true;

  static auto eval(Storage const& item, Real x) -> Real { return boost::math::legendre_p(item, x); }
  static auto prime(Storage const& item, Real x) -> Real {
    return boost::math::legendre_p_prime(item, x);
  }
  static auto zeros(Storage const& item) -> std::vector<Real> {
    using std::abs;
    auto z = boost::math::legendre_p_zeros<Real>(item);
    detail::reflect_in_place<detail::Reflection::Odd>(
        z, abs(z[0]) < 2 * std::numeric_limits<Real>::epsilon());
    return z;
  }
  static auto next(Storage const& item, Real x, Real Pl, Real Plm1) -> Real {
    return boost::math::legendre_next(item, x, Pl, Plm1);
  }

  template <bool check = false>
  static auto weights(Storage const& item, bounds_check<check> c = no_bounds_check)
      -> view<Real const> {
    return GaussQuadrature<Real, N>::weights(item + 1, c);
  }
  template <bool check = false>
  static auto abscissa(Storage const& item, bounds_check<check> c = no_bounds_check)
      -> view<Real const> {
    return GaussQuadrature<Real, N>::abscissa(item + 1, c);
  }
  static auto domain() -> std::pair<Real, Real> { return {-1, 1}; }
};

template <typename Real_>
struct LaguerreImpl : detail::BasicImpl<std::int16_t> {
  using Real = Real_;
  constexpr static inline bool is_orthonormal = true;

  static auto eval(Storage const& item, Real x) -> Real { return boost::math::laguerre(item, x); }
  static auto next(Storage const& item, Real x, Real Pl, Real Plm1) -> Real {
    return boost::math::laguerre_next(item, x, Pl, Plm1);
  }
  static auto domain() -> std::pair<Real, Real> { return {0, 1}; }
};

template <typename Real_>
struct HermiteImpl : detail::BasicImpl<std::uint16_t> {
  using Real = Real_;
  constexpr static inline bool is_orthonormal = true;

  static auto eval(Storage const& item, Real x) -> Real { return boost::math::hermite(item, x); }
  static auto next(Storage const& item, Real x, Real Pl, Real Plm1) -> Real {
    return boost::math::hermite_next(item, x, Pl, Plm1);
  }
  static auto domain() -> std::pair<Real, Real> {
    return {-std::numeric_limits<Real>::infinity(), std::numeric_limits<Real>::infinity()};
  }
};

template <typename, unsigned>
struct ChebyshevImpl;
namespace detail {
template <typename Real, unsigned N = QuadraturePoints>
struct ChebyshevQuadrature {
  static auto abscissa() -> view<Real const> {
    static std::vector<Real> v = ChebyshevImpl<Real, N>::zeros({N});
    return {v.data(), v.size()};
  }
  static auto weights() -> view<Real const> {
    static std::vector<Real> v(std::size_t(N), boost::math::constants::pi<Real>() / N);
    return {v.data(), v.size()};
  }
};
}  // namespace detail

template <typename Real, unsigned N = QuadraturePoints>
using ChebyshevQuadrature = detail::Quadrature<Real, detail::ChebyshevQuadrature, N, false>;

template <typename Real_, unsigned N = QuadraturePoints>
struct ChebyshevImpl : detail::BasicImpl<std::int16_t> {
  using Real = Real_;
  constexpr static inline bool is_orthonormal = true;

  static auto eval(Storage const& item, Real x) -> Real {
    return boost::math::chebyshev_t(item, x);
  }
  static auto prime(Storage const& item, Real x) -> Real {
    return boost::math::chebyshev_t_prime(item, x);
  }
  static auto next(Storage const& /* unused */, Real x, Real Pl, Real Plm1) -> Real {
    return boost::math::chebyshev_next(x, Pl, Plm1);
  }
  static auto zeros(Storage const& item) -> std::vector<Real> {
    using std::cos;
    std::vector<Real> values(item, 0);
    Real factor{boost::math::constants::pi<Real>() / 2.0 / item};
    auto half = item / 2;
    auto start = (item + 1) / 2;
    for (OrderType i = 0; i < half; ++i) {
      Real x = cos((2 * (start + i) + 1) * factor);
      values[start + i] = -x;
      values[half - i - 1] = x;
    }
    return values;
  }

  template <bool check = false>
  static auto weights(Storage const& item, bounds_check<check> c = no_bounds_check)
      -> view<Real const> {
    return ChebyshevQuadrature<Real, N>::weights(item + 1, c);
  }
  template <bool check = false>
  static auto abscissa(Storage const& item, bounds_check<check> c = no_bounds_check)
      -> view<Real const> {
    return ChebyshevQuadrature<Real, N>::abscissa(item + 1, c);
  }
  static auto domain() -> std::pair<Real, Real> { return {-1, 1}; }
};

template <typename Real_>
struct LegendreStieltjesImpl {
  using Real = Real_;
  using OrderType = std::int16_t;

  constexpr static inline bool is_orthonormal = true;

  class Storage {
   public:
    explicit Storage(OrderType m) : polynomial{m}, order{m} {};

   private:
    boost::math::legendre_stieltjes<Real> polynomial;
    OrderType order;

    template <class>
    friend struct LegendreStieltjesImpl;
  };

  static auto make_storage(OrderType order = 0) noexcept -> Storage { return Storage{order}; }
  static auto get_order(Storage const& item) noexcept -> OrderType { return item.order; }
  static void set_order(Storage& item, OrderType const& value) noexcept { item = Storage{value}; }

  static auto eval(Storage const& item, Real x) -> Real { return item.polynomial(x); }
  static auto prime(Storage const& item, Real x) -> Real { return item.polynomial.prime(x); }
  static auto zeros(Storage const& item) -> std::vector<Real> {
    auto vector = item.polynomial.zeros();
    detail::reflect_in_place<detail::Reflection::Odd>(vector, vector[0] == 0);
    return vector;
  }
  static auto domain() -> std::pair<Real, Real> { return {-1, 1}; }
};

template <typename Real_, unsigned N = QuadraturePoints>
struct GaussKronrodImpl : detail::BasicImpl<std::int16_t> {
  using Real = Real_;

  constexpr static inline bool is_orthonormal = true;

  template <bool check = false>
  static auto weights(Storage const& item, bounds_check<check> c = no_bounds_check)
      -> view<Real const> {
    return GaussKronrodQuadrature<Real, N>::weights(item, c);
  }
  template <bool check = false>
  static auto abscissa(Storage const& item, bounds_check<check> c = no_bounds_check)
      -> view<Real const> {
    return GaussKronrodQuadrature<Real, N>::abscissa(item, c);
  }
  static auto domain() -> std::pair<Real, Real> { return {-1, 1}; }
};

}  // namespace poly

#endif  // SRC_POLYNOMIALS_BOOST_POLYNOMIALS_HPP_
