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

#include <array>
#include <vector>

#include "config.hpp"

MSVC_WARNING_DISABLE(4619)
#include <boost/math/quadrature/gauss.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
MSVC_WARNING_POP()

#include "utils.hpp"

namespace poly {

namespace detail {
/// \brief function pointer type to get weights/abscissa
template <class Real>
using quadrature_getter = view<Real const> (*)();

template <Reflection Refl, class Container>
[[nodiscard]] auto full_vector(Container&& points, size_t expected_points [[maybe_unused]])
    -> decltype(auto) {
  using Real = std::remove_cvref_t<typename std::remove_cvref_t<Container>::value_type>;

  std::vector<Real> v = to_vector(points);
  if constexpr (Refl != Reflection::None) {
    reflect_in_place<Refl>(v, 2 * v.size() > expected_points);
  }

  return v;
}

template <class Real, template <class, unsigned> typename Impl, bool Reflect = true>
struct BaseQuadrature {
  template <unsigned Points>
  static auto weights() -> view<Real const> {
    static std::vector<Real> items =
        full_vector<Reflection::Even>(Impl<Real, Points>::weights(), Points);
    return {items.data(), items.size()};
  }

  template <unsigned Points>
  static auto abscissa() -> view<Real const> {
    static std::vector<Real> items =
        full_vector<Reflection::Odd>(Impl<Real, Points>::abscissa(), Points);
    return {items.data(), items.size()};
  }
};

/// \brief Wrapper for boost quadrature to convert compile time points count to runtime
/// \tparam Real floating point type
/// \tparam Impl Type implementing static functions weights() and abscissa()
/// \tparam N Maximum number of points in quadrature
/// \tparam Reflect Whether the values from Impl should be reflected first
template <typename Real, template <class, unsigned> typename Impl, unsigned N = QuadraturePoints,
          bool Reflect = true>
struct Quadrature : BaseQuadrature<Real, Impl, Reflect> {
  using Base = BaseQuadrature<Real, Impl, Reflect>;
  template <bool check>
  static auto weights(unsigned points, bounds_check<check> /* unused */ = no_bounds_check)
      -> view<Real const> {
    if constexpr (check)
      return weights_getters().at(points)();
    else
      return weights_getters()[points]();
  }

  template <bool check = false>
  static auto abscissa(unsigned points, bounds_check<check> /* unused */ = no_bounds_check)
      -> view<Real const> {
    if constexpr (check)
      return abscissa_getters().at(points)();
    else
      return abscissa_getters()[points]();
  }

 private:
  // using array of function pointers so that the values are only generated on first access
  using function_container = std::array<detail::quadrature_getter<Real>, N + 1> const;

  template <unsigned... I>
  static auto make_weights(std::integer_sequence<unsigned, I...> /*unused*/) {
    return function_container{&Base::template weights<I>...};
  }

  template <unsigned... I>
  static auto make_abscissa(std::integer_sequence<unsigned, I...> /*unused*/) {
    return function_container{&Base::template abscissa<I>...};
  }

  static auto weights_getters() -> function_container const& {
    static function_container functions =
        make_weights(std::make_integer_sequence<unsigned, N + 1>{});
    return functions;
  }

  static auto abscissa_getters() -> function_container const& {
    static function_container functions =
        make_abscissa(std::make_integer_sequence<unsigned, N + 1>{});
    return functions;
  }
};

template <typename R, unsigned n>
using GaussQuadrature = boost::math::quadrature::gauss<R, n>;

template <typename R, unsigned n>
using GaussKronrodQuadrature = boost::math::quadrature::gauss_kronrod<R, n>;
}  // namespace detail

template <typename Real, unsigned N = QuadraturePoints>
using GaussQuadrature = detail::Quadrature<Real, detail::GaussQuadrature, N, true>;

template <typename Real, unsigned N = QuadraturePoints>
using GaussKronrodQuadrature = detail::Quadrature<Real, detail::GaussKronrodQuadrature, N, true>;

}  // namespace poly
