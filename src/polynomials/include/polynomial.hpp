/**
 * @file polynomial.hpp
 * @author Daumantas Kavolis <dkavolis>
 * @brief Polynomial wrapper for concrete implementations
 * @date 21-Jun-2020
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

#ifndef SRC_POLYNOMIALS_POLYNOMIAL_HPP_
#define SRC_POLYNOMIALS_POLYNOMIAL_HPP_

#include "traits.hpp"

namespace poly {
template <typename Impl, typename traits = PolynomialTraits<Impl>>
class Polynomial {
 public:
  using Base = Impl;
  using Traits = traits;
  using OrderType = typename Traits::OrderType;
  using Real = typename Traits::Real;

 private:
  using Storage = typename Traits::Storage;

 public:
  Polynomial() = default;
  explicit Polynomial(OrderType order) : data_{Impl::make_storage(order)} {}

  [[nodiscard]] auto operator()(Real x) const -> Real {
    static_assert(Traits::has_eval, "Polynomial does not support evaluation");
    return Impl::eval(data_, x);
  }
  [[nodiscard]] auto prime(Real x) const -> Real {
    static_assert(Traits::has_prime, "Polynomial does not support derivatives");
    return Impl::prime(data_, x);
  }
  [[nodiscard]] auto zeros() const -> std::vector<Real> {
    static_assert(Traits::has_zeros, "Polynomial does not support zeros");
    return Impl::zeros(data_);
  }
  [[nodiscard]] auto next(Real x, Real Pl, Real Plm1) const -> Real {
    static_assert(Traits::has_next, "Polynomial does not support next");
    return Impl::next(data_, x, Pl, Plm1);
  }

  template <bool check = false>
  [[nodiscard]] auto weights(bounds_check<check> c = no_bounds_check) const
      -> view<Real const> {
    static_assert(Traits::has_weights, "Polynomial does not support weights");
    return Impl::weights(data_, c);
  }
  template <bool check = false>
  [[nodiscard]] auto abscissa(bounds_check<check> c = no_bounds_check) const
      -> view<Real const> {
    static_assert(Traits::has_abscissa, "Polynomial does not support abscissa");
    return Impl::abscissa(data_, c);
  }

  static auto domain() noexcept -> std::pair<Real, Real> { return Impl::domain(); }

  template <class F>
  auto integrate(F const& function) const -> Real {
    static_assert(Traits::has_quadrature, "Polynomial does not support quadrature");
    auto&& a = abscissa();
    auto&& w = weights();
    Real result = 0;
    for (std::size_t i = 0; i < a.size(); ++i) result += w[i] * function(a[i]);
    return result;
  }

  auto order() const noexcept -> OrderType { return Impl::get_order(data_); }
  void order(OrderType value) noexcept { Impl::set_order(data_, value); }

 private:
  Storage data_{Impl::make_storage()};
};
}  // namespace poly

#endif  // SRC_POLYNOMIALS_POLYNOMIAL_HPP_
