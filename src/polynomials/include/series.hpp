/**
 * @file series.hpp
 * @author Daumantas Kavolis <dkavolis>
 * @brief 1D polynomial series
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

#ifndef SRC_POLYNOMIALS_SERIES_HPP_
#define SRC_POLYNOMIALS_SERIES_HPP_

#include "config.hpp"

MSVC_WARNING_DISABLE(4619)
#include <boost/container/small_vector.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/size.hpp>
MSVC_WARNING_POP()

#include "polynomial.hpp"
#include "sequence.hpp"
#include "traits.hpp"

namespace poly {
template <class Poly, std::size_t N = SmallStorageSize>
class PolynomialSeries {
 public:
  using Traits = typename Poly::Traits;
  using Real = typename Traits::Real;
  using OrderType = typename Traits::OrderType;

  template <typename T>
  using small_vector = boost::container::small_vector<T, N>;

  explicit PolynomialSeries(OrderType n) { resize(n); }

  template <class Range, typename = std::enable_if_t<detail::is_range<Range>::value>>
  explicit PolynomialSeries(Range const& coefficients) {
    auto count = boost::size(coefficients);
    coefficients_.resize(count);
    polynomials_.resize(count);
    for (auto&& [index, coefficient] : coefficients | boost::adaptors::indexed()) {
      polynomials_[index].order(narrow_cast<OrderType>(index));
      coefficients_[index] = coefficient;
    }
  }

  [[nodiscard]] auto coefficients() const noexcept -> view<Real const> {
    return {coefficients_.data(), coefficients_.size()};
  }
  [[nodiscard]] auto polynomials() const noexcept -> view<Poly const> {
    return {polynomials_.data(), polynomials_.size()};
  }

  auto operator[](std::size_t index) noexcept -> Real& { return coefficients_[index]; }
  [[nodiscard]] auto operator[](std::size_t index) const noexcept -> Real {
    return coefficients_[index];
  }

  auto at(std::size_t index) noexcept -> Real& { return coefficients_.at(index); }
  [[nodiscard]] auto at(std::size_t index) const noexcept -> Real {
    return coefficients_.at(index);
  }

  template <class F, typename = std::enable_if_t<std::is_invocable_r_v<Real, F, Real>>>
  [[nodiscard]] static auto project(F const& function, OrderType order) -> PolynomialSeries {
    Poly poly{order};
    view<Real const> abscissa = poly.abscissa();
    return project(abscissa | boost::adaptors::transformed(
                                  [&function](Real const& x) { return function(x); }));
  }

  template <class Range, typename = std::enable_if_t<detail::is_range<Range>::value>>
  [[nodiscard]] static auto project(Range const& y_range) -> PolynomialSeries {
    OrderType count = narrow<OrderType>(boost::size(y_range));
    if (count == 0) return PolynomialSeries(0);
    OrderType max_order = count - 1;

    Poly poly{max_order};
    view<Real const> weights = poly.weights();
    view<Real const> abscissa = poly.abscissa();

    PolynomialSeries series(max_order);

    for (auto&& [j, y] : y_range | boost::adaptors::indexed()) {
      for (auto&& [i, f] :
           polynomial_sequence<Poly>(count, abscissa[j]) | boost::adaptors::indexed()) {
        series.coefficients_[i] += f * weights[j] * y;
      }
    }

    return series;
  }

  [[nodiscard]] auto operator()(Real x) const -> Real {
    if constexpr (Traits::has_next) {
      if (size() == 0) return 0;
      Real t0 = polynomials_[0](x);
      Real f = coefficients_[0] * t0;
      if (size() == 1) return f;
      Real t1 = polynomials_[1](x);
      for (std::size_t i = 1; i < size(); i++) {
        f += coefficients_[i] * t1;
        std::swap(t0, t1);
        t1 = polynomials_[i].next(x, t0, t1);
      }
      return f;
    } else {
      Real f = 0;
      for (auto&& [index, polynomial] : polynomials_ | boost::adaptors::indexed())
        f += coefficients_[index] * polynomial(x);
      return f;
    }
  }

  [[nodiscard]] auto size() const noexcept -> std::size_t { return polynomials_.size(); }
  void resize(OrderType new_size) {
    OrderType old_size = narrow<OrderType>(size());
    polynomials_.resize(new_size);
    coefficients_.resize(new_size, 0);

    for (OrderType i = old_size; i < new_size; i++) polynomials_[i].order(i);
  }

  template <bool check = false>
  [[nodiscard]] auto weights(bounds_check<check> c = no_bounds_check) const -> view<Real const> {
    return polynomials_[size() - 1].weights(c);
  }
  template <bool check = false>
  [[nodiscard]] auto abscissa(bounds_check<check> c = no_bounds_check) const -> view<Real const> {
    return polynomials_[size() - 1].abscissa(c);
  }

  static auto domain() noexcept -> std::pair<Real, Real> { return Poly::domain(); }

 private:
  small_vector<Real> coefficients_;
  small_vector<Poly> polynomials_;
};
}  // namespace poly

#endif  // SRC_POLYNOMIALS_SERIES_HPP_
