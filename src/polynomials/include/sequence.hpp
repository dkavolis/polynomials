/**
 * @file sequence.hpp
 * @author Daumantas Kavolis <dkavolis>
 * @brief Iterable polynomial value range at constant x
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

#ifndef SRC_POLYNOMIALS_SEQUENCE_HPP_
#define SRC_POLYNOMIALS_SEQUENCE_HPP_

#include "config.hpp"

MSVC_WARNING_DISABLE(4619)
#include <boost/iterator/iterator_facade.hpp>
MSVC_WARNING_POP()

#include "polynomial.hpp"
#include "traits.hpp"

namespace poly {

namespace detail {
template <typename Impl, typename traits = PolynomialTraits<Impl>>
class PolynomialSequenceIterator
    : public boost::iterator_facade<PolynomialSequenceIterator<Impl, traits>, typename traits::Real,
                                    boost::forward_traversal_tag, typename traits::Real> {
 public:
  using Traits = traits;
  using OrderType = typename Traits::OrderType;
  using Real = typename Traits::Real;
  using PolynomialType = Polynomial<Impl, Traits>;

  static_assert(Traits::has_eval, "Polynomial cannot be evaluated");

  explicit PolynomialSequenceIterator(OrderType order, Real x) noexcept
      : polynomial_{order}, x_{x} {}

 private:
  friend class boost::iterator_core_access;
  template <typename, typename>
  friend class PolynomialSequenceIterator;

  template <class OtherImpl, class OtherTraits>
  [[nodiscard]] auto equal(PolynomialSequenceIterator<OtherImpl, OtherTraits> const& other) const
      -> bool {
    return this->polynomial_.order() == other.polynomial_.order();
  }

  void increment() { polynomial_.order(polynomial_.order() + 1); }

  [[nodiscard]] auto dereference() const -> Real {
    if constexpr (Traits::has_next) {
      OrderType order = polynomial_.order();
      if (order < 2) {
        if (order == 0) {
          t0_ = polynomial_(x_);
          return t0_;
        }
        t1_ = polynomial_(x_);
        return t1_;
      }

      std::swap(t0_, t1_);
      PolynomialType poly{narrow_cast<OrderType>(polynomial_.order() - 1)};
      t1_ = poly.next(x_, t0_, t1_);
      return t1_;
    } else {
      return polynomial_(x_);
    }
  }

  PolynomialType polynomial_;
  Real x_;

  // make them mutable since they are never exposed to make dereference const
  mutable Real t0_;
  mutable Real t1_;
};
}  // namespace detail

template <class Polynomial>
class SequenceRange {
 public:
  using Traits = typename Polynomial::Traits;
  using OrderType = typename Traits::OrderType;
  using Real = typename Traits::Real;
  using iterator = detail::PolynomialSequenceIterator<typename Polynomial::Base, Traits>;
  using const_iterator = iterator;

  SequenceRange(OrderType end, Real x) noexcept : end_(end), x_(std::move(x)) {}

  [[nodiscard]] auto begin() const noexcept -> iterator { return iterator{0, x_}; }
  [[nodiscard]] auto end() const noexcept -> iterator { return iterator{end_, x_}; }

  auto x() noexcept -> Real& { return x_; }
  [[nodiscard]] auto x() const noexcept -> Real const& { return x_; }
  auto order() noexcept -> OrderType& { return end_; }
  [[nodiscard]] auto order() const noexcept -> OrderType { return end_; }
  [[nodiscard]] auto size() const noexcept -> std::size_t { return narrow_cast<std::size_t>(end_); }

 private:
  Real x_;
  OrderType end_;
};

template <typename T>
auto polynomial_sequence(typename T::Traits::OrderType end_order, typename T::Traits::Real x)
    -> SequenceRange<T> {
  return SequenceRange<T>{end_order, x};
}

}  // namespace poly

POLY_TEMPLATE_RANGE(poly::SequenceRange)

#endif  // SRC_POLYNOMIALS_SEQUENCE_HPP_
