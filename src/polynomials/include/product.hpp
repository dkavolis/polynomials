/**
 * @file tensor.hpp
 * @author Daumantas Kavolis <dkavolis>
 * @brief Multidimensional tensor product of polynomials
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

#ifndef SRC_POLYNOMIALS_TENSOR_PRODUCT_HPP_
#define SRC_POLYNOMIALS_TENSOR_PRODUCT_HPP_

#include "config.hpp"

MSVC_WARNING_DISABLE(4619)
#include <boost/container/small_vector.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/size.hpp>
MSVC_WARNING_POP()

#include "polynomial.hpp"
#include "traits.hpp"
#include "utils.hpp"

namespace poly {

template <class Poly>
class PolynomialProductView : public view<Poly> {
 public:
  using Base = view<Poly>;
  using Base::Base;

  using Traits = typename Poly::Traits;
  using Real = typename Traits::Real;
  using OrderType = typename Traits::OrderType;

  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  operator PolynomialProductView<Poly const>() const noexcept {
    return {this->data(), this->size()};
  }

  template <bool check = true, class Range,
            typename = std::enable_if_t<detail::is_range<Range>::value>>
  [[nodiscard]] auto operator()(Range const& x,
                                bounds_check<check> /* unused */ = check_bounds) const -> Real {
    if constexpr (check) {
      if (boost::size(x) != this->size())
        throw std::out_of_range("Dimensions do not match size of polynomials");
    }

    Real y = 1.0;
    for (auto&& [index, xx] : x | boost::adaptors::indexed()) y *= (*this)[index](xx);
    return y;
  }

  template <bool check = true, class Range,
            typename = std::enable_if_t<detail::is_range<Range>::value>>
  void assign(Range const& orders, bounds_check<check> c = check_bounds) const {
    check_size(orders, c);

    for (auto&& [index, order] : orders | boost::adaptors::indexed())
      (*this)[index].order(narrow<OrderType>(order));
  }

  template <bool check = true, class Range,
            typename = std::enable_if_t<detail::is_range<Range>::value>>
  auto matches_orders(Range const& orders, bounds_check<check> c = check_bounds) const -> bool {
    return match(
        orders,
        [](Poly const& poly, auto&& order) {
          return poly.order() == narrow_cast<OrderType>(order);
        },
        c);
  }

  template <bool check = true, class Range,
            typename = std::enable_if_t<detail::is_range<Range>::value>>
  auto matches_zeros(Range const& zeros, bounds_check<check> c = check_bounds) const
      -> std::size_t {
    return match(
        zeros,
        [](Poly const& poly, auto&& is_zero) {
          return (poly.order() == 0) == static_cast<bool>(is_zero);
        },
        c);
  }

  template <bool check = true, class Range,
            typename = std::enable_if_t<detail::is_range<Range>::value>>
  auto matches_non_zeros(Range const& non_zeros, bounds_check<check> c = check_bounds) const
      -> std::size_t {
    return match(
        non_zeros,
        [](Poly const& poly, auto&& is_non_zero) {
          return (poly.order() != 0) == static_cast<bool>(is_non_zero);
        },
        c);
  }

 private:
  template <class Range, bool check, class F>
  auto match(Range const& range, F const& function, bounds_check<check> c = check_bounds) const
      -> bool {
    check_size(range, c);
    for (auto&& [index, value] : range | boost::adaptors::indexed()) {
      if (!function((*this)[index], value)) return false;
    }
    return true;
  }

  template <class Range, bool check>
  void check_size(Range const& range, bounds_check<check> /* unused */ = check_bounds) const {
    if constexpr (check) {
      if (boost::size(range) != this->size())
        throw std::out_of_range("Size of orders does not match size of polynomials");
    }
  }
};

template <typename Poly, std::size_t N = SmallStorageSize>
class PolynomialProduct {
 public:
  using Traits = typename Poly::Traits;
  using Real = typename Traits::Real;
  using OrderType = typename Traits::OrderType;
  using PolynomialType = Poly;
  using View = PolynomialProductView<PolynomialType>;
  using ConstView = PolynomialProductView<PolynomialType const>;

  template <typename T>
  using small_vector = boost::container::small_vector<T, N>;
  using polynomials_vector = small_vector<PolynomialType>;
  using iterator = typename polynomials_vector::iterator;
  using const_iterator = typename polynomials_vector::const_iterator;

  explicit PolynomialProduct(OrderType dimensions) : polynomials_{dimensions} {}

  template <class Range, typename = std::enable_if_t<detail::is_range<Range>::value>>
  explicit PolynomialProduct(Range const& orders) {
    auto count = boost::size(orders);
    polynomials_.resize(count);
    static_cast<View>(*this).assign(orders, no_bounds_check);
  }

  [[nodiscard]] auto polynomials() noexcept -> view<PolynomialType> {
    return {polynomials_.data(), polynomials_.size()};
  }
  [[nodiscard]] auto polynomials() const noexcept -> view<PolynomialType const> {
    return {polynomials_.data(), polynomials_.size()};
  }

  template <class... Args>
  void emplace_back(Args... args) {
    polynomials_.emplace_back(std::forward<Args>(args)...);
  }

  auto operator[](std::size_t index) noexcept -> PolynomialType& { return polynomials_[index]; }
  [[nodiscard]] auto operator[](std::size_t index) const noexcept -> PolynomialType const& {
    return polynomials_[index];
  }

  auto at(std::size_t index) noexcept -> PolynomialType& { return polynomials_.at(index); }
  [[nodiscard]] auto at(std::size_t index) const noexcept -> PolynomialType const& {
    return polynomials_.at(index);
  }

  template <bool check = true, class Range,
            typename = std::enable_if_t<detail::is_range<Range>::value>>
  [[nodiscard]] auto operator()(Range const& x, bounds_check<check> c = check_bounds) const
      -> Real {
    return static_cast<ConstView>(*this).operator()(x, c);
  }

  [[nodiscard]] auto dimensions() const noexcept -> std::size_t { return polynomials_.size(); }
  void dimensions(OrderType new_size) { polynomials_.resize(new_size); }
  [[nodiscard]] auto size() const noexcept -> std::size_t { return dimensions(); }
  void resize(std::size_t new_size) { polynomials_.resize(new_size); }
  void clear() { polynomials_.clear(); }

  auto begin() noexcept -> iterator { return polynomials_.begin(); }
  auto begin() const noexcept -> const_iterator { return polynomials_.begin(); }
  auto end() noexcept -> iterator { return polynomials_.end(); }
  auto end() const noexcept -> const_iterator { return polynomials_.end(); }

  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  operator View() noexcept { return {polynomials_.data(), polynomials_.size()}; }

  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  operator ConstView() const noexcept { return {polynomials_.data(), polynomials_.size()}; }

 private:
  polynomials_vector polynomials_;
};
}  // namespace poly

POLY_TEMPLATE_RANGE(poly::PolynomialProduct)

#endif  // SRC_POLYNOMIALS_TENSOR_PRODUCT_HPP_
