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

#include <optional>
#include <vector>

#include "config.hpp"

MSVC_WARNING_DISABLE(4619)
#include <boost/iterator/iterator_facade.hpp>
#include <boost/range/adaptor/indexed.hpp>
#include <boost/range/adaptor/reversed.hpp>
#include <boost/range/adaptor/transformed.hpp>
MSVC_WARNING_POP()

#include "product.hpp"
#include "sobol.hpp"
#include "utils.hpp"

namespace poly {

template <class Poly>
class PolynomialProductSetView : public PolynomialProductView<Poly> {
 public:
  using Base = PolynomialProductView<Poly>;
  using Traits = typename Base::Traits;
  using Real = typename Base::Real;
  using OrderType = typename Base::OrderType;
  // use same constness as polynomials
  using Coefficient = std::conditional_t<std::is_const_v<Poly>, Real const, Real>;

  PolynomialProductSetView(Poly* iterator, std::size_t dimensions,
                           Coefficient& coefficient) noexcept
      : Base(iterator, dimensions), coefficient_{coefficient} {}

  auto coefficient() const noexcept -> Coefficient& { return coefficient_; }

  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  operator PolynomialProductSetView<Poly const>() const noexcept {
    return {this->data(), this->size(), coefficient_};
  }

 private:
  Coefficient& coefficient_;
};

template <class Poly>
class PolynomialProductSetIterator;

template <class Poly, template <class> class Allocator = std::allocator>
class PolynomialProductSet {
 public:
  using Traits = typename Poly::Traits;
  using Real = typename Traits::Real;
  using OrderType = typename Traits::OrderType;
  using PolynomialType = Poly;
  using View = PolynomialProductSetView<Poly>;
  using ConstView = PolynomialProductSetView<Poly const>;

  using iterator = PolynomialProductSetIterator<Poly>;
  using const_iterator = PolynomialProductSetIterator<Poly const>;

  explicit PolynomialProductSet(
      std::size_t dimensions, Allocator<Poly> const& polynomial_allocator = Allocator<Poly>(),
      Allocator<Real> const& coefficient_allocator = Allocator<Real>()) noexcept
      : polynomials_{polynomial_allocator},
        coefficients_{coefficient_allocator},
        dimensions_{dimensions} {}
  PolynomialProductSet(std::size_t size, std::size_t dimensions,
                       Allocator<Poly> const& polynomial_allocator = Allocator<Poly>(),
                       Allocator<Real> const& coefficient_allocator = Allocator<Real>())
      : polynomials_(size * dimensions, polynomial_allocator),
        coefficients_(size, coefficient_allocator),
        dimensions_{dimensions} {}

  template <class Integral>
  static auto full_set(view<Integral> const& orders) -> PolynomialProductSet {
    return full_set(static_cast<strided_view<Integral>>(orders));
  }

  template <class Integral>
  static auto full_set(strided_view<Integral> const& orders) -> PolynomialProductSet {
    std::size_t count = 1;
    std::size_t const dimensions = orders.size();
    for (Integral const& order : orders) { count *= order; }
    PolynomialProductSet set{count, orders.size()};
    for (std::size_t i = 1; i < count; ++i) {
      // next index set is a copy of the last one + 1 to the last index
      auto&& poly = set[i];
      poly.assign(set[i - 1] | boost::adaptors::transformed(
                                   [](auto&& polynomial) { return polynomial.order(); }));
      poly[dimensions - 1].order(poly[dimensions - 1].order() + 1);

      // now make sure all indices in the set are less than orders
      // doing in reverse because that is the order that index set is incremented in
      // no need to iterate over 0th index
      for (std::size_t j = dimensions - 1; j > 0; --j) {
        if (narrow_cast<Integral>(poly[j].order()) == orders[j]) {
          poly[j].order(0);
          poly[j - 1].order(poly[j - 1].order() + 1);
          continue;
        }

        break;
      }
    }

    return set;
  }

  template <bool check = false, class Range,
            typename = std::enable_if_t<detail::is_range<Range>::value>>
  void assign(Range const& orders, std::optional<std::size_t> const& dimensions = std::nullopt,
              narrowing_check<check> c = no_narrowing_check) {
    std::size_t dims = dimensions.value_or(dimensions_);

    std::size_t count = boost::size(orders);
    if (count % dims != 0) {
      throw std::length_error("Number of orders doesn't fill the last polynomial");
    }
    std::size_t length = count / dims;

    coefficients_.resize(length);
    polynomials_.resize(count);
    fill_orders(orders, c);
  }

  template <class Range, typename = std::enable_if_t<detail::is_range<Range>::value>>
  void assign_coefficients(Range const& coefficients) {
    if (boost::size(coefficients) != size())
      throw std::length_error("Invalid number of coefficients");

    for (auto&& [index, coefficient] : coefficients | boost::adaptors::indexed())
      coefficients_[index] = static_cast<Real>(coefficient);
  }

  void merge_repeated() {
    // use hashes for ~O(n) complexity but requires unordered_map, small sizes may be faster by
    // brute force
    std::unordered_map<std::size_t, std::size_t> hash_index_map(size());

    for (std::size_t i = 0; i < this->size(); ++i) {
      std::size_t hash = detail::hash_range(
          (*this)[i] | boost::adaptors::transformed(
                           [](auto&& polynomial) -> OrderType { return polynomial.order(); }));

      auto pair = hash_index_map.emplace(hash, i);

      // continue if hash was not already in map
      if (pair.second) continue;

      // assuming no hash collisions for simplicity merge coefficients and remove the duplicate
      std::size_t index = (*pair.first).second;
      coefficients_[index] += coefficients_[i];
      erase(i, no_bounds_check);
      --i;
    }
  }

  template <bool check = true>
  void erase(std::size_t index, bounds_check<check> /* unused */ = check_bounds) {
    if constexpr (check) {
      if (index >= size()) throw std::out_of_range("Index out of range");
    }

    coefficients_.erase(coefficients_.begin() + index);
    std::size_t offset = to_index(index);
    auto begin = polynomials_.begin() + offset;
    polynomials_.erase(begin, begin + dimensions_);
  }

  auto operator[](std::size_t index) noexcept -> View {
    std::size_t j = to_index(index);
    return {&polynomials_[j], dimensions_, coefficients_[index]};
  }
  auto operator[](std::size_t index) const noexcept -> ConstView {
    std::size_t j = to_index(index);
    return {&polynomials_[j], dimensions_, coefficients_[index]};
  }
  auto at(std::size_t index) -> View {
    std::size_t j = to_index(index);
    return {&polynomials_.at(j), dimensions_, coefficients_.at(index)};
  }
  [[nodiscard]] auto at(std::size_t index) const -> ConstView {
    std::size_t j = to_index(index);
    return {&polynomials_.at(j), dimensions_, coefficients_.at(index)};
  }

  template <bool check = true, class Integral>
  [[nodiscard]] auto index_of(view<Integral> orders, bounds_check<check> c = check_bounds) const
      -> std::optional<std::size_t> {
    return index_of(static_cast<strided_view<Integral>>(orders), c);
  }

  template <bool check = true, class Integral>
  [[nodiscard]] auto index_of(strided_view<Integral> orders,
                              bounds_check<check> /* unused */ = check_bounds) const
      -> std::optional<std::size_t> {
    if constexpr (check) {
      if (orders.size() != dimensions_) throw std::out_of_range("Orders/Dimensions do not match");
    }

    for (auto&& [index, product] : *this | boost::adaptors::indexed()) {
      if (product.matches_orders(orders, no_bounds_check)) return index;
    }
    return std::nullopt;
  }

  template <bool check = true, class F>
  auto operator()(view<F> x, bounds_check<check> c = check_bounds) const -> Real {
    return (*this)(static_cast<strided_view<F>>(x), c);
  }

  // use tag dispatch to force check dimensions or not at compile time depending on usage
  template <bool check = true, class F>
  auto operator()(strided_view<F> x, bounds_check<check> /* unused */ = check_bounds) const
      -> Real {
    if constexpr (check) {
      if (x.size() != dimensions_) throw std::out_of_range("Dimensions do not match");
    }

    Real y = 0;
    for (auto&& tensor : *this)
      if (tensor.coefficient() != 0)
        y += tensor.coefficient() *
             tensor(x, no_bounds_check);  // no need to check dimensions every time
    return y;
  }

  [[nodiscard]] auto dimensions() const noexcept -> std::size_t { return dimensions_; }
  void dimensions(std::size_t new_dimensions) {
    if (new_dimensions == 0) {
      clear();
      dimensions_ = 0;
      return;
    }

    // need a temp vector to copy polynomials to, this is not a simple copy that vector would do
    std::vector<Poly, Allocator<Poly>> temp(new_dimensions * size(), polynomials_.get_allocator());

    // only iterate over useful dimensions (existing and included)
    std::size_t dims = std::min(new_dimensions, dimensions_);
    for (std::size_t dim = 0; dim < dims; ++dim) {
      for (std::size_t i = dim, j = dim; i < polynomials_.size(); i += dimensions_) {
        temp[j] = std::move(polynomials_[i]);
        j += new_dimensions;
      }
    }

    std::swap(temp, polynomials_);
    dimensions_ = new_dimensions;
  }

  void clear() {
    polynomials_.clear();
    coefficients_.clear();
  }

  [[nodiscard]] auto size() const noexcept -> std::size_t { return coefficients_.size(); }
  [[nodiscard]] auto index_count() const noexcept -> std::size_t { return polynomials_.size(); }
  void resize(std::size_t new_size) {
    polynomials_.resize(new_size * dimensions_);
    coefficients_.resize(new_size);
  }

  auto begin() noexcept -> iterator {
    return {polynomials_.data(), coefficients_.data(), dimensions_};
  }
  auto begin() const noexcept -> const_iterator {
    return {polynomials_.data(), coefficients_.data(), dimensions_};
  }

  auto end() noexcept -> iterator {
    return {polynomials_.data() + index_count(), coefficients_.data() + size(), dimensions_};
  }
  auto end() const noexcept -> const_iterator {
    return {polynomials_.data() + index_count(), coefficients_.data() + size(), dimensions_};
  }

  template <template <class> class Alloc = Allocator>
  [[nodiscard]] auto sobol() const -> Sobol<Real, Alloc> {
    std::vector<std::size_t, Alloc<std::size_t>> indices{};
    indices.reserve(polynomials_.size());
    std::transform(polynomials_.begin(), polynomials_.end(), std::back_inserter(indices),
                   [](auto&& polynomial) { return polynomial.order(); });
    return Sobol<Real, Alloc>{std::move(indices), coefficients_, dimensions_};
  }

  auto polynomials() noexcept -> view<Poly> { return {polynomials_.data(), polynomials_.size()}; }
  auto polynomials() const noexcept -> view<Poly const> {
    return {polynomials_.data(), polynomials_.size()};
  }

  auto coefficients() noexcept -> view<Real> {
    return {coefficients_.data(), coefficients_.size()};
  }
  auto coefficients() const noexcept -> view<Real const> {
    return {coefficients_.data(), coefficients_.size()};
  }

 private:
  [[nodiscard]] auto to_index(std::size_t index) const noexcept -> std::size_t {
    return index * dimensions_;
  }

  template <bool check, class Range>
  void fill_orders(Range const& orders, narrowing_check<check> /* unused */ = no_narrowing_check) {
    for (auto&& [index, order] : orders | boost::adaptors::indexed())
      if constexpr (check) {
        polynomials_[index].order(narrow<OrderType>(order));
      } else {
        polynomials_[index].order(narrow_cast<OrderType>(order));
      }
  }

  std::vector<Poly, Allocator<Poly>> polynomials_;
  std::vector<Real, Allocator<Real>> coefficients_;
  std::size_t dimensions_;
};

template <class Poly>
class PolynomialProductSetIterator : public boost::iterator_facade<
                                         /* Derived = */ PolynomialProductSetIterator<Poly>,
                                         /* Value type = */ PolynomialProductSetView<Poly>,
                                         /* traversal tag = */ boost::random_access_traversal_tag,
                                         /* reference = */ PolynomialProductSetView<Poly>> {
 public:
  using Real = typename Poly::Real;
  using Coefficient = std::conditional_t<std::is_const_v<Poly>, Real const, Real>;

  PolynomialProductSetIterator(Poly* iterator, Coefficient* coefficients,
                               std::size_t dimensions) noexcept
      : iterator_(iterator), coefficients_(coefficients), dimensions_(dimensions) {}

 private:
  friend class boost::iterator_core_access;
  template <class>
  friend class PolynomialProductSetIterator;

  template <class OtherValue>
  auto equal(PolynomialProductSetIterator<OtherValue> const& other) const noexcept -> bool {
    return this->coefficients_ == other.coefficients_;
  }

  void increment() noexcept {
    iterator_ += dimensions_;
    ++coefficients_;
  }
  void decrement() noexcept {
    iterator_ -= dimensions_;
    --coefficients_;
  }

  void advance(std::ptrdiff_t n) noexcept {
    iterator_ += n * dimensions_;
    coefficients_ += n;
  }
  auto distance_to(PolynomialProductSetIterator const& other) const noexcept -> std::ptrdiff_t {
    return other.coefficients_ - coefficients_;
  }

  auto dereference() const noexcept -> PolynomialProductSetView<Poly> {
    return {iterator_, dimensions_, *coefficients_};
  }

  Poly* iterator_;
  Coefficient* coefficients_;
  std::size_t dimensions_;
};

}  // namespace poly

POLY_TEMPLATE_RANGE(poly::PolynomialProductSet)
