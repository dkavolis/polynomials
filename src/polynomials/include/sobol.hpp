/**
 * @file sobol.hpp
 * @author Daumantas Kavolis <dkavolis>
 * @brief
 * @date 30-Jun-2020
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

#ifndef SRC_POLYNOMIALS_SOBOL_HPP_
#define SRC_POLYNOMIALS_SOBOL_HPP_

#include "config.hpp"

MSVC_WARNING_DISABLE(4619)
#include <boost/container/small_vector.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <boost/range/adaptor/filtered.hpp>
#include <boost/range/adaptor/indexed.hpp>
MSVC_WARNING_POP()

#include "traits.hpp"
#include "utils.hpp"

namespace poly {

template <class Real_,
          class Index = std::conditional_t<std::is_const_v<Real_>, std::size_t const, std::size_t>>
class SobolItemView : public view<Index> {
 public:
  using Base = view<Index>;
  using IndexType = Index;
  using Real = Real_;

  SobolItemView(Index* indices, std::size_t count, Real& coefficient) noexcept
      : Base(indices, count), coefficient_(coefficient) {}

  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  operator SobolItemView<Real const, Index const>() const noexcept {
    return {this->data(), this->size(), coefficient_};
  }

  auto coefficient() const noexcept -> Real& { return coefficient_; }

  template <bool check = true, class Range,
            typename = std::enable_if_t<detail::is_range<Range>::value>>
  [[nodiscard]] auto matches_non_zeros(Range const& non_zeros,
                                       bounds_check<check> /* unused */ = check_bounds) const
      -> bool {
    if constexpr (check) {
      if (boost::size(non_zeros) != this->size())
        throw std::out_of_range("Dimensions do not match");
    }

    for (auto&& [index, is_non_zero] : non_zeros | boost::adaptors::indexed())
      if (((*this)[index] != 0) != static_cast<bool>(is_non_zero)) return false;

    return true;
  }

  template <bool check = true, class Range,
            typename = std::enable_if_t<detail::is_range<Range>::value>>
  [[nodiscard]] auto matches_zeros(Range const& zeros,
                                   bounds_check<check> /* unused */ = check_bounds) const -> bool {
    if constexpr (check) {
      if (boost::size(zeros) != this->size()) throw std::out_of_range("Dimensions do not match");
    }

    for (auto&& [index, is_zero] : zeros | boost::adaptors::indexed())
      if (((*this)[index] == 0) != static_cast<bool>(is_zero)) return false;

    return true;
  }

 private:
  Real& coefficient_;
};

template <class Real>
class SobolIterator : public boost::iterator_facade<
                          /* Derived = */ SobolIterator<Real>,
                          /* Value type = */ SobolItemView<Real>,
                          /* traversal tag = */ boost::random_access_traversal_tag,
                          /* reference = */ SobolItemView<Real>> {
 public:
  using Index = typename SobolItemView<Real>::IndexType;

  SobolIterator(Index* indices, Real* coefficients, std::size_t dimensions) noexcept
      : iterator_(indices), coefficients_(coefficients), dimensions_(dimensions) {}

 private:
  friend class boost::iterator_core_access;
  template <class>
  friend class SobolIterator;

  template <class OtherValue>
  auto equal(SobolIterator<OtherValue> const& other) const noexcept -> bool {
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
  auto distance_to(SobolIterator const& other) const noexcept -> std::ptrdiff_t {
    return other.coefficients_ - coefficients_;
  }

  auto dereference() const noexcept -> SobolItemView<Real> {
    return {iterator_, dimensions_, *coefficients_};
  }

  Index* iterator_;
  Real* coefficients_;
  std::size_t dimensions_;
};

/// \brief Sobol calculator for orthonormal PCEs. Under such assumption, inner products between
/// different polynomials are 0, same - 1.
///
/// Sudret, B. (2008). Global sensitivity analysis using polynomial chaos expansions.
/// Reliability Engineering & System Safety, 93(7), 964–979.
/// https://doi.org/10.1016/j.ress.2007.04.002
/// \tparam Real
/// \tparam Alloc
template <class Real, template <class> class Alloc = std::allocator>
class Sobol {
 private:
  using match_vector = boost::container::small_vector<bool, 128>;

 public:
  using indices_vector = std::vector<std::size_t, Alloc<std::size_t>>;
  using coefficients_vector = std::vector<Real, Alloc<Real>>;
  using View = SobolItemView<Real, std::size_t>;
  using ConstView = SobolItemView<Real const, std::size_t const>;

  using iterator = SobolIterator<Real const>;
  using const_iterator = iterator;

  Sobol(indices_vector indices, coefficients_vector coefficients, std::size_t dimensions)
      : indices_(std::move(indices)),
        coefficients_(std::move(coefficients)),
        dimensions_(dimensions) {
    calculate_mean_variance();
  }

  auto operator[](std::size_t index) noexcept -> View {
    return View{indices_.data() + index * dimensions_, dimensions_, coefficients_[index]};
  }
  auto operator[](std::size_t index) const noexcept -> ConstView {
    return ConstView{indices_.data() + index * dimensions_, dimensions_, coefficients_[index]};
  }

  auto at(std::size_t index) noexcept -> View {
    return View{indices_.data() + index * dimensions_, dimensions_, coefficients_.at(index)};
  }
  auto at(std::size_t index) const -> ConstView {
    return ConstView{indices_.data() + index * dimensions_, dimensions_, coefficients_.at(index)};
  }

  [[nodiscard]] auto size() const noexcept -> std::size_t { return coefficients_.size(); }
  [[nodiscard]] auto index_count() const noexcept -> std::size_t { return indices_.size(); }
  [[nodiscard]] auto dimensions() const noexcept -> std::size_t { return dimensions_; }
  [[nodiscard]] auto variance() const noexcept -> Real { return variance_; }
  [[nodiscard]] auto mean() const noexcept -> Real { return mean_; }
  [[nodiscard]] auto indices() const noexcept -> view<std::size_t const> {
    return {indices_.data(), indices_.size()};
  }
  [[nodiscard]] auto coefficients() const noexcept -> view<Real const> {
    return {coefficients_.data(), coefficients_.size()};
  }

  auto mutable_indices() noexcept -> view<std::size_t> {
    return {indices_.data(), indices_.size()};
  }
  auto mutable_coefficients() noexcept -> view<Real> {
    return {coefficients_.data(), coefficients_.size()};
  }
  void recalculate() { calculate_mean_variance(); }

  [[nodiscard]] auto begin() const noexcept -> SobolIterator<Real const> {
    return {indices_.data(), coefficients_.data(), dimensions_};
  }
  [[nodiscard]] auto end() const noexcept -> SobolIterator<Real const> {
    return {indices_.data() + index_count(), coefficients_.data() + size(), dimensions_};
  }
  [[nodiscard]] auto mutable_range() noexcept -> boost::iterator_range<SobolIterator<Real>> {
    using It = SobolIterator<Real>;
    It begin = {indices_.data(), coefficients_.data(), dimensions_};
    return boost::make_iterator_range_n(begin, size());
  }

  template <bool check = true>
  [[nodiscard]] auto sensitivity(std::size_t index,
                                 bounds_check<check> /* unused */ = check_bounds) const -> Real {
    if constexpr (check) {
      if (index >= dimensions_) throw std::out_of_range("Sensitivity index out of range");
    }
    match_vector non_zeros(dimensions_, false);
    non_zeros[index] = true;
    return calculate_sensitivity(non_zeros);
  }

  template <bool check = true>
  [[nodiscard]] auto sensitivity(std::pair<std::size_t, std::size_t> index,
                                 bounds_check<check> /* unused */ = check_bounds) const -> Real {
    if constexpr (check) {
      if (index.first >= dimensions_ || index.second >= dimensions_)
        throw std::out_of_range("Sensitivity index out of range");
    }
    match_vector non_zeros(dimensions_, false);
    non_zeros[index.first] = true;
    non_zeros[index.second] = true;
    return calculate_sensitivity(non_zeros);
  }

  template <bool check = true, class Range,
            typename = std::enable_if_t<detail::is_range<Range>::value>>
  [[nodiscard]] auto sensitivity(Range const& indices,
                                 bounds_check<check> /* unused */ = check_bounds) const -> Real {
    match_vector non_zeros(dimensions_, false);
    for (auto&& index : indices) {
      if constexpr (check) {
        if (index >= dimensions_) throw std::out_of_range("Sensitivity index out of range");
      }
      non_zeros[index] = true;
    }
    return calculate_sensitivity(non_zeros);
  }

  template <bool check = true>
  [[nodiscard]] auto total_sensitivity(std::size_t index,
                                       bounds_check<check> /* unused */ = check_bounds) const
      -> Real {
    if constexpr (check) {
      if (index >= dimensions_) throw std::out_of_range("Sensitivity index out of range");
    }
    return calculate_total_sensitivity(index);
  }

 private:
  // copying indices and coefficients allows to have precomputed variance and mean
  indices_vector indices_;
  coefficients_vector coefficients_;
  std::size_t dimensions_;
  Real variance_;
  Real mean_;
  Real inverse_variance_;

  void calculate_mean_variance() {
    if (coefficients_.size() * dimensions_ != indices_.size())
      throw std::out_of_range("Coefficients and indices sizes do not match");

    // mean is coefficient with all indices 0
    mean_ = 0;
    // should have a fake range that always dereferences to true
    match_vector zeros(dimensions_, true);
    for (auto&& item : *this) {
      if (item.matches_zeros(zeros, no_bounds_check)) {
        mean_ = item.coefficient();
        break;
      }
    }

    // variance is sum of squares of all the other coefficients
    variance_ = -(mean_ * mean_);
    for (auto&& coefficient : coefficients_) variance_ += coefficient * coefficient;

    inverse_variance_ = 1.0L / variance_;
  }

  [[nodiscard]] auto calculate_sensitivity(match_vector const& non_zeros) const -> Real {
    Real s = 0;
    for (auto&& item : *this) {
      if (item.matches_non_zeros(non_zeros, no_bounds_check))
        s += item.coefficient() * item.coefficient();
    }

    return s * inverse_variance_;
  }

  [[nodiscard]] auto calculate_total_sensitivity(std::size_t index) const -> Real {
    Real s = 0;
    for (auto&& item : *this) {
      if (item[index] != 0) s += item.coefficient() * item.coefficient();
    }

    return s * inverse_variance_;
  }
};
}  // namespace poly

POLY_TEMPLATE_RANGE(poly::Sobol)

#endif  // SRC_POLYNOMIALS_SOBOL_HPP_
