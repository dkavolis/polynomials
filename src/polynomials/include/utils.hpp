/**
 * @file utils.hpp
 * @author Daumantas Kavolis <dkavolis>
 * @brief Common utility functions
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

#ifndef SRC_POLYNOMIALS_UTILS_HPP_
#define SRC_POLYNOMIALS_UTILS_HPP_

#include <array>
#include <stdexcept>
#include <utility>
#include <vector>

#include "config.hpp"

MSVC_WARNING_DISABLE(4619)
#include <boost/range/adaptor/indexed.hpp>
#include <boost/tuple/tuple.hpp>
MSVC_WARNING_POP()

#include "traits.hpp"

// apparently boost tuples cannot be used with structured bindings by default...
// https://stackoverflow.com/a/55594747/13262469
namespace std {
template <typename T, typename U>
struct tuple_size<boost::tuples::cons<T, U>> : boost::tuples::length<boost::tuples::cons<T, U>> {};

template <size_t I, typename T, typename U>
struct tuple_element<I, boost::tuples::cons<T, U>>
    : boost::tuples::element<I, boost::tuples::cons<T, U>> {};

template <typename T, typename Indexable>
struct tuple_size<boost::range::index_value<T, Indexable>>
    : std::integral_constant<std::size_t, 2> {};

template <typename T, typename Indexable>
struct tuple_element<0, boost::range::index_value<T, Indexable>> {
  using type = Indexable;
};

template <typename T, typename Indexable>
struct tuple_element<1, boost::range::index_value<T, Indexable>> {
  using type = T;
};
}  // namespace std

namespace poly {

// borrow narrow and narrow_cast from ms-gsl without having to rely on it
// narrow_cast(): a searchable way to do narrowing casts of values
template <class T, class U>
constexpr auto narrow_cast(U&& u) noexcept -> T {
  return static_cast<T>(std::forward<U>(u));
}

struct narrowing_error : public std::exception {};

namespace detail {
template <class T, class U>
struct is_same_signedness
    : public std::integral_constant<bool, std::is_signed<T>::value == std::is_signed<U>::value> {};
}  // namespace detail

// narrow() : a checked version of narrow_cast() that throws if the cast changed the value
template <class T, class U>
constexpr auto narrow(U u) noexcept(false) -> T {
  T t = narrow_cast<T>(u);
  // if casting to larger type or T == U, skip checks
  if constexpr (sizeof(T) <= sizeof(U) && !std::is_same_v<T, U>) {
    if (static_cast<U>(t) != u) { throw narrowing_error{}; }
    if constexpr (!detail::is_same_signedness<T, U>::value) {
      if ((t < T{}) != (u < U{})) { throw narrowing_error{}; }
    }
  }
  return t;
}

template <class>
class strided_view;
template <class>
class view;

namespace detail {
template <typename T>
auto byte_offset(T* ptr, std::ptrdiff_t offset) noexcept -> T* {
  constexpr static bool is_const = std::is_const_v<T>;
  using cast_t = std::conditional_t<is_const, unsigned char const, unsigned char>;
  return reinterpret_cast<T*>(reinterpret_cast<cast_t*>(ptr) + offset);
}

template <class T>
auto strided_distance(T* begin, T* end, std::ptrdiff_t stride) noexcept -> std::ptrdiff_t {
  return (reinterpret_cast<unsigned char const*>(end) -
          reinterpret_cast<unsigned char const*>(begin)) /
         stride;
}

template <typename T>
class strided_iterator : public boost::iterator_facade<strided_iterator<T>, T,
                                                       boost::random_access_traversal_tag, T&> {
 public:
  strided_iterator(T* iterator, std::ptrdiff_t stride) noexcept
      : iterator_(iterator), stride_(stride) {}

 private:
  friend class boost::iterator_core_access;
  template <class>
  friend class py_object_iterator;

  template <class OtherValue>
  auto equal(strided_iterator<OtherValue> const& other) const -> bool {
    return this->iterator_ == other.iterator_;
  }

  void increment() { iterator_ = byte_offset(iterator_, stride_); }
  void decrement() { iterator_ = byte_offset(iterator_, -stride_); }

  void advance(std::ptrdiff_t n) noexcept { iterator_ = byte_offset(iterator_, n * stride_); }
  auto distance_to(strided_iterator const& other) const noexcept -> std::ptrdiff_t {
    return strided_distance(iterator_, other.iterator_, stride_);
  }

  auto dereference() const -> T& { return *iterator_; }

  T* iterator_;
  std::ptrdiff_t stride_;
};

template <class T>
auto access_size_member(strided_view<T>& view) -> std::size_t& {
  return view.size_;
}

template <class T>
auto access_size_member(view<T>& view) -> std::size_t& {
  return view.size_;
}

template <class T>
auto access_size_member(strided_view<T> const& view) -> std::size_t const& {
  return view.size_;
}

template <class T>
auto access_size_member(view<T> const& view) -> std::size_t const& {
  return view.size_;
}

template <class T>
auto access_stride_member(strided_view<T>& view) -> std::ptrdiff_t& {
  return view.stride_;
}

template <class T>
auto access_stride_member(strided_view<T> const& view) -> std::ptrdiff_t const& {
  return view.stride_;
}

}  // namespace detail

template <class T>
class strided_view {
 public:
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using iterator = detail::strided_iterator<T>;

  strided_view(T* data, std::size_t size, std::ptrdiff_t stride) noexcept
      : data_(data), size_(size), stride_(stride) {}
  strided_view(T* begin, T* end, std::ptrdiff_t stride)
      : data_(begin),
        size_(narrow<std::size_t>(detail::strided_distance(begin, end, stride))),
        stride_(stride) {}

  auto operator[](std::size_t index) const noexcept -> reference {
    return *detail::byte_offset(data_, index * stride_);
  }
  auto at(std::size_t index) const -> reference {
    if (index >= size_) throw std::out_of_range{"Index out of bounds"};
    return (*this)[index];
  }

  auto begin() const noexcept -> iterator { return {data_, stride_}; }
  auto end() const noexcept -> iterator { return {&(*this)[size_], stride_}; }

  [[nodiscard]] auto size() const noexcept -> std::size_t { return size_; }
  [[nodiscard]] auto stride() const noexcept -> std::ptrdiff_t { return stride_; }
  auto data() const noexcept -> pointer { return data_; }

  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  operator strided_view<T const>() const noexcept { return {data_, size_, stride_}; }

 private:
  pointer data_;
  std::size_t size_;
  std::ptrdiff_t stride_;

  template <class U>
  friend auto detail::access_size_member(strided_view<U>&) -> std::size_t&;
  template <class U>
  friend auto detail::access_size_member(strided_view<U> const&) -> std::size_t const&;
  template <class U>
  friend auto detail::access_stride_member(strided_view<U>&) -> std::ptrdiff_t&;
  template <class U>
  friend auto detail::access_stride_member(strided_view<U> const&) -> std::ptrdiff_t const&;
};

template <class T>
strided_view(view<T>) -> strided_view<T>;

template <class T>
class view {
 public:
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using iterator = pointer;

  view(T* data, std::size_t size) noexcept : data_(data), size_(size) {}
  view(T* begin, T* end) : data_(begin), size_(narrow<std::size_t>(end - begin)) {}

  template <class Alloc>
  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  view(std::vector<T, Alloc> const& vector) noexcept : data_(vector.data()), size_(vector.size()) {}

  template <std::size_t N>
  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  view(std::array<T, N> const& array) noexcept : data_(array.data()), size_(array.size()) {}

  auto operator[](std::size_t index) const noexcept -> reference { return data_[index]; }
  auto at(std::size_t index) const -> reference {
    if (index >= size_) throw std::out_of_range{"Index out of bounds"};
    return data_[index];
  }

  auto begin() const noexcept -> iterator { return data_; }
  auto end() const noexcept -> iterator { return data_ + size_; }

  [[nodiscard]] auto size() const noexcept -> std::size_t { return size_; }
  auto data() const noexcept -> pointer { return data_; }

  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  operator view<T const>() const noexcept { return {data_, size_}; }

  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  operator strided_view<T>() const noexcept {
    return {data_, size_, narrow_cast<std::ptrdiff_t>(sizeof(T))};
  }

  //  // NOLINTNEXTLINE(hicpp-explicit-conversions)
  //  operator strided_view<T const>() const noexcept {
  //    return {data_, size_, narrow_cast<std::ptrdiff_t>(sizeof(T))};
  //  }

 private:
  pointer data_;
  std::size_t size_;

  template <class U>
  friend auto detail::access_size_member(view<U>&) -> std::size_t&;
  template <class U>
  friend auto detail::access_size_member(view<U> const&) -> std::size_t const&;
};

template <class T, class Alloc>
view(std::vector<T, Alloc>) -> view<T>;
template <class T, std::size_t N>
view(std::array<T, N>) -> view<T>;

namespace detail {

template <class Range, typename = std::enable_if_t<detail::is_range<Range>::value>>
auto hash_range(Range const& range) -> size_t {
  std::size_t seed = 0;
  for (auto&& val : range) { boost::hash_combine(seed, val); }
  return seed;
}

enum struct Reflection { Odd, Even };

/// \brief boost skips -ve zeros in their polynomials so this function adds the missing values
/// \tparam type Reflection type: even/odd
/// \tparam T floating point type
/// \param vector value vector assuming the values are sorted and +ve
/// \param has_zero whether to treat value at [0] index as being at x=0
template <Reflection type, class T>
void reflect_in_place(std::vector<T>& vector, bool has_zero) {
  std::size_t count = vector.size();
  std::size_t total = 2 * count;
  if (has_zero) total -= 1;
  vector.resize(total);

  // shift values
  int shift = narrow_cast<int>(total - count);
  for (int i = narrow_cast<int>(count - 1); i >= 0; --i) { vector[i + shift] = vector[i]; }

  // reflect values
  for (int i = 0; i < count; ++i) {
    if constexpr (type == Reflection::Odd) {
      vector[i] = -vector[total - i - 1];
    } else {
      vector[i] = vector[total - i - 1];
    }
  }
}

template <class T, std::size_t N, class Alloc = std::allocator<T>>
auto to_vector(std::array<T, N> const& a, Alloc const& allocator = Alloc())
    -> std::vector<T, Alloc> {
  return std::vector<T>{a.begin(), a.end(), allocator};
}

template <typename T, class Allocator, class Alloc = Allocator>
auto to_vector(std::vector<T, Allocator> const& a, Alloc const& allocator = Alloc())
    -> std::vector<T, Alloc> {
  return {a, allocator};
}

template <class T, class Alloc = std::allocator<T>>
auto to_vector(view<T const> items, Alloc const& allocator = Alloc()) -> std::vector<T, Alloc> {
  return std::vector<T>{items.begin(), items.end(), allocator};
}

template <class T, class Alloc = std::allocator<T>>
auto to_vector(strided_view<T const> items, Alloc const& allocator = Alloc())
    -> std::vector<T, Alloc> {
  return std::vector<T>{items.begin(), items.end(), allocator};
}

}  // namespace detail

}  // namespace poly

// wtf boost, range iterator doesn't seem to work with custom ranges out of the box
namespace boost {
template <class T>
struct range_iterator<T,
                      std::enable_if_t<std::is_base_of_v<poly::view<typename T::value_type>, T>>> {
  using type = typename T::iterator;
};

template <class T>
struct range_iterator<
    T, std::enable_if_t<std::is_base_of_v<poly::strided_view<typename T::value_type>, T>>> {
  using type = typename T::iterator;
};
}  // namespace boost

#endif  // SRC_POLYNOMIALS_UTILS_HPP_
