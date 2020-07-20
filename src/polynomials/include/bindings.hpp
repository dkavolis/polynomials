/**
 * @file bindings.hpp
 * @author Daumantas Kavolis <dkavolis>
 * @brief
 * @date 12-Jun-2020
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

#ifndef SRC_POLYNOMIALS_BINDINGS_HPP_
#define SRC_POLYNOMIALS_BINDINGS_HPP_

#include "polynomials.hpp"

MSVC_WARNING_DISABLE(4127 4267)
#include <boost/iterator/iterator_facade.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
MSVC_WARNING_POP()

namespace py = pybind11;

namespace poly {

// map arbitrary floating point types to fundamental for numpy compatibility
template <class T, typename = void>
struct mapped_float;

// map all FPs larger than double to long double
template <class T>
struct mapped_float<
    T, std::enable_if_t<(std::numeric_limits<T>::digits > std::numeric_limits<double>::digits)>> {
  // on MSVC long double and double are the same so map it to double instead
  using type = std::conditional_t<(sizeof(long double) > sizeof(double)), long double, double>;
};

// map all FPs in between float and double to double
template <class T>
struct mapped_float<
    T, std::enable_if_t<(std::numeric_limits<T>::digits > std::numeric_limits<float>::digits &&
                         std::numeric_limits<T>::digits <= std::numeric_limits<double>::digits)>> {
  using type = double;
};

// map all FPs smaller than float to float
template <class T>
struct mapped_float<
    T, std::enable_if_t<(std::numeric_limits<T>::digits <= std::numeric_limits<float>::digits)>> {
  using type = float;
};

template <class T>
using mapped_float_t = typename mapped_float<T>::type;

template <bool check, ssize_t max_dim = -1, class T>
void check_array_dimensions(py::array_t<T> const& array, ssize_t dimension) noexcept(!check) {
  if constexpr (check) {
    if constexpr (max_dim > 0) {
      if (array.ndim() > max_dim) {
        throw py::value_error(
            py::str("Too many dimensions: got {:d}, expected {:d}").format(array.ndim(), max_dim));
      }
    }
    if (dimension >= array.ndim()) {
      throw py::value_error(
          py::str("Dimension too large: got {:d}, expected {:d}").format(dimension, array.ndim()));
    }
  }
}

/// \brief Range over dimension in numpy array
/// \tparam check whether to perform dimensions check
/// \tparam max_dim maximum allowed number of dimensions
/// \tparam T array value type
/// \param array numpy array
/// \param dimension which dimension to make range for
/// \return lazy range
template <bool check = true, ssize_t max_dim = -1, class T>
auto to_range(py::array_t<T> const& array, ssize_t dimension = 0) -> strided_view<T const> {
  check_array_dimensions<check, max_dim>(array, dimension);
  ssize_t stride = array.strides()[dimension];
  ssize_t count = array.shape()[dimension];
  return {array.data(), narrow_cast<std::size_t>(count), stride};
}

/// \brief Mutable range over dimension in numpy array
/// \tparam check whether to perform dimensions check
/// \tparam max_dim maximum allowed number of dimensions
/// \tparam T array value type
/// \param array numpy array
/// \param dimension which dimension to make range for
/// \return lazy range
template <bool check = true, ssize_t max_dim = -1, class T>
auto to_range(py::array_t<T>& array, ssize_t dimension = 0) -> strided_view<T> {
  check_array_dimensions<check, max_dim>(array, dimension);
  ssize_t stride = array.strides()[dimension];
  ssize_t count = array.shape()[dimension];
  return {array.mutable_data(), narrow_cast<std::size_t>(count), stride};
}

constexpr auto value_caster = [](auto&& value) { return py::str(py::cast(value)); };

template <class Range, class F = decltype(value_caster),
          typename = std::enable_if_t<detail::is_range<Range>::value>>
class RangeFormatter {
 public:
  constexpr explicit RangeFormatter(Range& range, const char* sep = ", ",
                                    F format = value_caster) noexcept
      : range_(range), sep_(sep), format_(std::move(format)) {}

  friend auto operator<<(std::ostream& os, RangeFormatter const& range) -> std::ostream& {
    os << "[";
    auto begin = boost::begin(range.range_);
    auto&& end = boost::end(range.range_);
    for (;;) {
      os << range.format_(*begin);
      ++begin;
      if (begin == end) {
        os << "]";
        break;
      }
      os << range.sep_;
    }

    return os;
  }

 private:
  Range& range_;
  const char* sep_;
  F format_;
};

template <class T, class F>
class CustomFormatter {
 public:
  constexpr explicit CustomFormatter(T& item, F format) noexcept
      : item_(item), format_(std::move(format)) {}

  friend auto operator<<(std::ostream& os, CustomFormatter const& self) -> std::ostream& {
    self.format_(os, self.item_);
    return os;
  }

 private:
  T& item_;
  F format_;
};

template <class F = decltype(value_caster)>
constexpr auto make_range_stream(const char* sep = ", ", F format = value_caster) {
  return [sep, format = std::move(format)](auto&& range) {
    return RangeFormatter<std::remove_reference_t<decltype(range)>, F>(range, sep, format);
  };
};

template <class F>
constexpr auto make_value_stream(F format) {
  return [format = std::move(format)](auto&& value) {
    return CustomFormatter<std::remove_reference_t<decltype(value)>, F>(value, format);
  };
}

constexpr auto range_stream = make_range_stream();
constexpr auto value_stream = value_caster;

template <class T, class F = decltype(value_stream), class... Options>
void def_repr_str(py::class_<T, Options...>& klass, std::string name,
                  F const& make_stream = value_stream) {
  klass
      .def("__repr__",
           [name = std::move(name), make_stream](T const& self) {
             std::ostringstream ss;
             ss << name << "(" << make_stream(self) << ")";
             return ss.str();
           })
      .def("__str__", [make_stream](T const& self) {
        std::ostringstream ss;
        ss << make_stream(self);
        return ss.str();
      });
}

template <class T>
struct identity {
  using type = T;
};

template <class... T>
struct array_types {};
template <class... T>
struct arg_types {};

template <class... T, class... V>
constexpr auto operator|(array_types<T...> /*l*/, array_types<V...> /*r*/) noexcept
    -> array_types<T..., V...> {
  return {};
}

constexpr array_types<std::int8_t, std::int16_t, std::int32_t, std::int64_t, long long>
    int_arrays{};
constexpr array_types<std::uint8_t, std::uint16_t, std::uint32_t, std::uint64_t, unsigned long long>
    unsigned_arrays{};
constexpr array_types<float, double, long double> float_arrays{};
constexpr array_types<bool> bool_arrays{};
constexpr auto integral_arrays = int_arrays | unsigned_arrays;
constexpr auto arithmetic_arrays = integral_arrays | float_arrays;

// lambdas in C++17 don't take template parameters
template <class T, class InputType, class F, class FF>
class Evaluator2D {
 public:
  using in_t = InputType;

  template <class U>
  using out_t = std::invoke_result_t<F, T&, strided_view<U>>;

  Evaluator2D(F function, FF size) : function_(std::move(function)), size_(std::move(size)) {}

  template <class U>
  auto operator()(T& self, py::array_t<U> array) const
      -> std::variant<py::array_t<out_t<U>>, out_t<U>> {
    using Out = out_t<U>;

    // 0 dimensional array -> return empty array
    ssize_t dims = array.ndim();
    if (dims == 0) return py::array_t<Out>(0);

    view<ssize_t const> strides{array.strides(), narrow_cast<std::size_t>(dims)};
    view<ssize_t const> shape{array.shape(), narrow_cast<std::size_t>(dims)};
    U const* data = array.data();
    ssize_t count = narrow<ssize_t>(size_(self));

    // if expect a single item for range, accept any array and return same shape one
    if (count == 1) {
      // same shape and layout array as input
      py::array_t<Out> results(shape, strides);

      ssize_t size = results.size();

      // simply iterate over each element in memory
      view<Out> out{results.mutable_data(), narrow_cast<std::size_t>(size)};
      view<U const> in{data, narrow_cast<std::size_t>(size)};
      for (auto&& [i, x] : in | boost::adaptors::indexed())
        out[i] = function_(self, view<U const>{&x, 1});

      return results;
    }

    // expect last dimension to match
    if (shape[dims - 1] != count)
      throw py::value_error(
          py::str("Shape/dimension mismatch, expected {} but got {} in the last dimension")
              .format(count, shape[1]));

    strided_view<U const> input{data, narrow_cast<std::size_t>(shape[0]), strides[0]};
    // 1d
    if (dims == 1) {
      // single value
      return function_(self, input);
    }
    // anything other than 2d is not allowed now
    if (dims != 2) {
      throw py::value_error(py::str("Only 1D/2D arrays are supported but got {}")
                                .format(py::array_t<ssize_t>(dims, shape.data())));
    }

    auto result = py::array_t<Out>(shape[0]);
    Out* output = result.mutable_data();

    ssize_t last_stride = strides[1];
    // 2d arrays with last shape == 1 will have 0 stride which doesn't work with iterators so set it
    // to at least the size of 1 element
    if (shape[1] == 1 && last_stride == 0) last_stride = narrow_cast<ssize_t>(sizeof(U));
    for (auto&& [index, value] : input | boost::adaptors::indexed()) {
      output[index] = function_(
          self, strided_view<U const>(&value, narrow_cast<std::size_t>(shape[1]), last_stride));
    }

    return result;
  }

 private:
  F function_;
  FF size_;
};

template <class T, class in_t, class F, class FF>
auto evaluator_2d(F&& function, FF&& size) -> Evaluator2D<T, in_t, F, FF> {
  return Evaluator2D<T, in_t, F, FF>{std::forward<F>(function), std::forward<FF>(size)};
}

template <class... T, class FDef, class F, class... Extra>
void def_ranged_impl(FDef const& def, const char* name, F const& generator,
                     array_types<T...> /*unused*/, Extra const&... extra) {
  using expander = int[];
  (void)expander{(static_cast<void>(def(name, generator(identity<T>{}), extra...)), 0)...};
}

template <class... T, class... U, class FDef, class F, class... Extra>
void def_ranged_impl(FDef const& def, const char* name, F&& function, array_types<T...> types,
                     arg_types<U...> /*unused*/, Extra const&... extra) {
  def_ranged_impl(
      def, name,
      [function = std::forward<F>(function)](auto tp) {
        using type = typename decltype(tp)::type;
        return [function](U... args, py::array_t<type> const& range) {
          return function(std::forward<U>(args)..., range);
        };
      },
      types, extra...);
}

template <class T, ssize_t max_dim = 1, class... V, class Klass, class F, class... Extra>
void def_ranged(Klass& klass, const char* name, F function, array_types<V...> types,
                Extra const&... extra) {
  def_ranged_impl([&klass](auto&&... args) { klass.def(args...); }, name,
                  [function = std::move(function)](T& self, auto&& range) {
                    return function(self, to_range<true, max_dim>(range));
                  },
                  types, arg_types<T&>{}, extra...);
}

template <class T, class in_t, class Klass, class F, class FF, class... V, class... Extra>
void def_ranged_2d(Klass& klass, const char* name, F function, FF size_f, array_types<V...> types,
                   Extra const&... extra) {
  auto evaluator = evaluator_2d<T, in_t>(std::move(function), std::move(size_f));
  def_ranged_impl(
      [&klass](auto&&... args) { klass.def(args...); }, name,
      [evaluator = std::move(evaluator)](T& self, auto&& range) { return evaluator(self, range); },
      types, arg_types<T&>{}, extra...);
}

template <ssize_t max_dim = 1, class Klass, class F, class... V, class... Extra>
void def_ranged_init(Klass& klass, F function, array_types<V...> types, Extra const&... extra) {
  def_ranged_impl(
      [&klass](const char* /*name*/, auto&& f, auto&&... args) { klass.def(py::init(f), args...); },
      "",
      [function = std::move(function)](auto&& range) {
        return function(to_range<true, max_dim>(range));
      },
      types, arg_types<>{}, extra...);
}

template <ssize_t max_dim = 1, class Klass, class F, class... V, class... Extra>
void def_ranged_static(Klass& klass, const char* name, F function, array_types<V...> types,
                       Extra const&... extra) {
  def_ranged_impl([&klass](auto&&... args) { klass.def_static(args...); }, name,
                  [function = std::move(function)](auto&& range) {
                    return function(to_range<true, max_dim>(range));
                  },
                  types, arg_types<>{}, extra...);
}

template <class T, class F, class FF>
decltype(auto) make_getitem(F&& getter, FF&& size_getter) {
  return [getter = std::forward<F>(getter), size_getter = std::forward<FF>(size_getter)](
             T& self, py::slice slice) -> py::object {
    ssize_t start = 0;
    ssize_t stop = 0;
    ssize_t step = 0;
    ssize_t length = 0;
    if (!slice.compute(size_getter(self), &start, &stop, &step, &length)) {
      throw py::error_already_set();
    }
    auto istart = static_cast<std::size_t>(start);
    auto istep = static_cast<std::size_t>(step);
    auto ilength = static_cast<std::size_t>(length);

    if (length == 1) { return py::cast(getter(self, istart)); }

    // TODO: should return a view
    py::list list(length);

    for (std::size_t i = 0; i < ilength; ++i) {
      list[i] = py::cast(getter(self, istart));
      istart += istep;
    }

    return list;
  };
}

template <class T, class F, class FF>
decltype(auto) make_setitem(F&& setter, FF&& size_getter) {
  return [setter = std::forward<F>(setter), size_getter = std::forward<FF>(size_getter)](
             T& self, py::slice slice, const py::object& items) {
    ssize_t start = 0;
    ssize_t stop = 0;
    ssize_t step = 0;
    ssize_t length = 0;
    if (!slice.compute(size_getter(self), &start, &stop, &step, &length)) {
      throw py::error_already_set();
    }
    auto istart = static_cast<std::size_t>(start);
    if (length == 1) {
      setter(self, istart, items);
      return;
    }

    if (length != py::len(items)) {
      throw std::runtime_error(
          "Left and right hand size of slice assignment have different sizes!");
    }
    auto istep = static_cast<std::size_t>(step);

    for (auto&& item : items) {
      setter(self, istart, item);
      istart += istep;
    }
  };
}

template <class T, class... Options>
void def_copy_ctor(py::class_<T, Options...>& klass) {
  klass.def(py::init<T const&>(), py::doc("Copy constructor"));
}

template <class T>
auto is_registered() -> bool {
  return py::detail::get_type_info(typeid(T), false) != nullptr;
}

inline auto get_views_submodule(py::module& m) -> py::module {
  py::object module = py::getattr(m, "views", nullptr);

  if (module.ptr() == nullptr) {
    module = m.def_submodule("views", "Internal submodule for bound view types");
  }

  return module;
}

namespace detail {
POLY_MAKE_MEMBER_DETECTOR(coefficient);
POLY_MAKE_MEMBER_DETECTOR(stride);
}  // namespace detail

template <class T>
struct format_string {
  static auto format() -> const char* {
    static std::string const str = py::format_descriptor<T>::format();
    return str.c_str();
  }
};

template <class T, typename = void>
struct is_bufferable : std::false_type {};
template <class T>
struct is_bufferable<T, std::void_t<decltype(py::format_descriptor<T>::format())>>
    : std::true_type {};

template <class T, class F = decltype(range_stream)>
auto bind_view(py::module& m, const char* name, F const& format = range_stream) -> py::class_<T> {
  using value_type = typename T::value_type;

  std::string name_ = name;
  name_ += "View";
  auto klass =
      py::class_<T>(m, name_.c_str())
          .def("__getitem__", py::overload_cast<std::size_t>(&T::at, py::const_))
          .def(
              "__iter__", [](T& self) { return py::make_iterator(self); }, py::keep_alive<0, 1>{})
          .def("__len__", &T::size);

  def_repr_str(klass, std::move(name_), format);

  if constexpr (!std::is_const_v<value_type>)
    klass.def("__setitem__", [](T& self, std::size_t index, value_type value) {
      self.at(index) = std::move(value);
    });

  if constexpr (detail::has_stride_member<T>::value)
    klass.def_property_readonly("stride", &T::stride);

  if constexpr (is_bufferable<value_type>::value) {
    auto& typeObject = reinterpret_cast<PyHeapTypeObject&>(*klass.ptr());

    typeObject.as_buffer.bf_getbuffer = [](PyObject* obj, Py_buffer* buffer, int flags) {
      if (PyErr_Occurred()) throw py::error_already_set();
      if (!buffer) throw py::value_error("Null buffer");

      /* Zero-initialize the output and ask the class to fill it. If that
         fails for some reason, give up. Need to list all members otherwise
         GCC 4.8 loudly complains about missing initializers. */
      *buffer =
          Py_buffer{nullptr, nullptr, 0, 0, 0, 0, nullptr, nullptr, nullptr, nullptr, nullptr};
      pybind11::detail::make_caster<T> caster;
      if (!caster.load(obj, /*convert=*/false))
        throw py::cast_error("Failed to cast py object to view");
      T& self = caster;

      if ((flags & PyBUF_WRITABLE) == PyBUF_WRITABLE && !std::is_const_v<value_type>) {
        PyErr_SetString(PyExc_BufferError, "view is not writable");
        return -1;
      }

      buffer->ndim = 1;
      buffer->itemsize = sizeof(value_type);
      buffer->len = sizeof(value_type) * self.size();
      buffer->buf = const_cast<std::decay_t<value_type>*>(self.data());
      buffer->readonly = std::is_const<value_type>::value;
      if ((flags & PyBUF_FORMAT) == PyBUF_FORMAT)
        buffer->format = const_cast<char*>(format_string<value_type>::format());
      if (flags != PyBUF_SIMPLE) {
        /* The view is immutable (can't change its size after it has been
           constructed), so referencing the size directly is okay */
        buffer->shape = reinterpret_cast<Py_ssize_t*>(&detail::access_size_member(self));
        if ((flags & PyBUF_STRIDES) == PyBUF_STRIDES) {
          if constexpr (detail::has_stride_member<T>::value) {
            // do the same to strides as to shape
            buffer->strides = reinterpret_cast<Py_ssize_t*>(&detail::access_stride_member(self));
          } else {
            buffer->strides = &buffer->itemsize;
          }
        }
      }

      /* Set the memory owner to the object and increase its reference count.
         We need to keep the object around because buffer->shapes /
         buffer->strides might be referring to it, moreover setting it to
         something else (like ArrayView's memory owner object) would mean
         Python calls the releasebuffer on that object instead of on us,
         leading to reference count getting negative in many cases. */
      if (buffer->obj) throw py::value_error("buffer already has an owner");
      buffer->obj = obj;
      Py_INCREF(buffer->obj);
      return 0;
    };

    // only a view, no allocations
    typeObject.as_buffer.bf_releasebuffer = nullptr;
  }

  return klass;
}

template <class T, template <class> class View = view, bool with_const = true>
void bind_views(py::module& m, const char* name,
                std::bool_constant<with_const> /* unused */ = std::true_type{}) {
  py::module views = get_views_submodule(m);
  if (!is_registered<View<T>>()) bind_view<View<T>>(views, name);
  if constexpr (with_const) {
    if (!is_registered<View<const T>>()) {
      bind_view<View<T const>>(views, (std::string("Immutable") + name).c_str());
      py::implicitly_convertible<View<T>, View<T const>>();
    }
  }
}

template <class T, template <class> class View = PolynomialProductView, bool with_const = true>
void bind_product_views(py::module& m, const char* name,
                        std::bool_constant<with_const> /* unused */ = std::true_type{});

template <class T, template <class> class View = SobolItemView, bool with_const = true>
void bind_sobol_views(py::module& m, const char* name,
                      std::bool_constant<with_const> /* unused */ = std::true_type{});

template <class Real>
auto bind_number(py::module& m, const char* name) -> py::class_<Real> {
  using FP = mapped_float_t<Real>;

  auto klass =
      py::class_<Real>(m, name)
          .def(py::init<std::string>())
          .def(py::init<long long>())
          .def(py::init<FP>())
          .def(py::self + py::self)
          .def(0.0L + py::self)
          .def(py::self += py::self)
          .def(py::self - py::self)
          .def(0.0L - py::self)
          .def(py::self -= py::self)
          .def(+py::self)
          .def(-py::self)
          .def(py::self * py::self)
          .def(0.0L * py::self)
          .def(py::self *= py::self)
          .def(py::self / py::self)
          .def(0.0L / py::self)
          .def(py::self /= py::self)
          .def(py::self == py::self)
          .def(py::self != py::self)
          .def(py::self < py::self)
          .def(py::self > py::self)
          .def(py::self <= py::self)
          .def(py::self >= py::self)
          .def("__abs__",
               [](Real const& self) {
                 using std::abs;
                 return abs(self);
               })
          .def("__pow__",
               [](Real const& self, Real const& exponent) {
                 using std::pow;
                 return pow(self, exponent);
               })
          .def(hash(py::self))
          .def(int_(py::self))
          .def(float_(py::self))
          .def("__repr__",
               [name = std::string(name)](Real const& self) {
                 return py::str("{}({})").format(name, static_cast<FP>(self));
               })
          .def("__str__",
               [](Real const& self) { return py::str("{}").format(static_cast<FP>(self)); })
          .def("__format__",
               [](Real const& self, py::str const& format_spec) {
                 // either cast to long double and lose precision or rewrite the format parser...
                 return py::cast(static_cast<FP>(self)).attr("__format__")(format_spec);
               })
          .def(py::pickle(
              [](Real const& self) {
                std::ostringstream ss;
                ss << self;
                return ss.str();
              },
              [](std::string const& str) { return Real(str); }));

  py::implicitly_convertible<long double, Real>();
  py::implicitly_convertible<double, Real>();
  py::implicitly_convertible<float, Real>();

  py::implicitly_convertible<long long, Real>();
  py::implicitly_convertible<long, Real>();
  py::implicitly_convertible<int, Real>();

  bind_views<Real>(m, name);

  return klass;
}

template <typename T>
auto bind_polynomial(py::module& m, const char* name) -> py::class_<T> {
  using Traits = typename T::Traits;
  using OrderType = typename Traits::OrderType;
  using Real = typename Traits::Real;
  using FP = mapped_float_t<Real>;

  // TODO: domain errors
  auto klass =
      py::class_<T>(m, name)
          .def(py::init<OrderType>(), py::doc("Initialize polynomial with a specified order"))
          .def_property("order", py::overload_cast<>(&T::order, py::const_),
                        py::overload_cast<OrderType>(&T::order), "Order of the polynomial")
          .def(py::pickle([](T const& self) { return self.order(); },
                          [](OrderType const& order) { return T(order); }))
          .def("__repr__",
               [name = std::string(name)](T const& self) {
                 return py::str("{}({})").format(name, self.order());
               })
          .def("__str__", [](T const& self) { return py::str("{}").format(self.order()); });

  def_copy_ctor(klass);

  klass.def_property_readonly_static(
      "domain", [](py::object /* unused */) { return T::domain(); },
      py::doc("Abscissa range of the polynomial"));

  if constexpr (Traits::has_eval) {
    constexpr const char* EvalDocstring =
        "Evaluate the polynomial at point(s), points must be inside domain";
    klass.def("__call__", &T::operator(), py::arg("x"), py::doc(EvalDocstring))
        .def("__call__", py::vectorize([](T const* self, FP x) {
               return static_cast<FP>(self->operator()(x));
             }),
             py::arg("x"), py::doc(EvalDocstring));
  }

  if constexpr (Traits::has_prime) {
    constexpr const char* PrimeDocstring =
        "Evaluate derivative of the polynomial at point(s), points must be inside domain";
    klass.def("prime", &T::prime, py::arg("x"), py::doc(PrimeDocstring))
        .def("prime",
             py::vectorize([](T const* self, FP x) { return static_cast<FP>(self->prime(x)); }),
             py::arg("x"), py::doc(PrimeDocstring));
  }

  if constexpr (Traits::has_zeros) {
    klass.def("zeros", &T::zeros, py::doc("Roots of the polynomial"));
  }

  if constexpr (Traits::has_next) {
    constexpr const char* NextDocstring = R"doc(Evaluate the polynomial value at order+1.

Parameters
----------
x : float or array
    abscissa value(s)
pl : float or array
    current order polynomial value(s)
pl1m : float or array
    previous order polynomial value(s)

Returns
-------
float or array
)doc";
    klass
        .def("next", &T::next, py::arg("x"), py::arg("pl"), py::arg("pl1m"), py::doc(NextDocstring))
        .def("next", py::vectorize([](T const* self, FP x, FP Pl, FP Pl1m) {
               return static_cast<FP>(self->next(x, Pl, Pl1m));
             }),
             py::arg("x"), py::arg("pl"), py::arg("pl1m"), py::doc(NextDocstring));
  }

  if constexpr (Traits::has_weights)
    klass.def(
        "weights", [](T const& self) { return self.weights(check_bounds); },
        py::doc("Quadrature weights"));

  if constexpr (Traits::has_abscissa)
    klass.def(
        "abscissa", [](T const& self) { return self.abscissa(check_bounds); },
        py::doc("Quadrature points"));

  if constexpr (Traits::has_quadrature)
    klass.def(
        "integrate",
        [](T const* self, std::function<Real(Real)> const& function) {
          return self->integrate(function);
        },
        py::arg("function"), py::doc("Integrate function using quadrature over domain"));

  bind_views<T>(m, name);
  bind_product_views<T, PolynomialProductView>(m, (std::string(name) + "Product").c_str());
  bind_product_views<T, PolynomialProductSetView>(m, (std::string(name) + "ProductSet").c_str());
  return klass;
}

template <typename T>
auto bind_polynomial_series(py::module& m, const char* name) -> py::class_<T> {
  using Traits = typename T::Traits;
  using Real = typename Traits::Real;
  using OrderType = typename Traits::OrderType;
  using FP = mapped_float_t<Real>;

  auto klass =
      py::class_<T>(m, name)
          .def(py::init<OrderType>())
          .def("__len__", &T::size)
          .def("__getitem__", py::overload_cast<std::size_t>(&T::at))
          .def("__setitem__", [](T& self, std::size_t index,
                                 typename T::Real coefficient) { self.at(index) = coefficient; })
          .def(py::pickle(
              [](T const& self) {
                auto const coefficients = self.coefficients();
                py::tuple tuple(coefficients.size());
                for (auto&& [index, coefficient] : coefficients | boost::adaptors::indexed())
                  tuple[index] = coefficient;
                return tuple;
              },
              [](py::tuple const& tuple) {
                T series(narrow<OrderType>(tuple.size()));
                std::size_t index = 0;
                for (auto&& item : tuple) series[index++] = item.cast<Real>();
                return series;
              }));

  def_repr_str(klass, name, make_value_stream([](std::ostream& os, T const& self) -> std::ostream& {
                 return os << range_stream(self.coefficients());
               }));

  def_copy_ctor(klass);
  def_ranged_init<1>(
      klass, [](auto const& range) { return std::make_unique<T>(range); }, integral_arrays);

  klass.def_static("domain", &T::domain);

  if constexpr (T::Traits::has_eval)
    klass.def("__call__", &T::operator()).def("__call__", py::vectorize([](T const* self, FP x) {
                                                return static_cast<FP>(self->operator()(x));
                                              }));

  //  if constexpr (T::has_prime)
  //    klass.def("prime", &T::prime).def("prime", py::vectorize([](T const* self, FP x) {
  //                                        return static_cast<FP>(self->prime(x));
  //                                      }));

  //  if constexpr (T::has_zeros) klass.def("zeros", &T::zeros);

  //  if constexpr (T::has_next)
  //    klass.def("next", &T::next)
  //        .def("next", py::vectorize([](T const* self, FP x, FP Pl, FP Pl1m) {
  //               return static_cast<FP>(self->next(x, Pl, Pl1m));
  //             }));

  if constexpr (T::Traits::has_weights)
    klass.def("weights", [](T const& self) { return self.weights(check_bounds); });

  if constexpr (T::Traits::has_weights)
    klass.def("abscissa", [](T const& self) { return self.abscissa(check_bounds); });

  // projection only works if polynomials are orthonormal
  if constexpr (T::Traits::is_orthonormal && T::Traits::has_quadrature) {
    def_ranged_static<1>(
        klass, "project", [](auto const& y) { return T::project(y); }, float_arrays);
    klass.def_static(
        "project",
        [](std::function<Real(Real)> const& function, OrderType order) {
          return T::project(function, order);
        },
        py::arg("function"), py::arg("order"));
  }
  return klass;
}

template <typename Poly>
auto bind_polynomial_sequence(py::module& m, const char* name) {
  using Traits = typename Poly::Traits;
  using OrderType = typename Traits::OrderType;
  using Real = typename Traits::Real;
  using Sequence =
      decltype(polynomial_sequence<Poly>(std::declval<OrderType>(), std::declval<Real>()));
  auto klass = py::class_<Sequence>(m, name)
                   .def(py::init([](OrderType max_order, Real x) {
                          return polynomial_sequence<Poly>(max_order, x);
                        }),
                        py::arg("max_order"), py::arg("x"),
                        py::doc("Create polynomial sequence at specified x value"))
                   .def("__len__", [](Sequence const& self) { return self.size(); })
                   .def(
                       "__iter__", [](Sequence& self) { return py::make_iterator(self); },
                       py::keep_alive<0, 1>{})
                   .def_property(
                       "x", [](Sequence const& self) { return self.x(); },
                       [](Sequence& self, Real x) { self.x() = std::move(x); })
                   .def_property(
                       "order", [](Sequence const& self) { return self.order(); },
                       [](Sequence& self, OrderType order) { self.order() = std::move(order); })
                   .def("__repr__",
                        [name = std::string(name)](Sequence const& self) {
                          return py::str("{}({}, {})").format(name, self.order(), self.x());
                        })
                   .def("__str__", [](Sequence const& self) {
                     return py::str("[{}, {}]").format(self.order(), self.x());
                   });

  def_copy_ctor(klass);
  return klass;
}

template <class T>
auto bind_product_view(py::module& m, const char* name) -> py::class_<T> {
  using Real = typename T::Real;
  using FP = mapped_float_t<Real>;
  using OrderType = typename T::OrderType;
  std::optional<py::class_<T>> klass_;
  if constexpr (detail::has_coefficient_member<T>::value)
    klass_ = bind_view<T>(
        m, name, make_value_stream([](std::ostream& os, T const& self) -> std::ostream& {
          os << "(" << range_stream(self) << ", " << value_stream(self.coefficient()) << ")";
          return os;
        }));
  else
    klass_ = bind_view<T>(m, name);

  auto klass = klass_.value();

  klass
      .def("matches_orders",
           [](T const& self, py::array_t<OrderType> const& orders) {
             return self.matches_orders(to_range(orders), check_bounds);
           })
      .def("matches_zeros",
           [](T const& self, py::array_t<bool> const& zeros) {
             return self.matches_zeros(to_range(zeros), check_bounds);
           })
      .def("matches_non_zeros", [](T const& self, py::array_t<bool> const& non_zeros) {
        return self.matches_non_zeros(to_range(non_zeros), check_bounds);
      });

  def_ranged_2d<T const, FP>(
      klass, "__call__",
      [](T const& self, auto const& range) {
        return static_cast<FP>(self(range, no_bounds_check));
      },
      [](T const& self) { return self.size(); }, float_arrays);

  if constexpr (!std::is_const_v<typename T::value_type>) {
    def_ranged<T, 1>(
        klass, "assign", [](T& self, auto const& orders) { self.assign(orders, check_bounds); },
        integral_arrays);
    klass.def("__setitem__", [](T& self, std::size_t index, typename T::OrderType value) {
      self.at(index).order(value);
    });
  }

  if constexpr (detail::has_coefficient_member<T>::value) {
    if constexpr (std::is_const_v<typename T::value_type>) {
      klass.def_property_readonly("coefficient", [](T const& self) { return self.coefficient(); });
    } else {
      klass.def_property(
          "coefficient", [](T const& self) { return self.coefficient(); },
          [](T& self, Real coefficient) { self.coefficient() = std::move(coefficient); });
    }
  }

  return klass;
}

template <class T, template <class> class View, bool with_const>
void bind_product_views(py::module& m, const char* name,
                        std::bool_constant<with_const> /* unused */) {
  py::module views = get_views_submodule(m);
  if (!is_registered<View<T>>()) bind_product_view<View<T>>(views, name);
  if constexpr (with_const) {
    if (!is_registered<View<const T>>()) {
      bind_product_view<View<T const>>(views, (std::string("Immutable") + name).c_str());
      py::implicitly_convertible<View<T>, View<T const>>();
    }
  }
}

template <class T>
auto bind_polynomial_product(py::module& m, const char* name) -> py::class_<T> {
  using Traits = typename T::Traits;
  using Real = typename Traits::Real;
  using FP = mapped_float_t<Real>;
  using OrderType = typename Traits::OrderType;
  using PolynomialType = typename T::PolynomialType;
  auto klass =
      py::class_<T>(m, name)
          .def(py::init<OrderType>(), py::arg("dimensions"))
          .def("__len__", py::overload_cast<>(&T::dimensions, py::const_))
          .def(
              "__iter__", [](T& self) { return py::make_iterator(self); }, py::keep_alive<0, 1>{})
          .def("__getitem__", py::overload_cast<std::size_t>(&T::at))
          .def("__setitem__", [](T& self, std::size_t index,
                                 PolynomialType poly) { self.at(index) = std::move(poly); })
          .def("__setitem__",
               [](T& self, std::size_t index, OrderType order) { self.at(index).order(order); })
          .def("append", [](T& self, PolynomialType poly) { self.emplace_back(std::move(poly)); })
          .def("append", [](T& self, OrderType order) { self.emplace_back(std::move(order)); })
          .def("clear", [](T& self) { self.clear(); })
          .def("resize", [](T& self, std::size_t size) { self.resize(size); })
          .def_property_readonly(
              "polynomials", [](T& self) { return self.polynomials(); }, py::keep_alive<0, 1>{})
          .def(py::pickle(
              [](T const& self) {
                py::tuple tuple(self.dimensions());
                for (auto&& [index, order] : self | boost::adaptors::indexed())
                  tuple[index] = order;

                return tuple;
              },
              [](py::tuple const& tuple) {
                T product(narrow<OrderType>(tuple.size()));
                std::size_t index = 0;
                for (auto&& item : tuple) product[index++].order(item.cast<OrderType>());

                return product;
              }));

  def_repr_str(klass, name, range_stream);

  def_ranged_init<1>(
      klass, [](auto const& orders) { return std::make_unique<T>(orders); }, integral_arrays);
  def_ranged_2d<T const, FP>(
      klass, "__call__",
      [](T const& self, auto const& x) { return static_cast<FP>(self(x, no_bounds_check)); },
      [](T const& self) { return self.dimensions(); }, float_arrays);

  def_copy_ctor(klass);
  py::implicitly_convertible<T, typename T::View>();
  py::implicitly_convertible<T, typename T::ConstView>();

  return klass;
}

template <class T>
auto bind_polynomial_product_set(py::module& m, const char* name) -> py::class_<T> {
  using Traits = typename T::Traits;
  using Real = typename Traits::Real;
  using FP = mapped_float_t<Real>;
  using OrderType = typename Traits::OrderType;
  using PolynomialType = typename T::PolynomialType;
  auto klass =
      py::class_<T>(m, name)
          .def(py::init<std::size_t>(), py::arg("dimensions"))
          .def(py::init<std::size_t, std::size_t>(), py::arg("size"), py::arg("dimensions"))
          .def("__len__", &T::size)
          .def_property("dimensions", py::overload_cast<>(&T::dimensions, py::const_),
                        py::overload_cast<std::size_t>(&T::dimensions))
          .def_property_readonly("index_count", &T::index_count)
          .def_property_readonly("coefficients", py::overload_cast<>(&T::coefficients),
                                 py::keep_alive<0, 1>{})
          .def(
              "__iter__", [](T& self) { return py::make_iterator(self); }, py::keep_alive<0, 1>{})
          .def("__getitem__", py::overload_cast<std::size_t>(&T::at), py::keep_alive<0, 1>{})
          .def("clear", [](T& self) { self.clear(); })
          .def("resize", [](T& self, std::size_t size) { self.resize(size); })
          .def("sobol", &T::template sobol<>)
          .def("erase", &T::template erase<true>)
          .def("merge_repeated", &T::merge_repeated)
          .def("assign",
               [](T& self,
                  // must be C style since T::assign expects last dimension to be contiguous
                  py::array_t<OrderType, py::array::c_style | py::array::forcecast> const& orders) {
                 if (orders.ndim() != 2)
                   throw py::value_error(
                       py::str("orders array must be 2D, got {} dimensions").format(orders.ndim()));
                 self.assign(strided_view<OrderType const>(orders.data(), orders.size(),
                                                           orders.strides()[0]),
                             orders.shape()[1]);
               })
          .def(py::pickle(
              [](T const& self) {
                return py::make_tuple(self.dimensions(), self.size(),
                                      detail::to_vector(self.polynomials()),
                                      detail::to_vector(self.coefficients()));
              },
              [](py::tuple const& tuple) {
                if (tuple.size() != 4) throw std::runtime_error("Invalid state!");

                T set(tuple[1].cast<std::size_t>(), tuple[0].cast<std::size_t>());
                auto&& polys = tuple[2].cast<py::sequence>();
                auto&& coefficients = tuple[3].cast<py::sequence>();

                if (py::len(polys) != set.index_count())
                  throw std::runtime_error("Invalid polynomials state!");
                for (auto&& [index, poly] : set.polynomials() | boost::adaptors::indexed())
                  poly = polys[index].template cast<PolynomialType>();

                if (py::len(coefficients) != set.size())
                  throw std::runtime_error("Invalid coefficients state!");
                for (auto&& [index, coefficient] : set.coefficients() | boost::adaptors::indexed())
                  coefficient = coefficients[index].template cast<Real>();

                return set;
              }));

  def_repr_str(klass, name, make_range_stream("\n", [](auto&& subrange) {
                 return make_value_stream([](std::ostream& os, auto&& subrange) -> std::ostream& {
                   os << "[" << range_stream(subrange) << ", "
                      << value_stream(subrange.coefficient()) << "]";
                   return os;
                 })(subrange);
               }));

  def_ranged_static(
      klass, "full_set", [](auto const& range) { return T::full_set(range); }, integral_arrays);

  auto size = [](T const& self) { return self.dimensions(); };
  def_ranged<T>(
      klass, "assign_coefficients",
      [](T& self, auto const& coefficients) { self.assign_coefficients(coefficients); },
      float_arrays);
  def_ranged_2d<T const, FP>(
      klass, "__call__",
      [](T const& self, auto const& x) { return static_cast<FP>(self(x, no_bounds_check)); }, size,
      float_arrays);
  def_ranged_2d<T const, OrderType>(
      klass, "index",
      [](T const& self, auto const& orders) {
        std::optional<std::size_t> index = self.index_of(orders, no_bounds_check);
        if (!index) throw std::out_of_range("No such multidimensional tensor");
        return index.value();
      },
      size, integral_arrays);

  def_copy_ctor(klass);
  return klass;
}

template <class T>
auto bind_sobol_view(py::module& m, const char* name) -> py::class_<T> {
  using Real = typename T::Real;
  auto klass = bind_view<T>(
      m, name, make_value_stream([](std::ostream& os, T const& self) -> std::ostream& {
        os << "(" << range_stream(self) << ", " << value_stream(self.coefficient()) << ")";
        return os;
      }));

  if constexpr (std::is_const_v<Real>)
    klass.def_property_readonly("coefficient", [](T const& self) { return self.coefficient(); });
  else
    klass.def_property(
        "coefficient", [](T const& self) { return self.coefficient(); },
        [](T& self, Real coefficient) { self.coefficient() = std::move(coefficient); });

  auto size = [](T const& self) { return self.size(); };
  def_ranged_2d<T const, bool>(
      klass, "matches_zeros",
      [](T const& self, auto const& zeros) { return self.matches_zeros(zeros, no_bounds_check); },
      size, bool_arrays);
  def_ranged_2d<T const, bool>(
      klass, "matches_non_zeros",
      [](T const& self, auto const& non_zeros) {
        return self.matches_non_zeros(non_zeros, no_bounds_check);
      },
      size, bool_arrays);

  return klass;
}

template <class T, template <class> class View, bool with_const>
void bind_sobol_views(py::module& m, const char* name,
                      std::bool_constant<with_const> /* unused */) {
  py::module views = get_views_submodule(m);
  if (!is_registered<View<T>>()) bind_sobol_view<View<T>>(views, name);
  if constexpr (with_const) {
    if (!is_registered<View<const T>>()) {
      bind_sobol_view<View<T const>>(views, (std::string("Immutable") + name).c_str());
      py::implicitly_convertible<View<T>, View<T const>>();
    }
  }
}

template <class Real>
auto bind_sobol(py::module& m, const char* name) -> py::class_<Sobol<Real>> {
  using T = Sobol<Real>;
  using FP = mapped_float_t<Real>;

  auto klass =
      py::class_<Sobol<Real>>(m, name)
          .def(py::init<std::vector<std::size_t>, std::vector<Real>, std::size_t>(),
               py::arg("indices"), py::arg("coefficients"), py::arg("dimensions"))
          .def("__getitem__", py::overload_cast<std::size_t>(&T::at), py::keep_alive<0, 1>{})
          .def("__len__", &T::size)
          .def(
              "__iter__",
              [](T& self) {
                auto range = self.mutable_range();
                return py::make_iterator(range.begin(), range.end());
              },
              py::keep_alive<0, 1>{})
          .def_property_readonly("index_count", &T::index_count)
          .def_property_readonly("dimensions", &T::dimensions)
          .def_property_readonly("variance", &T::variance)
          .def_property_readonly("mean", &T::mean)
          .def_property_readonly("indices", &T::mutable_indices)
          .def_property_readonly("coefficients", &T::mutable_coefficients)
          .def("recalculate", &T::recalculate)
          .def("sensitivity",
               [](T const& self, std::size_t index) {
                 return self.sensitivity(index, check_bounds);
               })
          .def("total_sensitivity",
               [](T const& self, std::size_t index) {
                 return self.total_sensitivity(index, check_bounds);
               })
          .def("total_sensitivity", py::vectorize([](T const* self, std::size_t index) {
                 return static_cast<FP>(self->total_sensitivity(index, check_bounds));
               }))
          .def(py::pickle(
              [](T const& self) {
                return py::make_tuple(self.dimensions(), detail::to_vector(self.indices()),
                                      detail::to_vector(self.coefficients()));
              },
              [](py::tuple const& tuple) {
                if (tuple.size() != 3) throw std::runtime_error("Invalid state!");

                T sobol(tuple[1].cast<typename T::indices_vector>(),
                        tuple[2].cast<typename T::coefficients_vector>(),
                        tuple[0].cast<std::size_t>());
                return sobol;
              }));

  def_repr_str(klass, name, make_range_stream("\n", [](auto&& subrange) {
                 return make_value_stream([](std::ostream& os, auto&& subrange) -> std::ostream& {
                   os << "[" << range_stream(subrange) << ", "
                      << value_stream(subrange.coefficient()) << "]";
                   return os;
                 })(subrange);
               }));

  def_ranged_2d<T const, std::size_t>(
      klass, "sensitivity",
      [](T const& self, auto const& indices) {
        return static_cast<FP>(self.sensitivity(indices, no_bounds_check));
      },
      [](T const& self) { return self.size(); }, integral_arrays);

  def_copy_ctor(klass);
  bind_views<std::size_t>(m, "Size");
  bind_sobol_views<Real, SobolItemView>(m, name);
  return klass;
}

template <class Impl, class Traits = PolynomialTraits<Impl>>
void bind_all_polynomials(py::module& m, std::string const& name) {
  using PolynomialT = Polynomial<Impl, Traits>;
  using Series = PolynomialSeries<PolynomialT>;
  using Product = PolynomialProduct<PolynomialT>;
  using ProductSet = PolynomialProductSet<PolynomialT>;

  bind_polynomial<PolynomialT>(m, name.c_str());
  bind_polynomial_series<Series>(m, (name + "Series").c_str());
  bind_polynomial_sequence<PolynomialT>(m, (name + "Sequence").c_str());
  bind_polynomial_product<Product>(m, (name + "Product").c_str());
  bind_polynomial_product_set<ProductSet>(m, (name + "ProductSet").c_str());
}

}  // namespace poly

#endif  // SRC_POLYNOMIALS_BINDING
