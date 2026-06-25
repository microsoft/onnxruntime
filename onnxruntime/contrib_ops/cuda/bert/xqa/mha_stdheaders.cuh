/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#ifndef GENERATE_CUBIN
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#endif

#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif

#define HOST_DEVICE_FUNC __host__ __device__
#define DEVICE_FUNC __device__

namespace mha {

#ifndef GENERATE_CUBIN
template <typename T>
using numeric_limits = std::numeric_limits<T>;
using std::max;
using std::min;
#else

using uint8_t = unsigned char;
using int8_t = signed char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;
using int32_t = int;
using uint64_t = unsigned long long;
using uintptr_t = uint64_t;
static_assert(sizeof(uint8_t) == 1);
static_assert(sizeof(int8_t) == 1);
static_assert(sizeof(uint16_t) == 2);
static_assert(sizeof(uint32_t) == 4);
static_assert(sizeof(int32_t) == 4);
static_assert(sizeof(uint64_t) == 8);

template <typename T>
class numeric_limits;

template <>
class numeric_limits<int32_t> {
 public:
  static constexpr int32_t max() noexcept {
    return 0x7FFFFFFF;
  }
};

template <>
class numeric_limits<float> {
 public:
  static constexpr float lowest() noexcept {
    return -3.40282347E+38F;
  }
};

template <typename T>
DEVICE_FUNC constexpr T const& max(T const& a, T const& b) {
  return a > b ? a : b;
}

template <typename T>
DEVICE_FUNC constexpr T const& min(T const& a, T const& b) {
  return a < b ? a : b;
}

#endif

#ifndef GENERATE_CUBIN
template <bool cond, class T, class F>
using conditional_t = std::conditional_t<cond, T, F>;

template <bool cond, class T = void>
using enable_if_t = typename std::enable_if<cond, T>::type;
#else

// https://en.cppreference.com/w/cpp/types/conditional
template <bool cond, class T, class F>
struct conditional {
  using type = T;
};

template <class T, class F>
struct conditional<false, T, F> {
  using type = F;
};

template <bool cond, class T, class F>
using conditional_t = typename conditional<cond, T, F>::type;

template <bool cond, class T = void>
struct enable_if {
};

template <class T>
struct enable_if<true, T> {
  typedef T type;
};

template <bool cond, class T = void>
using enable_if_t = typename enable_if<cond, T>::type;
#endif

#ifndef GENERATE_CUBIN
using byte = std::byte;
#else
// https://en.cppreference.com/w/cpp/types/byte
enum class byte : unsigned char {
};
#endif

#ifndef GENERATE_CUBIN
using std::declval;
#else

// https://en.cppreference.com/w/cpp/types/add_reference
namespace detail {
template <class T>
struct type_identity {
  using type = T;
};  // or use std::type_identity (since C++20)

template <class T>  // Note that `cv void&` is a substitution failure
DEVICE_FUNC auto try_add_lvalue_reference(int) -> type_identity<T&>;
template <class T>  // Handle T = cv void case
DEVICE_FUNC auto try_add_lvalue_reference(...) -> type_identity<T>;

template <class T>
DEVICE_FUNC auto try_add_rvalue_reference(int) -> type_identity<T&&>;
template <class T>
DEVICE_FUNC auto try_add_rvalue_reference(...) -> type_identity<T>;
}  // namespace detail

template <class T>
struct add_lvalue_reference : decltype(detail::try_add_lvalue_reference<T>(0)) {
};

template <class T>
struct add_rvalue_reference : decltype(detail::try_add_rvalue_reference<T>(0)) {
};

// https://en.cppreference.com/w/cpp/utility/declval
template <typename T>
DEVICE_FUNC typename add_rvalue_reference<T>::type declval() noexcept {
  static_assert(false, "declval not allowed in an evaluated context");
}
#endif

#ifndef GENERATE_CUBIN
template <class T, size_t N>
using array = std::array<T, N>;
#else
// https://en.cppreference.com/w/cpp/container/array
template <class T, size_t N>
struct array;
#endif

#ifndef GENERATE_CUBIN
template <typename T, typename U>
using is_same = std::is_same<T, U>;
using std::is_same_v;
#else

// https://en.cppreference.com/w/cpp/types/integral_constant
template <class T, T v>
struct integral_constant {
  static constexpr T value = v;
  using value_type = T;
  using type = integral_constant;  // using injected-class-name

  DEVICE_FUNC constexpr operator value_type() const noexcept {
    return value;
  }

  DEVICE_FUNC constexpr value_type operator()() const noexcept {
    return value;
  }  // since c++14
};

using false_type = integral_constant<bool, false>;
using true_type = integral_constant<bool, true>;

// https://en.cppreference.com/w/cpp/types/is_same
template <class T, class U>
struct is_same : false_type {
};

template <class T>
struct is_same<T, T> : true_type {
};

template <class T, class U>
inline constexpr bool is_same_v = is_same<T, U>::value;

#endif

#ifndef GENERATE_CUBIN

using std::forward;
using std::is_empty;
using std::move;

#else

// /usr/include/c++/11/type_traits
template <typename _Tp>
struct is_empty : public integral_constant<bool, __is_empty(_Tp)> {
};

template <class T>
struct remove_reference {
  typedef T type;
};

template <class T>
struct remove_reference<T&> {
  typedef T type;
};

template <class T>
struct remove_reference<T&&> {
  typedef T type;
};

template <typename T>
constexpr typename remove_reference<T>::type&& move(T&& arg) {
  return static_cast<typename remove_reference<T>::type&&>(arg);
}

template <typename T>
constexpr T&& forward(typename remove_reference<T>::type& param) {
  return static_cast<T&&>(param);
}

#endif

// https://gcc.gnu.org/onlinedocs/libstdc++/libstdc++-api-4.5/a01066_source.html
namespace libstdcpp {
// Adds a const reference to a non-reference type.
template <typename _Tp>
struct __add_c_ref {
  typedef _Tp const& type;
};

template <typename _Tp>
struct __add_c_ref<_Tp&> {
  typedef _Tp& type;
};

// Adds a reference to a non-reference type.
template <typename _Tp>
struct __add_ref {
  typedef _Tp& type;
};

template <typename _Tp>
struct __add_ref<_Tp&> {
  typedef _Tp& type;
};

template <size_t _Idx, typename _Head, bool _IsEmpty>
struct _Head_base;

template <size_t _Idx, typename _Head>
struct _Head_base<_Idx, _Head, true> : public _Head {
  DEVICE_FUNC _Head_base()
      : _Head() {
  }

  DEVICE_FUNC _Head_base(_Head const& __h)
      : _Head(__h) {
  }

  template <typename _UHead>
  DEVICE_FUNC _Head_base(_UHead&& __h)
      : _Head(forward<_UHead>(__h)) {
  }

  DEVICE_FUNC _Head& _M_head() {
    return *this;
  }

  DEVICE_FUNC _Head const& _M_head() const {
    return *this;
  }

  DEVICE_FUNC void _M_swap_impl(_Head&) { /* no-op */
  }
};

template <size_t _Idx, typename _Head>
struct _Head_base<_Idx, _Head, false> {
  DEVICE_FUNC _Head_base()
      : _M_head_impl() {
  }

  DEVICE_FUNC _Head_base(_Head const& __h)
      : _M_head_impl(__h) {
  }

  template <typename _UHead>
  DEVICE_FUNC _Head_base(_UHead&& __h)
      : _M_head_impl(forward<_UHead>(__h)) {
  }

  DEVICE_FUNC _Head& _M_head() {
    return _M_head_impl;
  }

  DEVICE_FUNC _Head const& _M_head() const {
    return _M_head_impl;
  }

  DEVICE_FUNC void _M_swap_impl(_Head& __h) {
    using std::swap;
    swap(__h, _M_head_impl);
  }

  _Head _M_head_impl;
};

/**
 * Contains the actual implementation of the @c tuple template, stored
 * as a recursive inheritance hierarchy from the first element (most
 * derived class) to the last (least derived class). The @c Idx
 * parameter gives the 0-based index of the element stored at this
 * point in the hierarchy; we use it to implement a constant-time
 * get() operation.
 */
template <size_t _Idx, typename... _Elements>
struct _Tuple_impl;

/**
 * Zero-element tuple implementation. This is the basis case for the
 * inheritance recursion.
 */
template <size_t _Idx>
struct _Tuple_impl<_Idx> {
 protected:
  DEVICE_FUNC void _M_swap_impl(_Tuple_impl&) { /* no-op */
  }
};

/**
 * Recursive tuple implementation. Here we store the @c Head element
 * and derive from a @c Tuple_impl containing the remaining elements
 * (which contains the @c Tail).
 */
template <size_t _Idx, typename _Head, typename... _Tail>
struct _Tuple_impl<_Idx, _Head, _Tail...> : public _Tuple_impl<_Idx + 1, _Tail...>,
                                            private _Head_base<_Idx, _Head, is_empty<_Head>::value> {
  typedef _Tuple_impl<_Idx + 1, _Tail...> _Inherited;
  typedef _Head_base<_Idx, _Head, is_empty<_Head>::value> _Base;

  DEVICE_FUNC _Head& _M_head() {
    return _Base::_M_head();
  }

  DEVICE_FUNC _Head const& _M_head() const {
    return _Base::_M_head();
  }

  DEVICE_FUNC _Inherited& _M_tail() {
    return *this;
  }

  DEVICE_FUNC _Inherited const& _M_tail() const {
    return *this;
  }

  DEVICE_FUNC _Tuple_impl()
      : _Inherited(), _Base() {
  }

  explicit DEVICE_FUNC _Tuple_impl(_Head const& __head, _Tail const&... __tail)
      : _Inherited(__tail...), _Base(__head) {
  }

  template <typename _UHead, typename... _UTail>
  explicit DEVICE_FUNC _Tuple_impl(_UHead&& __head, _UTail&&... __tail)
      : _Inherited(forward<_UTail>(__tail)...), _Base(forward<_UHead>(__head)) {
  }

  DEVICE_FUNC _Tuple_impl(_Tuple_impl const& __arg)
      : _Inherited(__arg._M_tail()), _Base(__arg._M_head()) {
  }

  DEVICE_FUNC _Tuple_impl(_Tuple_impl&& __arg)
      : _Inherited(move(__arg._M_tail())), _Base(forward<_Head>(__arg._M_head())) {
  }

  template <typename... _UElements>
  DEVICE_FUNC _Tuple_impl(_Tuple_impl<_Idx, _UElements...> const& __arg)
      : _Inherited(__arg._M_tail()), _Base(__arg._M_head()) {
  }

  template <typename... _UElements>
  DEVICE_FUNC _Tuple_impl(_Tuple_impl<_Idx, _UElements...>&& __arg)
      : _Inherited(move(__arg._M_tail())), _Base(move(__arg._M_head())) {
  }

  DEVICE_FUNC _Tuple_impl& operator=(_Tuple_impl const& __arg) {
    _M_head() = __arg._M_head();
    _M_tail() = __arg._M_tail();
    return *this;
  }

  DEVICE_FUNC _Tuple_impl& operator=(_Tuple_impl&& __arg) {
    _M_head() = move(__arg._M_head());
    _M_tail() = move(__arg._M_tail());
    return *this;
  }

  template <typename... _UElements>
  DEVICE_FUNC _Tuple_impl& operator=(_Tuple_impl<_Idx, _UElements...> const& __arg) {
    _M_head() = __arg._M_head();
    _M_tail() = __arg._M_tail();
    return *this;
  }

  template <typename... _UElements>
  DEVICE_FUNC _Tuple_impl& operator=(_Tuple_impl<_Idx, _UElements...>&& __arg) {
    _M_head() = move(__arg._M_head());
    _M_tail() = move(__arg._M_tail());
    return *this;
  }

 protected:
  DEVICE_FUNC void _M_swap_impl(_Tuple_impl& __arg) {
    _Base::_M_swap_impl(__arg._M_head());
    _Inherited::_M_swap_impl(__arg._M_tail());
  }
};

/// tuple
template <typename... _Elements>
class tuple : public _Tuple_impl<0, _Elements...> {
  typedef _Tuple_impl<0, _Elements...> _Inherited;

 public:
  DEVICE_FUNC tuple()
      : _Inherited() {
  }

  explicit DEVICE_FUNC tuple(_Elements const&... __elements)
      : _Inherited(__elements...) {
  }

  template <typename... _UElements>
  explicit DEVICE_FUNC tuple(_UElements&&... __elements)
      : _Inherited(forward<_UElements>(__elements)...) {
  }

  DEVICE_FUNC tuple(tuple const& __arg)
      : _Inherited(static_cast<_Inherited const&>(__arg)) {
  }

  DEVICE_FUNC tuple(tuple&& __arg)
      : _Inherited(static_cast<_Inherited&&>(__arg)) {
  }

  template <typename... _UElements>
  DEVICE_FUNC tuple(tuple<_UElements...> const& __arg)
      : _Inherited(static_cast<_Tuple_impl<0, _UElements...> const&>(__arg)) {
  }

  template <typename... _UElements>
  DEVICE_FUNC tuple(tuple<_UElements...>&& __arg)
      : _Inherited(static_cast<_Tuple_impl<0, _UElements...>&&>(__arg)) {
  }

  // XXX http://gcc.gnu.org/ml/libstdc++/2008-02/msg00047.html
  template <typename... _UElements>
  DEVICE_FUNC tuple(tuple<_UElements...>& __arg)
      : _Inherited(static_cast<_Tuple_impl<0, _UElements...> const&>(__arg)) {
  }

  DEVICE_FUNC tuple& operator=(tuple const& __arg) {
    static_cast<_Inherited&>(*this) = __arg;
    return *this;
  }

  DEVICE_FUNC tuple& operator=(tuple&& __arg) {
    static_cast<_Inherited&>(*this) = move(__arg);
    return *this;
  }

  template <typename... _UElements>
  DEVICE_FUNC tuple& operator=(tuple<_UElements...> const& __arg) {
    static_cast<_Inherited&>(*this) = __arg;
    return *this;
  }

  template <typename... _UElements>
  DEVICE_FUNC tuple& operator=(tuple<_UElements...>&& __arg) {
    static_cast<_Inherited&>(*this) = move(__arg);
    return *this;
  }

  void DEVICE_FUNC swap(tuple& __arg) {
    _Inherited::_M_swap_impl(__arg);
  }
};

template <>
class tuple<> {
 public:
  DEVICE_FUNC void swap(tuple&) { /* no-op */
  }
};

/// Gives the type of the ith element of a given tuple type.
template <size_t __i, typename _Tp>
struct tuple_element;

/**
 * Recursive case for tuple_element: strip off the first element in
 * the tuple and retrieve the (i-1)th element of the remaining tuple.
 */
template <size_t __i, typename _Head, typename... _Tail>
struct tuple_element<__i, tuple<_Head, _Tail...>> : tuple_element<__i - 1, tuple<_Tail...>> {
};

/**
 * Basis case for tuple_element: The first element is the one we're seeking.
 */
template <typename _Head, typename... _Tail>
struct tuple_element<0, tuple<_Head, _Tail...>> {
  typedef _Head type;
};

/// Finds the size of a given tuple type.
template <typename _Tp>
struct tuple_size;

/// class tuple_size
template <typename... _Elements>
struct tuple_size<tuple<_Elements...>> {
  static const size_t value = sizeof...(_Elements);
};

template <typename... _Elements>
const size_t tuple_size<tuple<_Elements...>>::value;

template <size_t __i, typename _Head, typename... _Tail>
DEVICE_FUNC inline typename __add_ref<_Head>::type __get_helper(_Tuple_impl<__i, _Head, _Tail...>& __t) {
  return __t._M_head();
}

template <size_t __i, typename _Head, typename... _Tail>
DEVICE_FUNC inline typename __add_c_ref<_Head>::type __get_helper(_Tuple_impl<__i, _Head, _Tail...> const& __t) {
  return __t._M_head();
}

// Return a reference (const reference) to the ith element of a tuple.
// Any const or non-const ref elements are returned with their original type.
template <size_t __i, typename... _Elements>
DEVICE_FUNC inline typename __add_ref<typename tuple_element<__i, tuple<_Elements...>>::type>::type get(
    tuple<_Elements...>& __t) {
  return __get_helper<__i>(__t);
}

template <size_t __i, typename... _Elements>
DEVICE_FUNC inline typename __add_c_ref<typename tuple_element<__i, tuple<_Elements...>>::type>::type get(
    tuple<_Elements...> const& __t) {
  return __get_helper<__i>(__t);
}

// This class helps construct the various comparison operations on tuples
template <size_t __check_equal_size, size_t __i, size_t __j, typename _Tp, typename _Up>
struct __tuple_compare;

template <size_t __i, size_t __j, typename _Tp, typename _Up>
struct __tuple_compare<0, __i, __j, _Tp, _Up> {
  DEVICE_FUNC static bool __eq(_Tp const& __t, _Up const& __u) {
    return (get<__i>(__t) == get<__i>(__u) && __tuple_compare<0, __i + 1, __j, _Tp, _Up>::__eq(__t, __u));
  }

  DEVICE_FUNC static bool __less(_Tp const& __t, _Up const& __u) {
    return ((get<__i>(__t) < get<__i>(__u)) || !(get<__i>(__u) < get<__i>(__t)) && __tuple_compare<0, __i + 1, __j, _Tp, _Up>::__less(__t, __u));
  }
};

template <size_t __i, typename _Tp, typename _Up>
struct __tuple_compare<0, __i, __i, _Tp, _Up> {
  static bool __eq(_Tp const&, _Up const&) {
    return true;
  }

  static bool __less(_Tp const&, _Up const&) {
    return false;
  }
};

template <typename... _TElements, typename... _UElements>
DEVICE_FUNC bool operator==(tuple<_TElements...> const& __t, tuple<_UElements...> const& __u) {
  typedef tuple<_TElements...> _Tp;
  typedef tuple<_UElements...> _Up;
  return (__tuple_compare<tuple_size<_Tp>::value - tuple_size<_Up>::value, 0, tuple_size<_Tp>::value, _Tp, _Up>::__eq(
      __t, __u));
}

template <typename... _TElements, typename... _UElements>
DEVICE_FUNC bool operator<(tuple<_TElements...> const& __t, tuple<_UElements...> const& __u) {
  typedef tuple<_TElements...> _Tp;
  typedef tuple<_UElements...> _Up;
  return (
      __tuple_compare<tuple_size<_Tp>::value - tuple_size<_Up>::value, 0, tuple_size<_Tp>::value, _Tp, _Up>::__less(
          __t, __u));
}

template <typename... _TElements, typename... _UElements>
DEVICE_FUNC inline bool operator!=(tuple<_TElements...> const& __t, tuple<_UElements...> const& __u) {
  return !(__t == __u);
}

template <typename... _TElements, typename... _UElements>
DEVICE_FUNC inline bool operator>(tuple<_TElements...> const& __t, tuple<_UElements...> const& __u) {
  return __u < __t;
}

template <typename... _TElements, typename... _UElements>
DEVICE_FUNC inline bool operator<=(tuple<_TElements...> const& __t, tuple<_UElements...> const& __u) {
  return !(__u < __t);
}

template <typename... _TElements, typename... _UElements>
DEVICE_FUNC inline bool operator>=(tuple<_TElements...> const& __t, tuple<_UElements...> const& __u) {
  return !(__t < __u);
}

template <size_t...>
struct __index_holder {
};

template <size_t __i, typename _IdxHolder, typename... _Elements>
struct __index_holder_impl;

template <size_t __i, size_t... _Indexes, typename _IdxHolder, typename... _Elements>
struct __index_holder_impl<__i, __index_holder<_Indexes...>, _IdxHolder, _Elements...> {
  typedef typename __index_holder_impl<__i + 1, __index_holder<_Indexes..., __i>, _Elements...>::type type;
};

template <size_t __i, size_t... _Indexes>
struct __index_holder_impl<__i, __index_holder<_Indexes...>> {
  typedef __index_holder<_Indexes...> type;
};

template <typename... _Elements>
struct __make_index_holder : __index_holder_impl<0, __index_holder<>, _Elements...> {
};

template <typename... _TElements, size_t... _TIdx, typename... _UElements, size_t... _UIdx>
DEVICE_FUNC inline tuple<_TElements..., _UElements...> __tuple_cat_helper(tuple<_TElements...> const& __t,
                                                                          __index_holder<_TIdx...> const&, tuple<_UElements...> const& __u, __index_holder<_UIdx...> const&) {
  return tuple<_TElements..., _UElements...>(get<_TIdx>(__t)..., get<_UIdx>(__u)...);
}

template <typename... _TElements, size_t... _TIdx, typename... _UElements, size_t... _UIdx>
DEVICE_FUNC inline tuple<_TElements..., _UElements...> __tuple_cat_helper(tuple<_TElements...>&& __t,
                                                                          __index_holder<_TIdx...> const&, tuple<_UElements...> const& __u, __index_holder<_UIdx...> const&) {
  return tuple<_TElements..., _UElements...>(move(get<_TIdx>(__t))..., get<_UIdx>(__u)...);
}

template <typename... _TElements, size_t... _TIdx, typename... _UElements, size_t... _UIdx>
DEVICE_FUNC inline tuple<_TElements..., _UElements...> __tuple_cat_helper(tuple<_TElements...> const& __t,
                                                                          __index_holder<_TIdx...> const&, tuple<_UElements...>&& __u, __index_holder<_UIdx...> const&) {
  return tuple<_TElements..., _UElements...>(get<_TIdx>(__t)..., move(get<_UIdx>(__u))...);
}

template <typename... _TElements, size_t... _TIdx, typename... _UElements, size_t... _UIdx>
DEVICE_FUNC inline tuple<_TElements..., _UElements...> __tuple_cat_helper(tuple<_TElements...>&& __t,
                                                                          __index_holder<_TIdx...> const&, tuple<_UElements...>&& __u, __index_holder<_UIdx...> const&) {
  return tuple<_TElements..., _UElements...>(move(get<_TIdx>(__t))..., move(get<_UIdx>(__u))...);
}

template <typename... _TElements, typename... _UElements>
DEVICE_FUNC inline tuple<_TElements..., _UElements...> tuple_cat(
    tuple<_TElements...> const& __t, tuple<_UElements...> const& __u) {
  return __tuple_cat_helper(__t, typename __make_index_holder<_TElements...>::type(), __u,
                            typename __make_index_holder<_UElements...>::type());
}

template <typename... _TElements, typename... _UElements>
DEVICE_FUNC inline tuple<_TElements..., _UElements...> tuple_cat(
    tuple<_TElements...>&& __t, tuple<_UElements...> const& __u) {
  return __tuple_cat_helper(move(__t), typename __make_index_holder<_TElements...>::type(), __u,
                            typename __make_index_holder<_UElements...>::type());
}

template <typename... _TElements, typename... _UElements>
DEVICE_FUNC inline tuple<_TElements..., _UElements...> tuple_cat(
    tuple<_TElements...> const& __t, tuple<_UElements...>&& __u) {
  return __tuple_cat_helper(__t, typename __make_index_holder<_TElements...>::type(), move(__u),
                            typename __make_index_holder<_UElements...>::type());
}

template <typename... _TElements, typename... _UElements>
DEVICE_FUNC inline tuple<_TElements..., _UElements...> tuple_cat(tuple<_TElements...>&& __t, tuple<_UElements...>&& __u) {
  return __tuple_cat_helper(move(__t), typename __make_index_holder<_TElements...>::type(), move(__u),
                            typename __make_index_holder<_UElements...>::type());
}

template <typename... _Elements>
DEVICE_FUNC inline tuple<_Elements&...> tie(_Elements&... __args) {
  return tuple<_Elements&...>(__args...);
}

template <typename... _Elements>
DEVICE_FUNC inline void swap(tuple<_Elements...>& __x, tuple<_Elements...>& __y) {
  __x.swap(__y);
}

// A class (and instance) which can be used in 'tie' when an element
// of a tuple is not required
struct _Swallow_assign {
  template <class _Tp>
  DEVICE_FUNC _Swallow_assign& operator=(_Tp const&) {
    return *this;
  }
};

// TODO: Put this in some kind of shared file.
namespace {
_Swallow_assign ignore;
};  // anonymous namespace
}  // namespace libstdcpp

template <typename... Types>
using tuple = libstdcpp::tuple<Types...>;

using libstdcpp::tie;
using libstdcpp::tuple_cat;

#ifndef GENERATE_CUBIN
template <class T>
using remove_cv = std::remove_cv<T>;
template <class T>
using remove_cv_t = typename std::remove_cv<T>::type;
template <typename T>
using decay = std::decay<T>;
template <typename T>
using decay_t = std::decay_t<T>;
#else

// https://en.cppreference.com/w/cpp/types/is_array
template <class T>
struct is_array : false_type {
};

template <class T>
struct is_array<T[]> : true_type {
};

template <class T, size_t N>
struct is_array<T[N]> : true_type {
};

// https://en.cppreference.com/w/cpp/types/remove_extent
template <class T>
struct remove_extent {
  using type = T;
};

template <class T>
struct remove_extent<T[]> {
  using type = T;
};

template <class T, size_t N>
struct remove_extent<T[N]> {
  using type = T;
};

// https://en.cppreference.com/w/cpp/types/is_function
template <class>
struct is_function : false_type {
};

// specialization for regular functions
template <class Ret, class... Args>
struct is_function<Ret(Args...)> : true_type {
};

// specialization for variadic functions such as printf
template <class Ret, class... Args>
struct is_function<Ret(Args......)> : true_type {
};

// specialization for function types that have cv-qualifiers
template <class Ret, class... Args>
struct is_function<Ret(Args...) const> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) volatile> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) const volatile> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) const> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) volatile> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) const volatile> : true_type {
};

// specialization for function types that have ref-qualifiers
template <class Ret, class... Args>
struct is_function<Ret(Args...) &> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) const&> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) volatile&> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) const volatile&> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) &> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) const&> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) volatile&> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) const volatile&> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) &&> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) const&&> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) volatile&&> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) const volatile&&> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) &&> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) const&&> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) volatile&&> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) const volatile&&> : true_type {
};

// specializations for noexcept versions of all the above (C++17 and later)
template <class Ret, class... Args>
struct is_function<Ret(Args...) noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) const noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) volatile noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) const volatile noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) const noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) volatile noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) const volatile noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) & noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) const & noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) volatile & noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) const volatile & noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) & noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) const & noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) volatile & noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) const volatile & noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) && noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) const && noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) volatile && noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args...) const volatile && noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) && noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) const && noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) volatile && noexcept> : true_type {
};

template <class Ret, class... Args>
struct is_function<Ret(Args......) const volatile && noexcept> : true_type {
};

// https://en.cppreference.com/w/cpp/types/remove_cv
template <class T>
struct remove_cv {
  typedef T type;
};

template <class T>
struct remove_cv<T const> {
  typedef T type;
};

template <class T>
struct remove_cv<T volatile> {
  typedef T type;
};

template <class T>
struct remove_cv<const volatile T> {
  typedef T type;
};

template <class T>
struct remove_const {
  typedef T type;
};

template <class T>
struct remove_const<T const> {
  typedef T type;
};

template <class T>
struct remove_volatile {
  typedef T type;
};

template <class T>
struct remove_volatile<T volatile> {
  typedef T type;
};

template <class T>
using remove_cv_t = typename remove_cv<T>::type;

// https://en.cppreference.com/w/cpp/types/add_pointer
namespace detail {
template <class T>
auto try_add_pointer(int) -> type_identity<typename remove_reference<T>::type*>;  // usual case

template <class T>
auto try_add_pointer(...) -> type_identity<T>;  // unusual case (cannot form std::remove_reference<T>::type*)
}  // namespace detail

template <class T>
struct add_pointer : decltype(detail::try_add_pointer<T>(0)) {
};

// https://en.cppreference.com/w/cpp/types/decay
template <class T>
struct decay {
 private:
  typedef typename remove_reference<T>::type U;

 public:
  typedef typename conditional<is_array<U>::value, typename add_pointer<typename remove_extent<U>::type>::type,
                               typename conditional<is_function<U>::value, typename add_pointer<U>::type,
                                                    typename remove_cv<U>::type>::type>::type type;
};

template <typename T>
using decay_t = typename decay<T>::type;
#endif

#ifndef GENERATE_CUBIN
template <typename T>
using is_void = std::is_void<T>;
template <typename T>
inline constexpr bool is_void_v = std::is_void_v<T>;
#else
template <typename T>
using is_void = is_same<remove_cv_t<T>, void>;
template <typename T>
inline constexpr bool is_void_v = is_void<T>::value;
#endif

#ifndef GENERATE_CUBIN
template <typename T1, typename T2>
using pair = std::pair<T1, T2>;
#else
template <typename T1, typename T2>
struct pair {
  T1 first;
  T2 second;
};
#endif

}  // namespace mha

#if GENERATE_CUBIN
using uint8_t = mha::uint8_t;
using int8_t = mha::int8_t;
using uint16_t = mha::uint16_t;
using int32_t = mha::int32_t;
using uint32_t = mha::uint32_t;
using uint64_t = mha::uint64_t;
using uintptr_t = mha::uintptr_t;
#endif
