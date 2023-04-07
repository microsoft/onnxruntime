// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "boost/mp11.hpp"

namespace onnxruntime {
namespace utils {

/**
* Check if the set of types contains the specified type.
*/
template <typename TypeSet, typename T>
constexpr bool HasType() {
  static_assert(boost::mp11::mp_is_set<TypeSet>::value, "TypeSet must be a type set.");

  return boost::mp11::mp_set_contains<TypeSet, T>::value;
}

template <typename T>
using SizeOfT = boost::mp11::mp_size_t<sizeof(T)>;

/**
* Check if the set of types contains a type with the same size as T.
*
* @remarks e.g. will return true if T is int32_t and the list contains any 4 byte type (i.e. sizeof(int32_t))
*               such as int32_t, uint32_t or float.
*/
template <typename TypeSet, typename T>
constexpr bool HasTypeWithSameSize() {
  static_assert(boost::mp11::mp_is_set<TypeSet>::value, "TypeSet must be a type set.");

  using EnabledTypeSizes = boost::mp11::mp_unique<boost::mp11::mp_transform<SizeOfT, TypeSet>>;
  return boost::mp11::mp_set_contains<EnabledTypeSizes, SizeOfT<T>>::value;
}

/**
 * The union of the given type sets.
 */
template <typename... TypeSets>
using TypeSetUnion = boost::mp11::mp_set_union<TypeSets...>;

}  // namespace utils
}  // namespace onnxruntime
