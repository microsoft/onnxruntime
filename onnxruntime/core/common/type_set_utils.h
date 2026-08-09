// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "boost/mp11.hpp"

namespace onnxruntime {
namespace utils {
namespace type_set {

// Given type sets A and B, IsSubset<A, B>::value is true when each element in A is also in B.
template <typename A, typename B>
struct IsSubset {
  static_assert(boost::mp11::mp_is_set<A>::value && boost::mp11::mp_is_set<B>::value,
                "A and B must be type sets.");

  static constexpr bool value =
      // each element of A is contained in B
      boost::mp11::mp_all_of_q<
          A,
          boost::mp11::mp_bind_front<boost::mp11::mp_set_contains, B>>::value;
};

// Given type sets A and B, IsEqual<A, B>::value is true when A contains the same elements as B.
template <typename A, typename B>
struct IsEqual {
  static_assert(boost::mp11::mp_is_set<A>::value && boost::mp11::mp_is_set<B>::value,
                "A and B must be type sets.");

  static constexpr bool value =
      boost::mp11::mp_size<A>::value == boost::mp11::mp_size<B>::value &&
      IsSubset<A, B>::value;
};

}  // namespace type_set
}  // namespace utils
}  // namespace onnxruntime
