// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/op_kernel_type_control.h"

#include <cstdint>

#include "boost/mp11.hpp"

#include "core/common/type_list.h"

namespace onnxruntime {
namespace test {

template <typename A, typename B>
struct TypeSetsEqual {
 private:
  static_assert(boost::mp11::mp_is_set<A>::value && boost::mp11::mp_is_set<B>::value,
                "A and B must both be sets.");
  using ABIntersection = boost::mp11::mp_set_intersection<A, B>;

 public:
  static constexpr bool value =
      (boost::mp11::mp_size<A>::value == boost::mp11::mp_size<B>::value) &&
      (boost::mp11::mp_size<ABIntersection>::value == boost::mp11::mp_size<A>::value);
};

// test types to match op_kernel_type_control::TypesHolder
template <typename... T>
struct TestTypesHolder {
    using types = TypeList<T...>;
};

struct TestTypesHolderUnspecified {
};

// supported + allowed for Op
static_assert(
    TypeSetsEqual<
        op_kernel_type_control::EnabledTypes<
            TestTypesHolder<int32_t, int64_t, float, double>,
            TypeList<
                TestTypesHolder<float, int64_t, char>,
                TestTypesHolderUnspecified>>::types,
        TypeList<int64_t, float>>::value,
    "unexpected enabled types: supported + allowed + unspecified allowed");

// supported + allowed for Op + allowed globally
static_assert(
    TypeSetsEqual<
        op_kernel_type_control::EnabledTypes<
            TestTypesHolder<int32_t, int64_t, float, double>,
            TypeList<
                TestTypesHolder<float, int64_t, char>,
                TestTypesHolder<int64_t>>>::types,
        TypeList<int64_t>>::value,
    "unexpected enabled types: supported + allowed + allowed");

// supported
static_assert(
    TypeSetsEqual<
        op_kernel_type_control::EnabledTypes<
            TestTypesHolder<int32_t, int64_t, float, double>,
            TypeList<
                TestTypesHolderUnspecified,
                TestTypesHolderUnspecified>>::types,
        TypeList<int32_t, int64_t, float, double>>::value,
    "unexpected enabled types: supported + unspecified allowed + unspecified allowed");

}  // namespace test
}  // namespace onnxruntime
