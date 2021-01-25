// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/op_kernel_type_control.h"

#include <cstdint>

#include "boost/mp11.hpp"

#include "core/common/type_list.h"

namespace onnxruntime {
namespace test {

namespace {
template <typename A, typename B>
struct TypeSetEqual {
 private:
  static_assert(boost::mp11::mp_is_set<A>::value && boost::mp11::mp_is_set<B>::value,
                "A and B must both be sets.");
  using ABIntersection = boost::mp11::mp_set_intersection<A, B>;

 public:
  static constexpr bool value =
      (boost::mp11::mp_size<A>::value == boost::mp11::mp_size<B>::value) &&
      (boost::mp11::mp_size<ABIntersection>::value == boost::mp11::mp_size<A>::value);
};
}  // namespace

}  // namespace test

// specify supported and allowed
namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_SUPPORTED_TYPES(TestOpA, Input, 0, int32_t, int64_t, float, double);
ORT_SPECIFY_OP_KERNEL_ARG_ALLOWED_TYPES(TestOpA, Input, 0, float, int64_t, char);
}  // namespace op_kernel_type_control

static_assert(
    test::TypeSetEqual<
        ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(TestOpA, Input, 0),
        TypeList<float, int64_t>>::value,
    "Unexpected enabled types for TestOpA.");

// specify supported
namespace op_kernel_type_control {
ORT_SPECIFY_OP_KERNEL_ARG_SUPPORTED_TYPES(TestOpB, Output, 1, int32_t, int64_t);
}  // namespace op_kernel_type_control

static_assert(
    test::TypeSetEqual<
        ORT_OP_KERNEL_ARG_ENABLED_TYPE_LIST(TestOpB, Output, 1),
        TypeList<int32_t, int64_t>>::value,
    "Unexpected enabled types for TestOpB.");

}  // namespace onnxruntime
