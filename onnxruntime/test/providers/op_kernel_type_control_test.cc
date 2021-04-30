// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/op_kernel_type_control.h"

#include <cstdint>

#include "boost/mp11.hpp"

#include "core/common/type_list.h"
#include "core/common/type_set_utils.h"

namespace onnxruntime {
namespace test {

namespace {
// test types to match op_kernel_type_control::TypesHolder
template <typename... T>
struct TestTypesHolder {
    using types = TypeList<T...>;
};

struct TestTypesHolderUnspecified {
};
}  // namespace

// default
static_assert(
    utils::type_set::IsEqual<
        op_kernel_type_control::EnabledTypes<
            TestTypesHolder<int32_t, int64_t, float, double>,
            TestTypesHolderUnspecified,
            TypeList<
                TestTypesHolderUnspecified,
                TestTypesHolderUnspecified>>::types,
        TypeList<int32_t, int64_t, float, double>>::value,
    "unexpected enabled types");

// default + allowed for Op
static_assert(
    utils::type_set::IsEqual<
        op_kernel_type_control::EnabledTypes<
            TestTypesHolder<int32_t, int64_t, float, double>,
            TestTypesHolderUnspecified,
            TypeList<
                TestTypesHolder<float, int64_t>,
                TestTypesHolderUnspecified>>::types,
        TypeList<int64_t, float>>::value,
    "unexpected enabled types");

// default + allowed for Op, all enabled
static_assert(
    utils::type_set::IsEqual<
        op_kernel_type_control::EnabledTypes<
            TestTypesHolder<int32_t, int64_t, float, double>,
            TestTypesHolderUnspecified,
            TypeList<
                TestTypesHolder<float, double, int32_t, int64_t>,
                TestTypesHolderUnspecified>>::types,
        TypeList<int32_t, int64_t, float, double>>::value,
    "unexpected enabled types");

// default + allowed for Op, allowed not subset of default
static_assert(
    utils::type_set::IsEqual<
        op_kernel_type_control::EnabledTypes<
            TestTypesHolder<int32_t, int64_t, float, double>,
            TestTypesHolderUnspecified,
            TypeList<
                TestTypesHolder<double, int64_t, char>,
                TestTypesHolderUnspecified>>::types,
        TypeList<int64_t, double>>::value,
    "unexpected enabled types");

// default + allowed for Op, all disabled
static_assert(
    utils::type_set::IsEqual<
        op_kernel_type_control::EnabledTypes<
            TestTypesHolder<int32_t, int64_t, float, double>,
            TestTypesHolderUnspecified,
            TypeList<
                TestTypesHolder<>,
                TestTypesHolderUnspecified>>::types,
        TypeList<>>::value,
    "unexpected enabled types");

// default + allowed for Op + allowed globally
static_assert(
    utils::type_set::IsEqual<
        op_kernel_type_control::EnabledTypes<
            TestTypesHolder<int32_t, int64_t, float, double>,
            TestTypesHolderUnspecified,
            TypeList<
                TestTypesHolder<float, int64_t>,
                TestTypesHolder<int64_t>>>::types,
        TypeList<int64_t>>::value,
    "unexpected enabled types");

// default + required + allowed for Op
static_assert(
    utils::type_set::IsEqual<
        op_kernel_type_control::EnabledTypes<
            TestTypesHolder<int32_t, int64_t, float, double>,
            TestTypesHolder<int64_t>,
            TypeList<
                TestTypesHolder<int32_t, double>,
                TestTypesHolderUnspecified>>::types,
        TypeList<int32_t, int64_t, double>>::value,
    "unexpected enabled types");

// default + required + allowed for Op, only required enabled
static_assert(
    utils::type_set::IsEqual<
        op_kernel_type_control::EnabledTypes<
            TestTypesHolder<int32_t, int64_t, float, double>,
            TestTypesHolder<int64_t>,
            TypeList<
                TestTypesHolder<>,
                TestTypesHolderUnspecified>>::types,
        TypeList<int64_t>>::value,
    "unexpected enabled types");

// default + required + allowed for Op + allowed globally
static_assert(
    utils::type_set::IsEqual<
        op_kernel_type_control::EnabledTypes<
            TestTypesHolder<int32_t, int64_t, float, double>,
            TestTypesHolder<int64_t>,
            TypeList<
                TestTypesHolder<int32_t, double>,
                TestTypesHolder<double>>>::types,
        TypeList<int64_t, double>>::value,
    "unexpected enabled types");

}  // namespace test
}  // namespace onnxruntime
