// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Compile-time tests for the type detection utilities in program.h.
// These static_asserts verify that the has_member_* variable templates
// and has_*_correct_type concepts correctly detect static const std::array members.

#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {
namespace details {
namespace test {

// Test-specific member detection for a member named 'a'
template <typename T, typename = void>
inline constexpr bool test_has_a = false;
template <typename T>
inline constexpr bool test_has_a<T, std::void_t<decltype(T::a)>> = true;

// Test-specific type correctness concept for a member named 'a'
template <typename T>
concept test_has_a_correct_type = requires {
  T::a;
  requires is_const_std_array<decltype(T::a)>::value;
  requires std::is_const_v<decltype(T::a)>;
  requires !std::is_member_pointer_v<decltype(&T::a)>;
};

struct TestClass_Empty {};
static_assert(!test_has_a<TestClass_Empty>);
static_assert(!test_has_a_correct_type<TestClass_Empty>);

struct TestClass_NotArray_0 {
  int b;
};
static_assert(!test_has_a<TestClass_NotArray_0>);
static_assert(!test_has_a_correct_type<TestClass_NotArray_0>);

struct TestClass_NotArray_1 {
  int a;
};
static_assert(test_has_a<TestClass_NotArray_1>);
static_assert(!test_has_a_correct_type<TestClass_NotArray_1>);

struct TestClass_NotArray_2 {
  const int a;
};
static_assert(test_has_a<TestClass_NotArray_2>);
static_assert(!test_has_a_correct_type<TestClass_NotArray_2>);

struct TestClass_NotStdArray_0 {
  const int a[2];
};
static_assert(test_has_a<TestClass_NotStdArray_0>);
static_assert(!test_has_a_correct_type<TestClass_NotStdArray_0>);

struct TestClass_NotStdArray_1 {
  static constexpr int a[] = {0};
};
static_assert(test_has_a<TestClass_NotStdArray_1>);
static_assert(!test_has_a_correct_type<TestClass_NotStdArray_1>);

struct TestClass_NotStdArray_2 {
  static int a[];
};
static_assert(test_has_a<TestClass_NotStdArray_2>);
static_assert(!test_has_a_correct_type<TestClass_NotStdArray_2>);

struct TestClass_NotStdArray_3 {
  static const int a[];
};
static_assert(test_has_a<TestClass_NotStdArray_3>);
static_assert(!test_has_a_correct_type<TestClass_NotStdArray_3>);

struct TestClass_StdArray_0 {
  std::array<int, 1> a = {1};
};
static_assert(test_has_a<TestClass_StdArray_0>);
static_assert(!test_has_a_correct_type<TestClass_StdArray_0>);

struct TestClass_StdArray_1 {
  static constexpr std::array<int, 2> a = {1, 2};
};
static_assert(test_has_a<TestClass_StdArray_1>);
static_assert(test_has_a_correct_type<TestClass_StdArray_1>);

struct TestClass_StdArray_2 {
  static const std::array<int, 3> a;
};
static_assert(test_has_a<TestClass_StdArray_2>);
static_assert(test_has_a_correct_type<TestClass_StdArray_2>);

struct TestClass_StdArray_3 {
  static constexpr const std::array<int, 4> a = {1, 2, 3, 4};
};
static_assert(test_has_a<TestClass_StdArray_3>);
static_assert(test_has_a_correct_type<TestClass_StdArray_3>);

struct TestClass_StdArray_4 {
  static std::array<int, 5> a;
};
static_assert(test_has_a<TestClass_StdArray_4>);
static_assert(!test_has_a_correct_type<TestClass_StdArray_4>);

}  // namespace test
}  // namespace details
}  // namespace webgpu
}  // namespace onnxruntime
