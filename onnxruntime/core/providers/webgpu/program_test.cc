// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Compile-time tests for the type detection utilities in program.h.
// These static_asserts verify that is_const_std_array, the has_member_* variable templates,
// and has_*_correct_type concepts correctly detect static const std::array members.

#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {
namespace details {
namespace test {

// ============================================================================
// Tests for is_const_std_array
// ============================================================================

static_assert(!is_const_std_array<int>::value);
static_assert(!is_const_std_array<const int>::value);
static_assert(!is_const_std_array<std::array<int, 3>>::value);
static_assert(is_const_std_array<const std::array<int, 3>>::value);
static_assert(is_const_std_array<const std::array<int, 0>>::value);

// ============================================================================
// Tests for has_member_constants / has_constants_correct_type
// ============================================================================

struct NoMembers {};
static_assert(!has_member_constants<NoMembers>);
static_assert(!has_constants_correct_type<NoMembers>);

struct Constants_WrongName {
  static constexpr std::array<int, 1> not_constants = {1};
};
static_assert(!has_member_constants<Constants_WrongName>);
static_assert(!has_constants_correct_type<Constants_WrongName>);

struct Constants_PlainInt {
  int constants;
};
static_assert(has_member_constants<Constants_PlainInt>);
static_assert(!has_constants_correct_type<Constants_PlainInt>);

struct Constants_ConstInt {
  const int constants;
};
static_assert(has_member_constants<Constants_ConstInt>);
static_assert(!has_constants_correct_type<Constants_ConstInt>);

struct Constants_CArray {
  const int constants[2];
};
static_assert(has_member_constants<Constants_CArray>);
static_assert(!has_constants_correct_type<Constants_CArray>);

struct Constants_StaticCArray {
  static constexpr int constants[] = {0};
};
static_assert(has_member_constants<Constants_StaticCArray>);
static_assert(!has_constants_correct_type<Constants_StaticCArray>);

struct Constants_StaticNonConstCArray {
  static int constants[];
};
static_assert(has_member_constants<Constants_StaticNonConstCArray>);
static_assert(!has_constants_correct_type<Constants_StaticNonConstCArray>);

struct Constants_StaticConstCArray {
  static const int constants[];
};
static_assert(has_member_constants<Constants_StaticConstCArray>);
static_assert(!has_constants_correct_type<Constants_StaticConstCArray>);

struct Constants_NonConstStdArray {
  std::array<int, 1> constants = {1};
};
static_assert(has_member_constants<Constants_NonConstStdArray>);
static_assert(!has_constants_correct_type<Constants_NonConstStdArray>);

struct Constants_NonConstStaticStdArray {
  static std::array<int, 5> constants;
};
static_assert(has_member_constants<Constants_NonConstStaticStdArray>);
static_assert(!has_constants_correct_type<Constants_NonConstStaticStdArray>);

struct Constants_StaticConstexprStdArray {
  static constexpr std::array<int, 2> constants = {1, 2};
};
static_assert(has_member_constants<Constants_StaticConstexprStdArray>);
static_assert(has_constants_correct_type<Constants_StaticConstexprStdArray>);

struct Constants_StaticConstStdArray {
  static const std::array<int, 3> constants;
};
static_assert(has_member_constants<Constants_StaticConstStdArray>);
static_assert(has_constants_correct_type<Constants_StaticConstStdArray>);

struct Constants_StaticConstexprConstStdArray {
  static constexpr const std::array<int, 4> constants = {1, 2, 3, 4};
};
static_assert(has_member_constants<Constants_StaticConstexprConstStdArray>);
static_assert(has_constants_correct_type<Constants_StaticConstexprConstStdArray>);

// ============================================================================
// Tests for has_member_overridable_constants / has_overridable_constants_correct_type
// ============================================================================

static_assert(!has_member_overridable_constants<NoMembers>);
static_assert(!has_overridable_constants_correct_type<NoMembers>);

struct OverridableConstants_WrongType {
  int overridable_constants;
};
static_assert(has_member_overridable_constants<OverridableConstants_WrongType>);
static_assert(!has_overridable_constants_correct_type<OverridableConstants_WrongType>);

struct OverridableConstants_Correct {
  static constexpr std::array<int, 1> overridable_constants = {1};
};
static_assert(has_member_overridable_constants<OverridableConstants_Correct>);
static_assert(has_overridable_constants_correct_type<OverridableConstants_Correct>);

// ============================================================================
// Tests for has_member_uniform_variables / has_uniform_variables_correct_type
// ============================================================================

static_assert(!has_member_uniform_variables<NoMembers>);
static_assert(!has_uniform_variables_correct_type<NoMembers>);

struct UniformVariables_WrongType {
  int uniform_variables;
};
static_assert(has_member_uniform_variables<UniformVariables_WrongType>);
static_assert(!has_uniform_variables_correct_type<UniformVariables_WrongType>);

struct UniformVariables_Correct {
  static constexpr std::array<int, 1> uniform_variables = {1};
};
static_assert(has_member_uniform_variables<UniformVariables_Correct>);
static_assert(has_uniform_variables_correct_type<UniformVariables_Correct>);

}  // namespace test
}  // namespace details
}  // namespace webgpu
}  // namespace onnxruntime
