// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>
#include <iosfwd>

#include <absl/strings/str_join.h>

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace webgpu {
class ShaderHelper;
class ComputeContext;
class WebGpuContext;

// data type of uniform variable
enum class ProgramUniformVariableDataType {
  Float32,
  Float16,
  Uint32,
  Int32,
};
std::ostream& operator<<(std::ostream& os, ProgramUniformVariableDataType);

constexpr size_t ProgramUniformVariableDataTypeSize[] = {sizeof(float), sizeof(uint16_t), sizeof(uint32_t), sizeof(int32_t)};

constexpr std::string_view ProgramUniformVariableDataTypeName[] = {"f32", "f16", "u32", "i32"};

// represents a runtime value of a uniform variable
struct ProgramUniformVariableValue {
  ProgramUniformVariableValue();  // representing an empty uniform variable
  ProgramUniformVariableValue(float value);
  ProgramUniformVariableValue(uint32_t value);
  ProgramUniformVariableValue(int32_t value);
  ProgramUniformVariableValue(MLFloat16 value);
  ProgramUniformVariableValue(gsl::span<const float> values);
  ProgramUniformVariableValue(gsl::span<const uint32_t> values);
  ProgramUniformVariableValue(gsl::span<const int32_t> values);
  ProgramUniformVariableValue(gsl::span<const MLFloat16> values);

  size_t length;
  ProgramUniformVariableDataType data_type;
  std::vector<uint8_t> data;

 private:
  ProgramUniformVariableValue(ProgramUniformVariableDataType data_type, const void* ptr, size_t element_byte_size, size_t length = 1);
};

// represents a uniform variable definition
struct ProgramUniformVariableDefinition {
  constexpr ProgramUniformVariableDefinition(std::string_view name, ProgramUniformVariableDataType data_type)
      : name{name}, data_type{data_type} {}

  std::string_view name;
  ProgramUniformVariableDataType data_type;
};

// data type of constant
enum class ProgramConstantDataType {
  Float32,
  Float16,
  Uint32,
  Int32,
  Bool
};
std::ostream& operator<<(std::ostream& os, ProgramConstantDataType);

constexpr std::string_view ProgramConstantDataTypeName[] = {"f32", "f16", "u32", "i32", "bool"};

// represents a constant in a program
struct ProgramConstant {
  constexpr ProgramConstant(std::string_view name, float value) : name{name}, type{ProgramConstantDataType::Float32}, f32{value} {}
  constexpr ProgramConstant(std::string_view name, uint32_t value) : name{name}, type{ProgramConstantDataType::Uint32}, u32{value} {}
  constexpr ProgramConstant(std::string_view name, int32_t value) : name{name}, type{ProgramConstantDataType::Int32}, i32{value} {}
  constexpr ProgramConstant(std::string_view name, MLFloat16 value) : name{name}, type{ProgramConstantDataType::Float16}, f16{value} {}
  constexpr ProgramConstant(std::string_view name, bool value) : name{name}, type{ProgramConstantDataType::Bool}, boolean{value} {}

  std::string_view name;
  ProgramConstantDataType type;
  union {
    float f32;
    uint32_t u32;
    int32_t i32;
    MLFloat16 f16;
    bool boolean;
  };
};

// represents a runtime value of an overridable constant
struct ProgramOverridableConstantValue {
  constexpr ProgramOverridableConstantValue() : type{}, u32{}, has_value{false} {}  // representing not overriding
  constexpr ProgramOverridableConstantValue(float value) : type{ProgramConstantDataType::Float32}, f32{value}, has_value{true} {}
  constexpr ProgramOverridableConstantValue(uint32_t value) : type{ProgramConstantDataType::Uint32}, u32{value}, has_value{true} {}
  constexpr ProgramOverridableConstantValue(int32_t value) : type{ProgramConstantDataType::Int32}, i32{value}, has_value{true} {}
  constexpr ProgramOverridableConstantValue(MLFloat16 value) : type{ProgramConstantDataType::Float16}, f16{value}, has_value{true} {}
  constexpr ProgramOverridableConstantValue(bool value) : type{ProgramConstantDataType::Bool}, boolean{value}, has_value{true} {}

  ProgramConstantDataType type;
  union {
    float f32;
    uint32_t u32;
    int32_t i32;
    MLFloat16 f16;
    bool boolean;
  };
  bool has_value;
};

// represents an overridable constant definition. may or may not have a default value.
struct ProgramOverridableConstantDefinition {
  constexpr ProgramOverridableConstantDefinition(std::string_view name, ProgramConstantDataType type)
      : name{name}, type{type}, u32{}, has_default_value{false} {}
  constexpr ProgramOverridableConstantDefinition(std::string_view name, float value)
      : name{name}, type{ProgramConstantDataType::Float32}, f32{value}, has_default_value{true} {}
  constexpr ProgramOverridableConstantDefinition(std::string_view name, uint32_t value)
      : name{name}, type{ProgramConstantDataType::Uint32}, u32{value}, has_default_value{true} {}
  constexpr ProgramOverridableConstantDefinition(std::string_view name, int32_t value)
      : name{name}, type{ProgramConstantDataType::Int32}, i32{value}, has_default_value{true} {}
  constexpr ProgramOverridableConstantDefinition(std::string_view name, MLFloat16 value)
      : name{name}, type{ProgramConstantDataType::Float16}, f16{value}, has_default_value{true} {}
  constexpr ProgramOverridableConstantDefinition(std::string_view name, bool value)
      : name{name}, type{ProgramConstantDataType::Bool}, boolean{value}, has_default_value{true} {}

  std::string_view name;
  ProgramConstantDataType type;
  union {
    float f32;
    uint32_t u32;
    int32_t i32;
    MLFloat16 f16;
    bool boolean;
  };
  bool has_default_value;
};

// represents whether the program shader depends on the type, rank, or shape of an input/output tensor
enum class ProgramTensorMetadataDependency : int {
  None = 0,
  Type = 1,
  Rank = 2,
  Shape = 4,
  TypeAndRank = Type | Rank,
  TypeAndShape = Type | Shape,
};
std::ostream& operator<<(std::ostream& os, ProgramTensorMetadataDependency);

inline ProgramTensorMetadataDependency operator|(ProgramTensorMetadataDependency a, ProgramTensorMetadataDependency b) {
  return (ProgramTensorMetadataDependency)((int&)a | (int&)b);
}
inline ProgramTensorMetadataDependency operator&(ProgramTensorMetadataDependency a, ProgramTensorMetadataDependency b) {
  return (ProgramTensorMetadataDependency)((int&)a & (int&)b);
}
inline ProgramTensorMetadataDependency& operator|=(ProgramTensorMetadataDependency& a, ProgramTensorMetadataDependency b) {
  return (ProgramTensorMetadataDependency&)((int&)a |= (int&)b);
}
inline ProgramTensorMetadataDependency& operator&=(ProgramTensorMetadataDependency& a, ProgramTensorMetadataDependency b) {
  return (ProgramTensorMetadataDependency&)((int&)a &= (int&)b);
}

constexpr SafeInt<uint32_t> WORKGROUP_SIZE = 64;

// represents the scope of a variable in a shader program.
//
// this is not a full list of all possible variable scopes in shader programs.
// it only includes what are used in WebGPU EP.
enum class ProgramVariableScope {
  Input = 0,   // storage buffer variable with access mode "read"
  Output = 1,  // storage buffer variable with access mode "read_write"
  Local = 2,   // local variable

  Count  // should always be the last element
};

// data type of variable
//
// this is not a full list of all possible data types in shader programs.
// it only includes what are used in WebGPU EP.
enum class ProgramVariableDataType {
  InvalidType = -1,
  Float32,
  Vec2Float32,
  Vec4Float32,
  Float16,
  Vec2Float16,
  Vec4Float16,
  Int32,
  Vec2Int32,
  Vec4Int32,
  Uint32,
  Vec2Uint32,
  Vec4Uint32,
  Int64,
  Uint64,
  Vec4Bool,
};
#ifndef NDEBUG
std::ostream& operator<<(std::ostream& os, ProgramVariableDataType);
#endif

int NumberOfComponents(ProgramVariableDataType type);

ProgramVariableDataType ToProgramVariableDataType(int32_t element_type, int component = 1);

struct ProgramInput {
  ProgramInput(const Tensor* tensor)
      : ProgramInput{tensor, ProgramTensorMetadataDependency::TypeAndRank} {}
  ProgramInput(const Tensor* tensor, ProgramTensorMetadataDependency dependency, int component = 1)
      : tensor{tensor},
        dependency{dependency},
        var_type{ToProgramVariableDataType(tensor->GetElementType(), component)},
        use_override_shape{false},
        override_shape{} {}
  ProgramInput(const Tensor* tensor, ProgramTensorMetadataDependency dependency, const TensorShape& override_shape, int component)
      : tensor{tensor},
        dependency{dependency},
        var_type{ToProgramVariableDataType(tensor->GetElementType(), component)},
        use_override_shape{true},
        override_shape{override_shape} {}

  const Tensor* tensor;
  ProgramTensorMetadataDependency dependency;
  ProgramVariableDataType var_type;
  bool use_override_shape;
  TensorShape override_shape;
};

struct ProgramOutput {
  ProgramOutput(Tensor* tensor)
      : ProgramOutput{tensor, ProgramTensorMetadataDependency::None} {}
  ProgramOutput(Tensor* tensor, ProgramTensorMetadataDependency dependency, int component = 1)
      : tensor{tensor},
        dependency{dependency},
        var_type{ToProgramVariableDataType(tensor->GetElementType(), component)},
        use_override_shape{false},
        override_shape{} {}
  ProgramOutput(Tensor* tensor, ProgramTensorMetadataDependency dependency, const TensorShape& override_shape, int component)
      : tensor{tensor},
        dependency{dependency},
        var_type{ToProgramVariableDataType(tensor->GetElementType(), component)},
        use_override_shape{true},
        override_shape{override_shape} {}

  Tensor* tensor;
  ProgramTensorMetadataDependency dependency;
  ProgramVariableDataType var_type;
  bool use_override_shape;
  TensorShape override_shape;
};

namespace detail {
class ProgramWrapper;
}

struct ProgramMetadata;

class ProgramBase {
 public:
  //
  // chain-style methods for setting properties
  //

  // set the cache hint for the program
  template <typename T>
  ProgramBase& CacheHint(T&& hint) {
    cache_hint_ = std::forward<T>(hint);
    return *this;
  }

  // append a program input
  ProgramBase& Input(ProgramInput&& input);
  // append multiple program inputs
  ProgramBase& Inputs(std::initializer_list<ProgramInput> inputs);
  // append a program output
  ProgramBase& Output(ProgramOutput&& output);
  // set one or more program outputs
  ProgramBase& Outputs(std::initializer_list<ProgramOutput> outputs);

  // set the size of dispatch groups. Y and Z are 1 if not specified.
  ProgramBase& DispatchGroupSize(uint32_t x);
  // set the size of dispatch groups. Z is 1 if not specified.
  ProgramBase& DispatchGroupSize(uint32_t x, uint32_t y);
  // set the size of dispatch groups.
  ProgramBase& DispatchGroupSize(uint32_t x, uint32_t y, uint32_t z);

  // set the size of a workgroup grid. Y and Z are 1 if not specified.
  ProgramBase& WorkgroupSize(uint32_t x);
  // set the size of a workgroup grid. Z is 1 if not specified.
  ProgramBase& WorkgroupSize(uint32_t x, uint32_t y);
  // set the size of a workgroup grid.
  ProgramBase& WorkgroupSize(uint32_t x, uint32_t y, uint32_t z);

  // append a uniform variable.
  //
  // the specified uniform variable should match the uniform definition in the class,
  // specified by macro WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES.
  ProgramBase& UniformVariable(ProgramUniformVariableValue&& variable);
  // append multiple uniform variables.
  //
  // the specified uniform variables should match the uniform definition in the class,
  // specified by macro WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES.
  ProgramBase& UniformVariables(std::initializer_list<ProgramUniformVariableValue> variables);

  // set the overridable constants
  //
  // the specified overridable constants should match the overridable constant definition in the class,
  // specified by macro WEBGPU_PROGRAM_DEFINE_OVERRIDABLE_CONSTANTS.
  ProgramBase& OverridableConstants(std::initializer_list<ProgramOverridableConstantValue> overridable_constants);

  //
  // shader code generation
  //

  virtual Status GenerateShaderCode(ShaderHelper& shader) const = 0;

  //
  // abstract methods for getting metadata
  //
  // A derived class may contain any of the following static members:
  //
  // \code{.cpp}
  //   // define a list of constant that used in the shader program
  //   static constexpr const ProgramConstant constants[] = { ... };
  //
  //   // define a list of overridable constant that used in the shader program
  //   static constexpr const ProgramOverridableConstantDefinition overridable_constants[] = { ... };
  //
  //   // define a list of uniform variables that used in the shader program
  //   static constexpr const ProgramUniformVariableDefinition uniform_variables[] = { ... };
  // \endcode
  //
  // If those static members exist, the value of them will be used to generate the metadata.
  virtual ProgramMetadata GetMetadata() const = 0;

  //
  // Properties Getters
  //

  inline const std::string& Name() const { return name_; }
  inline const std::string& CacheHint() const { return cache_hint_; }
  inline const std::vector<ProgramInput>& Inputs() const { return inputs_; }
  inline const std::vector<ProgramOutput>& Outputs() const { return outputs_; }
  inline uint32_t DispatchGroupSizeX() const { return dispatch_group_size_x_; }
  inline uint32_t DispatchGroupSizeY() const { return dispatch_group_size_y_; }
  inline uint32_t DispatchGroupSizeZ() const { return dispatch_group_size_z_; }
  inline uint32_t WorkgroupSizeX() const { return workgroup_size_x_; }
  inline uint32_t WorkgroupSizeY() const { return workgroup_size_y_; }
  inline uint32_t WorkgroupSizeZ() const { return workgroup_size_z_; }
  inline const std::vector<ProgramUniformVariableValue>& UniformVariables() const { return variables_; }
  inline const std::vector<ProgramOverridableConstantValue>& OverridableConstants() const { return overridable_constants_; }

 protected:
  virtual ~ProgramBase() = default;

 private:
  // Make the constructor private to prevent direct instantiation or inheritance from this class
  // Use the Program template class as base class to create a new program class
  explicit ProgramBase(const std::string& name);

  std::string name_;
  std::string cache_hint_;
  std::vector<ProgramInput> inputs_;
  std::vector<ProgramOutput> outputs_;

  uint32_t dispatch_group_size_x_;
  uint32_t dispatch_group_size_y_;
  uint32_t dispatch_group_size_z_;

  uint32_t workgroup_size_x_;
  uint32_t workgroup_size_y_;
  uint32_t workgroup_size_z_;

  std::vector<ProgramUniformVariableValue> variables_;
  std::vector<ProgramOverridableConstantValue> overridable_constants_;

  friend class detail::ProgramWrapper;
};

namespace detail {
// class ProgramWrapper is for accessing private constructor of ProgramBase.
// only ProgramWrapper can access the constructor of ProgramBase because ProgramWrapper is the only friend class of
// ProgramBase. This design is used to prevent direct instantiation or inheritance from ProgramBase.
class ProgramWrapper : public ProgramBase {
 protected:
  template <typename... Args>
  ProgramWrapper(Args&&... args) : ProgramBase{std::forward<Args>(args)...} {}
};

#if defined(ORT_WEBGPU_REGISTER_DERIVED_PROGRAM_CLASS_TYPE_CHECK)
#error "macro ORT_WEBGPU_REGISTER_DERIVED_PROGRAM_CLASS_TYPE_CHECK is already defined"
#endif

#define ORT_WEBGPU_REGISTER_DERIVED_PROGRAM_CLASS_TYPE_CHECK(identifier, element_type)                                                   \
 private:                                                                                                                                \
  template <typename U>                                                                                                                  \
  static auto test_has_##identifier(int)->decltype(U::identifier, std::true_type{}); /* checks if member exists */                       \
  template <typename...>                                                                                                                 \
  static auto test_has_##identifier(...)->std::false_type;                                                                               \
                                                                                                                                         \
  template <typename U,                                                                       /* The following type check uses SFINAE */ \
            typename = std::enable_if_t<                                                      /* to ensure the specific member:       */ \
                                        is_const_std_array<decltype(U::identifier)>::value && /*  - is a const std::array             */ \
                                        std::is_const_v<decltype(U::identifier)> &&           /*  - has "const" modifier              */ \
                                        !std::is_member_pointer_v<decltype(&U::identifier)>>> /*  - is static                         */ \
  static auto test_has_##identifier##_with_correct_type(int)->std::true_type;                                                            \
  template <typename...>                                                                                                                 \
  static auto test_has_##identifier##_with_correct_type(...)->std::false_type;                                                           \
                                                                                                                                         \
 public:                                                                                                                                 \
  static constexpr bool has_##identifier = decltype(test_has_##identifier<T>(0))::value;                                                 \
  static constexpr bool has_##identifier##_with_correct_type = decltype(test_has_##identifier##_with_correct_type<T>(0))::value

// the following template class checks whether the type is a const std::array
template <typename T>
struct is_const_std_array : std::false_type {};
template <typename T, size_t N>
struct is_const_std_array<const std::array<T, N>> : std::true_type {};

// the following template class checks whether certain static members exist in the derived class (SFINAE)
template <typename T>
class DerivedProgramClassTypeCheck {
  ORT_WEBGPU_REGISTER_DERIVED_PROGRAM_CLASS_TYPE_CHECK(constants, ProgramConstant);
  ORT_WEBGPU_REGISTER_DERIVED_PROGRAM_CLASS_TYPE_CHECK(overridable_constants, ProgramOverridableConstantDefinition);
  ORT_WEBGPU_REGISTER_DERIVED_PROGRAM_CLASS_TYPE_CHECK(uniform_variables, ProgramUniformVariableDefinition);
};

// compile-time tests for the type check
//
// TODO: move this to test folder
namespace test {

template <typename T>
class TestTypeCheck {
  ORT_WEBGPU_REGISTER_DERIVED_PROGRAM_CLASS_TYPE_CHECK(a, int);
};

struct TestClass_Empty {};
static_assert(!TestTypeCheck<TestClass_Empty>::has_a);
static_assert(!TestTypeCheck<TestClass_Empty>::has_a_with_correct_type);

struct TestClass_NotArray_0 {
  int b;
};
static_assert(!TestTypeCheck<TestClass_NotArray_0>::has_a);
static_assert(!TestTypeCheck<TestClass_NotArray_0>::has_a_with_correct_type);

struct TestClass_NotArray_1 {
  int a;
};
static_assert(TestTypeCheck<TestClass_NotArray_1>::has_a);
static_assert(!TestTypeCheck<TestClass_NotArray_1>::has_a_with_correct_type);

struct TestClass_NotArray_2 {
  const int a;
};
static_assert(TestTypeCheck<TestClass_NotArray_2>::has_a);
static_assert(!TestTypeCheck<TestClass_NotArray_2>::has_a_with_correct_type);

struct TestClass_NotStdArray_0 {
  const int a[2];
};
static_assert(TestTypeCheck<TestClass_NotStdArray_0>::has_a);
static_assert(!TestTypeCheck<TestClass_NotStdArray_0>::has_a_with_correct_type);

struct TestClass_NotStdArray_1 {
  static constexpr int a[] = {0};
};
static_assert(TestTypeCheck<TestClass_NotStdArray_1>::has_a);
static_assert(!TestTypeCheck<TestClass_NotStdArray_1>::has_a_with_correct_type);

struct TestClass_NotStdArray_2 {
  static int a[];
};
static_assert(TestTypeCheck<TestClass_NotStdArray_2>::has_a);
static_assert(!TestTypeCheck<TestClass_NotStdArray_2>::has_a_with_correct_type);

struct TestClass_NotStdArray_3 {
  static const int a[];
};
static_assert(TestTypeCheck<TestClass_NotStdArray_3>::has_a);
static_assert(!TestTypeCheck<TestClass_NotStdArray_3>::has_a_with_correct_type);

struct TestClass_StdArray_0 {
  std::array<int, 1> a = {1};
};
static_assert(TestTypeCheck<TestClass_StdArray_0>::has_a);
static_assert(!TestTypeCheck<TestClass_StdArray_0>::has_a_with_correct_type);

struct TestClass_StdArray_1 {
  static constexpr std::array<int, 2> a = {1, 2};
};
static_assert(TestTypeCheck<TestClass_StdArray_1>::has_a);
static_assert(TestTypeCheck<TestClass_StdArray_1>::has_a_with_correct_type);

struct TestClass_StdArray_2 {
  static const std::array<int, 3> a;
};
static_assert(TestTypeCheck<TestClass_StdArray_2>::has_a);
static_assert(TestTypeCheck<TestClass_StdArray_2>::has_a_with_correct_type);

struct TestClass_StdArray_3 {
  static constexpr const std::array<int, 4> a = {1, 2, 3, 4};
};
static_assert(TestTypeCheck<TestClass_StdArray_3>::has_a);
static_assert(TestTypeCheck<TestClass_StdArray_3>::has_a_with_correct_type);

struct TestClass_StdArray_4 {
  static std::array<int, 5> a;
};
static_assert(TestTypeCheck<TestClass_StdArray_4>::has_a);
static_assert(!TestTypeCheck<TestClass_StdArray_4>::has_a_with_correct_type);

}  // namespace test

#undef ORT_WEBGPU_REGISTER_DERIVED_PROGRAM_CLASS_TYPE_CHECK

}  // namespace detail

struct ProgramMetadata {
  gsl::span<const ProgramConstant> constants;
  gsl::span<const ProgramOverridableConstantDefinition> overridable_constants;
  gsl::span<const ProgramUniformVariableDefinition> uniform_variables;
};

template <typename T>
class Program : public detail::ProgramWrapper {
 public:
  template <typename... Args>
  Program(Args&&... args) : detail::ProgramWrapper{std::forward<Args>(args)...} {}

  virtual ProgramMetadata GetMetadata() const final {
    ProgramMetadata metadata;
    if constexpr (detail::DerivedProgramClassTypeCheck<T>::has_constants) {
      constexpr const ProgramConstant* ptr = T::constants.data();
      constexpr size_t len = T::constants.size();

      static_assert(detail::DerivedProgramClassTypeCheck<T>::has_constants_with_correct_type,
                    "Derived class of \"Program\" has member \"constants\" but its type is incorrect. "
                    "Please use macro WEBGPU_PROGRAM_DEFINE_CONSTANTS() or WEBGPU_PROGRAM_EXTEND_CONSTANTS() to declare constants.");

      metadata.constants = {ptr, len};
    } else {
      metadata.constants = {};
    }

    if constexpr (detail::DerivedProgramClassTypeCheck<T>::has_overridable_constants) {
      constexpr const ProgramOverridableConstantDefinition* ptr = T::overridable_constants.data();
      constexpr size_t len = T::overridable_constants.size();

      static_assert(detail::DerivedProgramClassTypeCheck<T>::has_overridable_constants_with_correct_type,
                    "Derived class of \"Program\" has member \"overridable_constants\" but its type is incorrect. "
                    "Please use macro WEBGPU_PROGRAM_DEFINE_OVERRIDABLE_CONSTANTS() or WEBGPU_PROGRAM_EXTEND_OVERRIDABLE_CONSTANTS() to declare overridable constants.");

      metadata.overridable_constants = {ptr, len};
    } else {
      metadata.overridable_constants = {};
    }

    if constexpr (detail::DerivedProgramClassTypeCheck<T>::has_uniform_variables) {
      constexpr const ProgramUniformVariableDefinition* ptr = T::uniform_variables.data();
      constexpr size_t len = T::uniform_variables.size();

      static_assert(detail::DerivedProgramClassTypeCheck<T>::has_uniform_variables_with_correct_type,
                    "Derived class of \"Program\" has member \"uniform_variables\" but its type is incorrect. "
                    "Please use macro WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES() or WEBGPU_PROGRAM_EXTEND_UNIFORM_VARIABLES() to declare uniform variables.");

      metadata.uniform_variables = {ptr, len};
    } else {
      metadata.uniform_variables = {};
    }

    return metadata;
  }
};

namespace detail {
// helper function to convert a C-style array to std::array
//
// This is basically the same as std::to_array in C++20.
//
template <typename T, size_t N, size_t... Idx>
constexpr auto _to_std_array_impl(T (&arr)[N], std::index_sequence<Idx...>) -> std::array<std::remove_cv_t<T>, N> {
  return {{arr[Idx]...}};
}

template <typename T, size_t N>
constexpr auto _to_std_array(T (&arr)[N]) -> std::array<std::remove_cv_t<T>, N> {
  return _to_std_array_impl(arr, std::make_index_sequence<N>{});
}

// helper function to concatenate a std::array and a C-style array to a std::array
//
template <typename T, size_t L, size_t... IdxL, size_t R, size_t... IdxR>
constexpr std::array<std::remove_cv_t<T>, L + R> _concat2_impl(const std::array<T, L>& lhs,
                                                               T (&rhs)[R],
                                                               std::index_sequence<IdxL...>,
                                                               std::index_sequence<IdxR...>) {
  return {{lhs[IdxL]..., rhs[IdxR]...}};
}

template <typename T, size_t L, size_t R>
constexpr std::array<std::remove_cv_t<T>, L + R> _concat2(const std::array<T, L>& lhs, T (&rhs)[R]) {
  return _concat2_impl(lhs, rhs, std::make_index_sequence<L>{}, std::make_index_sequence<R>{});
}

}  // namespace detail
#define WEBGPU_PROGRAM_DEFINE_(identifier, T, ...)             \
  static constexpr const T identifier##_own[] = {__VA_ARGS__}; \
  static constexpr const auto identifier =                     \
      onnxruntime::webgpu::detail::_to_std_array(identifier##_own)

#define WEBGPU_PROGRAM_EXTEND_(identifier, T, BASE, ...)       \
  static constexpr const T identifier##_own[] = {__VA_ARGS__}; \
  static constexpr const auto identifier =                     \
      onnxruntime::webgpu::detail::_concat2(BASE::identifier, identifier##_own)

#define WEBGPU_PROGRAM_DEFINE_CONSTANTS(...) \
  WEBGPU_PROGRAM_DEFINE_(constants, onnxruntime::webgpu::ProgramConstant, __VA_ARGS__)

#define WEBGPU_PROGRAM_EXTEND_CONSTANTS(BASE, ...) \
  WEBGPU_PROGRAM_EXTEND_(constants, onnxruntime::webgpu::ProgramConstant, BASE, __VA_ARGS__)

#define WEBGPU_PROGRAM_DEFINE_OVERRIDABLE_CONSTANTS(...) \
  WEBGPU_PROGRAM_DEFINE_(overridable_constants, onnxruntime::webgpu::ProgramOverridableConstantDefinition, __VA_ARGS__)

#define WEBGPU_PROGRAM_EXTEND_OVERRIDABLE_CONSTANTS(BASE, ...) \
  WEBGPU_PROGRAM_EXTEND_(overridable_constants, onnxruntime::webgpu::ProgramOverridableConstantDefinition, BASE, __VA_ARGS__)

#define WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(...) \
  WEBGPU_PROGRAM_DEFINE_(uniform_variables, onnxruntime::webgpu::ProgramUniformVariableDefinition, __VA_ARGS__)

#define WEBGPU_PROGRAM_EXTEND_UNIFORM_VARIABLES(BASE, ...) \
  WEBGPU_PROGRAM_EXTEND_(uniform_variables, onnxruntime::webgpu::ProgramUniformVariableDefinition, BASE, __VA_ARGS__)

}  // namespace webgpu
}  // namespace onnxruntime
