// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <array>
#include <string>
#include <vector>
#include <iosfwd>

#ifdef _MSC_VER
#pragma warning(push)
// C4702: unreachable code
#pragma warning(disable : 4702)
#endif  // _MSC_VER

#include <absl/strings/str_join.h>

#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/tensor.h"

#include "core/providers/webgpu/string_utils.h"

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
OStringStream& operator<<(OStringStream& os, ProgramUniformVariableDataType);

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
OStringStream& operator<<(OStringStream& os, ProgramConstantDataType);

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
OStringStream& operator<<(OStringStream& os, ProgramTensorMetadataDependency);

inline ProgramTensorMetadataDependency operator|(ProgramTensorMetadataDependency a, ProgramTensorMetadataDependency b) {
  return static_cast<ProgramTensorMetadataDependency>(static_cast<int>(a) | static_cast<int>(b));
}
inline ProgramTensorMetadataDependency operator&(ProgramTensorMetadataDependency a, ProgramTensorMetadataDependency b) {
  return static_cast<ProgramTensorMetadataDependency>(static_cast<int>(a) & static_cast<int>(b));
}
inline ProgramTensorMetadataDependency& operator|=(ProgramTensorMetadataDependency& a, ProgramTensorMetadataDependency b) {
  return a = a | b;
}
inline ProgramTensorMetadataDependency& operator&=(ProgramTensorMetadataDependency& a, ProgramTensorMetadataDependency b) {
  return a = a & b;
}

constexpr SafeInt<uint32_t> WORKGROUP_SIZE = 64;

// data type of variable
//
// this is not a full list of all possible data types in shader programs.
// it only includes what are used in WebGPU EP.
enum class ProgramVariableDataType {
  InvalidType = -1,
  Float32,
  Float32x2,
  Float32x4,
  Float16,
  Float16x2,
  Float16x4,
  Int32,
  Int32x2,
  Int32x4,
  Uint32,
  Uint32x2,
  Uint32x4,
  Int64,
  Uint64,
  Boolx4,
  Uint8x4,
  Uint8x8,
  Uint8x16,
  Int8x4,
  Int8x8,
  Int8x16,
  Uint4x8,
  Int4x8,
  // if you add a new type here, you also need to update ProgramVariableDataTypeName
};
#ifndef NDEBUG
OStringStream& operator<<(OStringStream& os, ProgramVariableDataType);
#endif

int NumberOfComponents(ProgramVariableDataType type);

ProgramVariableDataType ToProgramVariableDataType(int32_t element_type, int component = 1);

struct ProgramInput {
 private:
  struct FlattenTag {};

 public:
  constexpr static const FlattenTag Flatten{};

  ProgramInput(const Tensor* tensor);
  ProgramInput(const Tensor* tensor, ProgramTensorMetadataDependency dependency, int component = 1);
  ProgramInput(const Tensor* tensor, ProgramTensorMetadataDependency dependency, FlattenTag, int component = 1);
  ProgramInput(const Tensor* tensor, ProgramTensorMetadataDependency dependency, const TensorShape& override_shape, int component);

  const Tensor* tensor;
  ProgramTensorMetadataDependency dependency;
  ProgramVariableDataType var_type;
  bool use_override_shape;
  TensorShape override_shape;
};

struct ProgramOutput {
 private:
  struct AtomicTag {};
  struct FlattenTag {};

 public:
  constexpr static const AtomicTag Atomic{};
  constexpr static const FlattenTag Flatten{};

  ProgramOutput(Tensor* tensor);
  ProgramOutput(Tensor* tensor, ProgramTensorMetadataDependency dependency, int component = 1);
  ProgramOutput(Tensor* tensor, ProgramTensorMetadataDependency dependency, AtomicTag);
  ProgramOutput(Tensor* tensor, ProgramTensorMetadataDependency dependency, const TensorShape& override_shape, int component);
  ProgramOutput(Tensor* tensor, ProgramTensorMetadataDependency dependency, FlattenTag, int component = 1);

  Tensor* tensor;
  ProgramTensorMetadataDependency dependency;
  ProgramVariableDataType var_type;
  bool is_atomic;
  bool use_override_shape;
  TensorShape override_shape;
};

enum class ValidationMode {
  Disabled = 0,
  WGPUOnly,
  Basic,
  Full
};
std::ostream& operator<<(std::ostream& os, ValidationMode mode);

namespace details {
class ProgramWrapper;
}

struct ProgramMetadata {
  gsl::span<const ProgramConstant> constants;
  gsl::span<const ProgramOverridableConstantDefinition> overridable_constants;
  gsl::span<const ProgramUniformVariableDefinition> uniform_variables;
};

class ProgramBase {
 public:
  //
  // chain-style methods for setting properties
  //

  // set the cache hint for the program
  template <typename... T>
  ProgramBase& CacheHint(T&&... hints) {
    cache_hint_ = absl::StrJoin(std::forward_as_tuple(std::forward<T>(hints)...), "|");
    return *this;
  }

  // add a program input
  ProgramBase& AddInput(ProgramInput&& input);
  // add multiple program inputs
  ProgramBase& AddInputs(std::initializer_list<ProgramInput> inputs);
  // add a program output
  ProgramBase& AddOutput(ProgramOutput&& output);
  // add multiple program outputs
  ProgramBase& AddOutputs(std::initializer_list<ProgramOutput> outputs);
  // add a program variable for indices
  template <typename... Args>
  ProgramBase& AddIndices(Args&&... args) {
    indices_.emplace_back(std::forward<Args>(args)...);
    return *this;
  }

  // set the size of dispatch groups. Y and Z are 1 if not specified.
  ProgramBase& SetDispatchGroupSize(uint32_t x);
  // set the size of dispatch groups. Z is 1 if not specified.
  ProgramBase& SetDispatchGroupSize(uint32_t x, uint32_t y);
  // set the size of dispatch groups.
  ProgramBase& SetDispatchGroupSize(uint32_t x, uint32_t y, uint32_t z);

  // set indirect dispatch tensor for indirect dispatch
  ProgramBase& SetIndirectDispatchTensor(const Tensor* indirect_dispatch_tensor);

  // set the size of a workgroup grid. Y and Z are 1 if not specified.
  ProgramBase& SetWorkgroupSize(uint32_t x);
  // set the size of a workgroup grid. Z is 1 if not specified.
  ProgramBase& SetWorkgroupSize(uint32_t x, uint32_t y);
  // set the size of a workgroup grid.
  ProgramBase& SetWorkgroupSize(uint32_t x, uint32_t y, uint32_t z);

  // add a uniform variable.
  //
  // the specified uniform variable should match the uniform definition in the class,
  // specified by macro WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES.
  ProgramBase& AddUniformVariable(ProgramUniformVariableValue&& variable);
  // add multiple uniform variables.
  //
  // the specified uniform variables should match the uniform definition in the class,
  // specified by macro WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES.
  ProgramBase& AddUniformVariables(std::initializer_list<ProgramUniformVariableValue> variables);

  // set the overridable constants
  //
  // the specified overridable constants should match the overridable constant definition in the class,
  // specified by macro WEBGPU_PROGRAM_DEFINE_OVERRIDABLE_CONSTANTS.
  ProgramBase& SetOverridableConstants(std::initializer_list<ProgramOverridableConstantValue> overridable_constants);

  //
  // shader code generation
  //

  virtual Status GenerateShaderCode(ShaderHelper& shader) const = 0;

  //
  // Properties Getters
  //

  inline const std::string& Name() const { return name_; }
  inline const ProgramMetadata& Metadata() const { return metadata_; }
  inline const std::string& CacheHint() const { return cache_hint_; }
  inline const std::vector<ProgramInput>& Inputs() const { return inputs_; }
  inline const std::vector<ProgramOutput>& Outputs() const { return outputs_; }
  inline const std::vector<TensorShape>& Indices() const { return indices_; }
  inline uint32_t DispatchGroupSizeX() const { return dispatch_group_size_x_; }
  inline uint32_t DispatchGroupSizeY() const { return dispatch_group_size_y_; }
  inline uint32_t DispatchGroupSizeZ() const { return dispatch_group_size_z_; }
  inline const Tensor* IndirectDispatchTensor() const { return indirect_dispatch_tensor_; }
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
  explicit ProgramBase(std::string_view name, ProgramMetadata&& metadata);

  std::string name_;
  ProgramMetadata metadata_;

  std::string cache_hint_;
  std::vector<ProgramInput> inputs_;
  std::vector<ProgramOutput> outputs_;
  std::vector<TensorShape> indices_;

  uint32_t dispatch_group_size_x_;
  uint32_t dispatch_group_size_y_;
  uint32_t dispatch_group_size_z_;

  const Tensor* indirect_dispatch_tensor_;

  uint32_t workgroup_size_x_;
  uint32_t workgroup_size_y_;
  uint32_t workgroup_size_z_;

  std::vector<ProgramUniformVariableValue> variables_;
  std::vector<ProgramOverridableConstantValue> overridable_constants_;

  friend class details::ProgramWrapper;
};

namespace details {
// class ProgramWrapper is for accessing private constructor of ProgramBase.
// only ProgramWrapper can access the constructor of ProgramBase because ProgramWrapper is the only friend class of
// ProgramBase. This design is used to prevent direct instantiation or inheritance from ProgramBase.
class ProgramWrapper : public ProgramBase {
 protected:
  template <typename... Args>
  ProgramWrapper(Args&&... args) : ProgramBase{std::forward<Args>(args)...} {}
};

// the following template class checks whether the type is a const std::array
template <typename T>
struct is_const_std_array : std::false_type {};
template <typename T, size_t N>
struct is_const_std_array<const std::array<T, N>> : std::true_type {};

// The following variable templates check whether certain static members exist in the derived class.
// Uses std::void_t with decltype(T::member) for SFINAE-based detection of named static data members.

template <typename T, typename = void>
inline constexpr bool has_member_constants = false;
template <typename T>
inline constexpr bool has_member_constants<T, std::void_t<decltype(T::constants)>> = true;

template <typename T, typename = void>
inline constexpr bool has_member_overridable_constants = false;
template <typename T>
inline constexpr bool has_member_overridable_constants<T, std::void_t<decltype(T::overridable_constants)>> = true;

template <typename T, typename = void>
inline constexpr bool has_member_uniform_variables = false;
template <typename T>
inline constexpr bool has_member_uniform_variables<T, std::void_t<decltype(T::uniform_variables)>> = true;

// C++20 concepts for checking whether the member has the correct type (static const std::array).

template <typename T>
concept has_constants_correct_type = requires {
  T::constants;
  requires is_const_std_array<decltype(T::constants)>::value;
  requires std::is_const_v<decltype(T::constants)>;
  requires !std::is_member_pointer_v<decltype(&T::constants)>;
};

template <typename T>
concept has_overridable_constants_correct_type = requires {
  T::overridable_constants;
  requires is_const_std_array<decltype(T::overridable_constants)>::value;
  requires std::is_const_v<decltype(T::overridable_constants)>;
  requires !std::is_member_pointer_v<decltype(&T::overridable_constants)>;
};

template <typename T>
concept has_uniform_variables_correct_type = requires {
  T::uniform_variables;
  requires is_const_std_array<decltype(T::uniform_variables)>::value;
  requires std::is_const_v<decltype(T::uniform_variables)>;
  requires !std::is_member_pointer_v<decltype(&T::uniform_variables)>;
};

}  // namespace details

template <typename T>
class Program : public details::ProgramWrapper {
 public:
  template <typename... Args>
  Program(Args&&... args) : details::ProgramWrapper{std::forward<Args>(args)..., GetMetadata()} {}

  static ProgramMetadata GetMetadata() {
    ProgramMetadata metadata;
    if constexpr (details::has_member_constants<T>) {
      static_assert(details::has_constants_correct_type<T>,
                    "Derived class of \"Program\" has member \"constants\" but its type is incorrect. "
                    "Please use macro WEBGPU_PROGRAM_DEFINE_CONSTANTS() or WEBGPU_PROGRAM_EXTEND_CONSTANTS() to declare constants.");

      constexpr const ProgramConstant* ptr = T::constants.data();
      constexpr size_t len = T::constants.size();

      metadata.constants = {ptr, len};
    } else {
      metadata.constants = {};
    }

    if constexpr (details::has_member_overridable_constants<T>) {
      static_assert(details::has_overridable_constants_correct_type<T>,
                    "Derived class of \"Program\" has member \"overridable_constants\" but its type is incorrect. "
                    "Please use macro WEBGPU_PROGRAM_DEFINE_OVERRIDABLE_CONSTANTS() or WEBGPU_PROGRAM_EXTEND_OVERRIDABLE_CONSTANTS() to declare overridable constants.");

      constexpr const ProgramOverridableConstantDefinition* ptr = T::overridable_constants.data();
      constexpr size_t len = T::overridable_constants.size();

      metadata.overridable_constants = {ptr, len};
    } else {
      metadata.overridable_constants = {};
    }

    if constexpr (details::has_member_uniform_variables<T>) {
      static_assert(details::has_uniform_variables_correct_type<T>,
                    "Derived class of \"Program\" has member \"uniform_variables\" but its type is incorrect. "
                    "Please use macro WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES() or WEBGPU_PROGRAM_EXTEND_UNIFORM_VARIABLES() to declare uniform variables.");

      constexpr const ProgramUniformVariableDefinition* ptr = T::uniform_variables.data();
      constexpr size_t len = T::uniform_variables.size();

      metadata.uniform_variables = {ptr, len};
    } else {
      metadata.uniform_variables = {};
    }

    return metadata;
  }
};

namespace details {
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

}  // namespace details
#define WEBGPU_PROGRAM_DEFINE_(identifier, T, ...)             \
  static constexpr const T identifier##_own[] = {__VA_ARGS__}; \
  static constexpr const auto identifier =                     \
      std::to_array(identifier##_own)

#define WEBGPU_PROGRAM_EXTEND_(identifier, T, BASE, ...)       \
  static constexpr const T identifier##_own[] = {__VA_ARGS__}; \
  static constexpr const auto identifier =                     \
      onnxruntime::webgpu::details::_concat2(BASE::identifier, identifier##_own)

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
