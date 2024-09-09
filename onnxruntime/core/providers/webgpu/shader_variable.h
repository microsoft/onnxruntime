// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <sstream>
#include <set>

#include "core/framework/tensor_shape.h"

#include "core/providers/webgpu/program.h"

namespace onnxruntime {
namespace webgpu {

template <typename TIdx>
std::string GetElementAt(std::string_view var, const TIdx& idx, int rank, bool is_f16 = false) {
  // "std::string::rfind(str, 0) == 0" is equivalent to "std::string::starts_with(str)" before C++20.
  if (var.rfind("uniforms.", 0) == 0) {
    if (rank > 4) {
      if constexpr (std::is_integral_v<TIdx>) {
        if (is_f16) {
          return MakeStringWithClassicLocale(var, "[", idx / 8, "][", (idx % 8) / 4, "][", (idx % 8) % 4, "]");
        } else {
          return MakeStringWithClassicLocale(var, "[", idx / 4, "][", idx % 4, "]");
        }
      } else {
        if (is_f16) {
          return MakeStringWithClassicLocale(var, "[(", idx, ") / 8][(", idx, ") % 8 / 4][(", idx, ") % 8 % 4]");
        } else {
          return MakeStringWithClassicLocale(var, "[(", idx, ") / 4][(", idx, ") % 4]");
        }
      }
    }
  }

  return rank > 1 ? MakeStringWithClassicLocale(var, "[", idx, "]") : std::string{var};
}

class ShaderVariable {
 public:
  enum Usage : uint32_t {
    None = 0,                             // no usage. this means no additional implementation code will be generated.
    UseIndicesTypeAlias = 1,              // use type alias "{name}_indices_t" for indices (eg. u32, vec2<u32>, vec3<u32>, vec4<u32>, ...)
    UseValueTypeAlias = 2,                // use type alias "{name}_value_t" for value (eg. f32, vecT<f32>, vec4<bool>, ...)
    UseElementTypeAlias = 4,              // use type alias "{name}_element_t" for element (eg. f32, bool, ...)
    UseShapeAndStride = 16,               // use shape and stride for the variable
    UseOffsetToIndices = 32,              // use implementation of fn o2i_{name}
    UseIndicesToOffset = 64,              // use implementation of fn i2o_{name}
    UseBroadcastedIndicesToOffset = 128,  // use implementation of fn {broadcasted_result_name}_bi2o_{name}
    UseSet = 256,                         // use implementation of fn set_{name}
    UseSetByIndices = 512,                // use implementation of fn set_{name}_by_indices
    UseGet = 1024,                        // use implementation of fn get_{name}
    UseGetByIndices = 2048,               // use implementation of fn get_{name}_by_indices
    UseUniform = 32768,                   // use uniform for shape and stride
  };

  ShaderVariable(std::string_view name, ProgramVariableDataType type, Usage usage, const TensorShape& dims);

  ShaderVariable(ShaderVariable&&) = default;
  ShaderVariable& operator=(ShaderVariable&&) = default;

  // get the name of the variable.
  inline std::string_view Name() const { return name_; }

  // create a WGSL expression ({varname}_indices_t) for getting indices from offset.
  // \param offset: a WGSL expression (u32) representing the offset.
  inline std::string OffsetToIndices(std::string_view offset_expr) const;

  // create a WGSL expression (u32) for getting offset from indices.
  // \param indices: a WGSL expression ({varname}_indices_t) representing the indices.
  inline std::string IndicesToOffset(std::string_view indices_expr) const;

  // create a WGSL expression (u32) for getting original offset from broadcasted indices.
  // \param indices: a WGSL expression ({broadcasted_result_varname}_indices_t) representing the broadcasted indices.
  // \param broadcasted_result: the broadcasted result variable.
  inline std::string BroadcastedIndicesToOffset(std::string_view indices_expr, const ShaderVariable& broadcasted_result) const;

  // create a WGSL expression ({varname}_indices_t) as an indices literal
  // \param init: a list of indices values.
  template <typename... TIndices>
  inline std::string Indices(TIndices&&... indices_args) const;

  // create a WGSL statement for setting value of the specified dimension of the indices.
  // \param indices_var: name of the indices variable ({varname}_indices_t).
  // \param idx: the index (i32|u32) of the dimension to set.
  // \param value: the value (u32) to set.
  template <typename TIdx, typename TVal>
  inline std::string IndicesSet(std::string_view indices_var, const TIdx& idx_expr, const TVal& value) const;

  // create a WGSL expression (u32) for getting value of the specified dimension of the indices.
  // \param indices_var: name of the indices variable ({varname}_indices_t).
  // \param idx: the index (i32|u32) of the dimension to get.
  template <typename TIdx>
  inline std::string IndicesGet(std::string_view indices_var, const TIdx& idx_expr) const;

  // create a WGSL statement for setting data at the given indices.
  // \param args: a list of indices values (u32) followed by a value ({varname}_value_t).
  template <typename... TIndicesAndValue>
  inline std::string Set(TIndicesAndValue&&... args) const;

  // create a WGSL statement for setting data at the given indices.
  // \param indices_var: name of the indices variable ({varname}_indices_t).
  // \param value: the value ({varname}_value_t) to set.
  inline std::string SetByIndices(std::string_view indices_var, std::string_view value) const;

  // create a WGSL statement for setting data at the given offset.
  // \param offset: a WGSL expression (u32) representing the offset.
  // \param value: the value ({varname}_value_t) to set.
  template <typename TOffset, typename TValue>
  inline std::string SetByOffset(TOffset&& offset, TValue&& value) const;

  // create a WGSL expression ({varname}_value_t) for getting data at the given indices.
  // \param indices: a list of indices values (u32).
  template <typename... TIndices>
  inline std::string Get(TIndices&&... indices) const;

  // create a WGSL expression ({varname}_value_t) for getting data at the given indices.
  // \param indices_var: name of the indices variable ({varname}_indices_t).
  inline std::string GetByIndices(std::string_view indices_var) const;

  // create a WGSL expression ({varname}_value_t) for getting data at the given offset.
  // \param offset: a WGSL expression (u32) representing the offset.
  template <typename TOffset>
  inline std::string GetByOffset(TOffset&& offset) const;

 private:
  friend ShaderVariable::Usage operator|(ShaderVariable::Usage a, ShaderVariable::Usage b);
  friend ShaderVariable::Usage operator&(ShaderVariable::Usage a, ShaderVariable::Usage b);
  friend ShaderVariable::Usage& operator|=(ShaderVariable::Usage& a, ShaderVariable::Usage b);
  friend ShaderVariable::Usage& operator&=(ShaderVariable::Usage& a, ShaderVariable::Usage b);

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ShaderVariable);

  void Impl(std::ostringstream& ss) const;

  std::string GetByOffsetImpl(std::string_view offset) const;
  std::string SetByOffsetImpl(std::string_view offset, std::string_view value) const;
  std::string_view StorageType() const;
  std::string_view ValueType() const;
  std::string_view ElementType() const;
  std::string_view IndicesType() const;

  std::string name_;
  ProgramVariableDataType type_;
  int num_components_;
  int rank_;
  TensorShape dims_;

  mutable Usage usage_;
  mutable std::set<const ShaderVariable*> broadcasted_to_;

  // unlike storage/element/value type, indices type is not a string view to a constant string. so we need to store it.
  std::string indices_type_;

  // the alias for the types
  std::string value_type_alias_;
  std::string element_type_alias_;
  std::string indices_type_alias_;

  friend class ShaderHelper;
};

inline ShaderVariable::Usage operator|(ShaderVariable::Usage a, ShaderVariable::Usage b) {
  return (ShaderVariable::Usage)((uint32_t&)a | (uint32_t&)b);
}
inline ShaderVariable::Usage operator&(ShaderVariable::Usage a, ShaderVariable::Usage b) {
  return (ShaderVariable::Usage)((uint32_t&)a & (uint32_t&)b);
}
inline ShaderVariable::Usage& operator|=(ShaderVariable::Usage& a, ShaderVariable::Usage b) {
  return (ShaderVariable::Usage&)((uint32_t&)a |= (uint32_t&)b);
}
inline ShaderVariable::Usage& operator&=(ShaderVariable::Usage& a, ShaderVariable::Usage b) {
  return (ShaderVariable::Usage&)((uint32_t&)a &= (uint32_t&)b);
}

namespace detail {
template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
std::string pass_as_string(T&& v) {
  return std::to_string(std::forward<T>(v));
}
template <typename...>
std::string_view pass_as_string(std::string_view sv) {
  return sv;
}
template <typename T>
std::string pass_as_string(T&& v) {
  return std::forward<T>(v);
}
}  // namespace detail

inline std::string ShaderVariable::OffsetToIndices(std::string_view offset_expr) const {
  usage_ |= UseOffsetToIndices | UseShapeAndStride;
  return rank_ < 2 ? std::string{offset_expr}
                   : MakeStringWithClassicLocale("o2i_", name_, '(', offset_expr, ')');
}

inline std::string ShaderVariable::IndicesToOffset(std::string_view indices_expr) const {
  usage_ |= UseIndicesToOffset | UseShapeAndStride;
  return rank_ < 2 ? std::string{indices_expr}
                   : MakeStringWithClassicLocale("i2o_", name_, '(', indices_expr, ')');
}

inline std::string ShaderVariable::BroadcastedIndicesToOffset(std::string_view indices_expr, const ShaderVariable& broadcasted_result) const {
  usage_ |= UseBroadcastedIndicesToOffset | UseShapeAndStride;
  broadcasted_to_.insert(&broadcasted_result);
  return rank_ == 0
             ? "0"
             : MakeStringWithClassicLocale(broadcasted_result.name_, "_bi2o_", name_, '(', indices_expr, ')');
}

template <typename... TIndices>
inline std::string ShaderVariable::Indices(TIndices&&... indices_args) const {
  usage_ |= UseShapeAndStride;
  return rank_ == 0
             ? "0"
             : MakeStringWithClassicLocale(IndicesType(), "(",
                                           absl::StrJoin(std::forward_as_tuple(std::forward<TIndices>(indices_args)...), ", "),
                                           ')');
}

template <typename TIdx, typename TVal>
inline std::string ShaderVariable::IndicesSet(std::string_view indices_var, const TIdx& idx_expr, const TVal& value) const {
  usage_ |= UseShapeAndStride;
  return rank_ < 2 ? MakeStringWithClassicLocale(indices_var, '=', value, ';')
                   : MakeStringWithClassicLocale(GetElementAt(indices_var, idx_expr, rank_), '=', value, ';');
}

template <typename TIdx>
inline std::string ShaderVariable::IndicesGet(std::string_view indices_var, const TIdx& idx_expr) const {
  usage_ |= UseShapeAndStride;
  return rank_ < 2 ? std::string{indices_var}
                   : GetElementAt(indices_var, idx_expr, rank_);
}

template <typename TOffset, typename TValue>
inline std::string ShaderVariable::SetByOffset(TOffset&& offset, TValue&& value) const {
  return SetByOffsetImpl(detail::pass_as_string(offset), detail::pass_as_string(value));
}

template <typename... TIndicesAndValue>
inline std::string ShaderVariable::Set(TIndicesAndValue&&... args) const {
  usage_ |= UseShapeAndStride;
  ORT_ENFORCE(sizeof...(TIndicesAndValue) == rank_ + 1, "Number of arguments should be ", rank_ + 1, "(rank + 1)");
  if constexpr (sizeof...(TIndicesAndValue) == 1) {
    return SetByOffset("0", std::forward<TIndicesAndValue>(args)...);
  } else if constexpr (sizeof...(TIndicesAndValue) == 2) {
    return SetByOffset(std::forward<TIndicesAndValue>(args)...);
  } else {
    usage_ |= UseSet | UseSetByIndices | UseIndicesToOffset;
    return MakeStringWithClassicLocale("set_", name_, '(',
                                       absl::StrJoin(std::forward_as_tuple(std::forward<TIndicesAndValue>(args)...), ", "),
                                       ");");
  }
}

inline std::string ShaderVariable::SetByIndices(std::string_view indices_var, std::string_view value) const {
  usage_ |= UseShapeAndStride;
  if (rank_ < 2) {
    return SetByOffset(indices_var, value);
  } else {
    usage_ |= UseSetByIndices | UseIndicesToOffset;
    return MakeStringWithClassicLocale("set_", name_, "_by_indices(", indices_var, ", ", value, ");");
  }
}

template <typename TOffset>
inline std::string ShaderVariable::GetByOffset(TOffset&& offset) const {
  return GetByOffsetImpl(detail::pass_as_string(offset));
}

template <typename... TIndices>
inline std::string ShaderVariable::Get(TIndices&&... indices) const {
  usage_ |= UseShapeAndStride;
  ORT_ENFORCE(sizeof...(TIndices) == rank_, "Number of arguments should be ", rank_, "(rank)");
  if constexpr (sizeof...(TIndices) == 0) {
    return GetByOffset("0");
  } else if constexpr (sizeof...(TIndices) == 1) {
    return GetByOffset(std::forward<TIndices>(indices)...);
  } else {
    usage_ |= UseGet | UseGetByIndices | UseIndicesToOffset;
    return MakeStringWithClassicLocale("get_", name_, '(',
                                       absl::StrJoin(std::forward_as_tuple(std::forward<TIndices>(indices)...), ", "),
                                       ')');
  }
}

inline std::string ShaderVariable::GetByIndices(std::string_view indices_var) const {
  usage_ |= UseShapeAndStride;
  if (rank_ < 2) {
    return GetByOffset(indices_var);
  } else {
    usage_ |= UseGetByIndices | UseIndicesToOffset;
    return MakeStringWithClassicLocale("get_", name_, "_by_indices(", indices_var, ")");
  }
}

}  // namespace webgpu
}  // namespace onnxruntime
