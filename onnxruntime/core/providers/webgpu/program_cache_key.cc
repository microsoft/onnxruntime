// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/program_cache_key.h"
#include "core/providers/webgpu/configured_program.h"

#include "core/providers/webgpu/string_macros.h"

namespace onnxruntime {
namespace webgpu {

// macro "D" - append to the ostream only in debug build
#ifndef NDEBUG  // if debug build
#define D(str) << str
#else
#define D(str)
#endif

namespace {
std::string CalculateStructuredKey(const ProgramBase& program,
                                   std::span<const uint32_t> inputs_segments,
                                   std::span<const uint32_t> outputs_segments,
                                   const void* generator_type) {
  std::string key;
  key.reserve(128);
  key.push_back('\1');
  // The token identifies a static generator and its immutable metadata in this process.
  // This is not a persistent cache or an encoding of a configuration pointer.
  AppendConfigScalar(key, reinterpret_cast<uintptr_t>(generator_type));
  program.AppendSpecializationKey(key);
  AppendConfigScalar(key, program.WorkgroupSizeX());
  AppendConfigScalar(key, program.WorkgroupSizeY());
  AppendConfigScalar(key, program.WorkgroupSizeZ());
  AppendConfigScalar(key, program.SubgroupSize());
  AppendConfigScalar(key, program.IndirectDispatchTensor() != nullptr);
  AppendConfigScalar(key, program.OverridableConstants().size());
  for (const auto& value : program.OverridableConstants()) {
    AppendConfigScalar(key, value.has_value);
    if (value.has_value) {
      AppendConfigScalar(key, value.type);
      switch (value.type) {
        case ProgramConstantDataType::Float32:
          AppendConfigScalar(key, value.f32);
          break;
        case ProgramConstantDataType::Float16:
          AppendConfigScalar(key, value.f16.val);
          break;
        case ProgramConstantDataType::Int32:
          AppendConfigScalar(key, value.i32);
          break;
        case ProgramConstantDataType::Uint32:
          AppendConfigScalar(key, value.u32);
          break;
        case ProgramConstantDataType::Bool:
          AppendConfigScalar(key, value.boolean);
          break;
      }
    }
  }
  AppendConfigScalar(key, program.UniformVariables().size());
  for (const auto& uniform : program.UniformVariables()) {
    AppendConfigScalar(key, uniform.length);
    if (uniform.length != 0) AppendConfigScalar(key, uniform.data_type);
  }
  const auto append_tensor = [&](const auto& tensor, uint32_t segments, size_t owner) {
    AppendConfigScalar(key, tensor.var_type);
    AppendConfigScalar(key, tensor.tensor->GetElementType());
    const auto& shape = tensor.use_override_shape ? tensor.override_shape : tensor.tensor->Shape();
    AppendConfigScalar(key, shape.NumDimensions());
    AppendConfigScalar(key, segments);
    AppendConfigScalar(key, tensor.is_buffer_view);
    if (tensor.is_buffer_view) AppendConfigScalar(key, owner);
  };
  AppendConfigScalar(key, program.Inputs().size());
  for (size_t i = 0; i < program.Inputs().size(); ++i) {
    append_tensor(program.Inputs()[i], inputs_segments[i], program.InputBufferOwner(i));
  }
  AppendConfigScalar(key, program.Outputs().size());
  for (size_t i = 0; i < program.Outputs().size(); ++i) {
    append_tensor(program.Outputs()[i], outputs_segments[i], program.OutputBufferOwner(i));
    AppendConfigScalar(key, program.Outputs()[i].is_atomic);
  }
  AppendConfigScalar(key, program.Indices().size());
  for (const auto& indices : program.Indices()) AppendConfigScalar(key, indices.NumDimensions());
  return key;
}

// append the info of an input or output to the cachekey
void AppendTensorInfo(OStringStream& ss,
                      const TensorShape& tensor_shape,
                      ProgramVariableDataType var_type,
                      ProgramTensorMetadataDependency dependency,
                      bool& first,
                      uint32_t segments,
                      bool is_buffer_view,
                      size_t buffer_owner) {
  if (first) {
    first = false;
  } else {
    ss << '|';
  }

  if ((dependency & ProgramTensorMetadataDependency::Type) == ProgramTensorMetadataDependency::Type) {
#ifndef NDEBUG  // if debug build
    ss << var_type;
#else
    ss << static_cast<int>(var_type);
#endif
    ss << ';';
  }

  if (segments != 1) {
    ss D("Segs=") << 'S' << segments << ';';
  }
  if (is_buffer_view) {
    ss D("View=") << 'V' << buffer_owner << ';';
  }

  if ((dependency & ProgramTensorMetadataDependency::Shape) == ProgramTensorMetadataDependency::Shape) {
    ss D("Dims=") << tensor_shape.ToString();
  } else if ((dependency & ProgramTensorMetadataDependency::Rank) == ProgramTensorMetadataDependency::Rank) {
    ss D("Rank=") << tensor_shape.NumDimensions();
  }
}
}  // namespace

std::string ProgramCacheKeyForLogging(std::string_view key) {
  if (key.empty() || key[0] != '\1') return std::string{key};
  constexpr char hex[] = "0123456789abcdef";
  std::string result{"configured:"};
  for (unsigned char byte : key) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 15]);
  }
  return result;
}

std::string CalculateProgramCacheKey(const ProgramBase& program,
                                     std::span<uint32_t> inputs_segments,
                                     std::span<uint32_t> outputs_segments) {
  if (const void* generator_type = program.StructuredKeyType()) {
    return CalculateStructuredKey(program, inputs_segments, outputs_segments, generator_type);
  }
  SS(ss, kStringInitialSizeCacheKey);

  // final key format:
  // <KEY>=<PROGRAM_NAME>[<CUSTOM_CACHE_HINT>]:<WORKGROUP_SIZE>:<SUBGROUP_SIZE>:<UNIFORMS>:<INPUTS_INFO>
  //
  // <CUSTOM_CACHE_HINT> = <HINT_0>|<HINT_1>|...
  // <WORKGROUP_SIZE>    = <X_IF_OVERRIDDEN>,<Y_IF_OVERRIDDEN>,<Z_IF_OVERRIDDEN>
  // <SUBGROUP_SIZE>     = <SUBGROUP_SIZE_IF_OVERRIDDEN>
  // <UNIFORMS>          = <UNIFORMS_INFO_0>|<UNIFORMS_INFO_1>|...
  // <UNIFORMS_INFO_i>   = <UNIFORM_LENGTH>
  // <INPUTS_INFO>       = <INPUTS_INFO_0>|<INPUTS_INFO_1>|...
  // <INPUTS_INFO_i>     = <TENSOR_ELEMENT_TYPE_OR_EMPTY>;<TENSOR_SEGMENTS_OR_EMPTY>;<TENSOR_SHAPE_OR_RANK_OR_EMPTY>
  ss << program.Name();

  // append custom cache hint if any
  if (auto& hint = program.CacheHint(); !hint.empty()) {
    ss << '[' D("CacheHint=") << hint << ']';
  }

  // append workgroup size if overridden
  if (auto x = program.WorkgroupSizeX(), y = program.WorkgroupSizeY(), z = program.WorkgroupSizeZ();
      x != 0 || y != 0 || z != 0) {
    ss << ":" D("WorkgroupSize=");
    // only append non-zero values. zero values are considered as use default
    if (x > 0) {
      ss << x;
    }
    ss << ",";
    if (y > 0) {
      ss << y;
    }
    ss << ",";
    if (z > 0) {
      ss << z;
    }
  }

  // append the requested subgroup size (subgroup-size-control) if any
  if (auto subgroup_size = program.SubgroupSize(); subgroup_size != 0) {
    ss << ":" D("SubgroupSize=") << subgroup_size;
  }

  ss << ":" D("UniformSizes=");
  bool first = true;
  for (const auto& uniform : program.UniformVariables()) {
    if (first) {
      first = false;
    } else {
      ss << "|";
    }
    if (uniform.length > 0) {
      ss << uniform.length;
    }
  }

  ss << ":" D("Inputs=");
  first = true;
  for (size_t i = 0; i < program.Inputs().size(); i++) {
    const auto& input = program.Inputs()[i];
    AppendTensorInfo(ss,
                     input.use_override_shape ? input.override_shape : input.tensor->Shape(),
                     input.var_type,
                     input.dependency,
                     first,
                     inputs_segments[i],
                     input.is_buffer_view,
                     program.InputBufferOwner(i));
  }

  ss << ":" D("Outputs=");
  first = true;
  for (size_t i = 0; i < program.Outputs().size(); i++) {
    const auto& output = program.Outputs()[i];
    AppendTensorInfo(ss,
                     output.use_override_shape ? output.override_shape : output.tensor->Shape(),
                     output.var_type,
                     output.dependency,
                     first,
                     outputs_segments[i],
                     output.is_buffer_view,
                     program.OutputBufferOwner(i));
  }

  if (!program.Indices().empty()) {
    ss << ":" D("Indices=");
    first = true;
    for (const auto& indices_shape : program.Indices()) {
      if (first) {
        first = false;
      } else {
        ss << '|';
      }
      ss D("Rank=") << indices_shape.NumDimensions();
    }
  }

  return SS_GET(ss);
}

}  // namespace webgpu
}  // namespace onnxruntime
