// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/program_cache_key.h"

#include "core/providers/webgpu/shader_macros.h"

namespace onnxruntime {
namespace webgpu {

std::string CalculateProgramCacheKey(const ProgramBase& program, bool is_1d_dispatch) {
  std::ostringstream ss;
  ss.imbue(std::locale::classic());

  // final key format:
  // <KEY>=<PROGRAM_NAME>[<CUSTOM_CACHE_HINT>]:<WORKGROUP_SIZE>:<DISPATCH_FLAG>:<UNIFORMS>:<INPUTS_INFO>
  //
  // <CUSTOM_CACHE_HINT> = <HINT_0>|<HINT_1>|...
  // <WORKGROUP_SIZE>    = <X_IF_OVERRIDED>,<Y_IF_OVERRIDED>,<Z_IF_OVERRIDED>
  // <DISPATCH_FLAG>     = <!IS_1D_DISPATCH>
  // <UNIFORMS>          = <UNIFORMS_INFO_0>|<UNIFORMS_INFO_1>|...
  // <UNIFORMS_INFO_i>   = <UNIFORM_LENGTH>
  // <INPUTS_INFO>       = <INPUTS_INFO_0>|<INPUTS_INFO_1>|...
  // <INPUTS_INFO_i>     = <TENSOR_ELEMENT_TYPE_OR_EMPTY>;<TENSOR_SHAPE_OR_RANK_OR_EMPTY>
  ss << program.Name();

  // append custom cache hint if any
  if (auto& hint = program.CacheHint(); !hint.empty()) {
    ss << "[" D("CacheHint=") << hint << "]";
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

  ss << ":" D("DispatchDim=") << (is_1d_dispatch ? "1" : "3");
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
  for (const auto& input : program.Inputs()) {
    if (first) {
      first = false;
    } else {
      ss << "|";
    }
    if ((input.dependency & ProgramInputTensorDependency::Type) == ProgramInputTensorDependency::Type) {
#ifndef NDEBUG  // if debug build
      ss << DataTypeImpl::ToString(input.tensor->DataType());
#else
      ss << input.tensor->GetElementType();
#endif
    }
    ss << ";";
    if ((input.dependency & ProgramInputTensorDependency::Rank) == ProgramInputTensorDependency::Rank) {
      ss D("Rank=") << input.tensor->Shape().NumDimensions();
    } else if ((input.dependency & ProgramInputTensorDependency::Shape) == ProgramInputTensorDependency::Shape) {
      ss D("Dims=") << input.tensor->Shape().ToString();
    }
  }

  return ss.str();
}

}  // namespace webgpu
}  // namespace onnxruntime
