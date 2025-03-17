// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/program_cache_key.h"

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
// append the info of an input or output to the cachekey
void AppendTensorInfo(std::ostream& ss, const Tensor& tensor, ProgramVariableDataType var_type, ProgramTensorMetadataDependency dependency,
                      bool& first) {
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

  if ((dependency & ProgramTensorMetadataDependency::Shape) == ProgramTensorMetadataDependency::Shape) {
    ss D("Dims=") << tensor.Shape().ToString();
  } else if ((dependency & ProgramTensorMetadataDependency::Rank) == ProgramTensorMetadataDependency::Rank) {
    ss D("Rank=") << tensor.Shape().NumDimensions();
  }
}
}  // namespace

std::string CalculateProgramCacheKey(const ProgramBase& program, bool is_1d_dispatch) {
  SS(ss, kStringInitialSizeCacheKey);

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
    AppendTensorInfo(ss, *input.tensor, input.var_type, input.dependency, first);
  }

  ss << ":" D("Outputs=");
  first = true;
  for (const auto& output : program.Outputs()) {
    AppendTensorInfo(ss, *output.tensor, output.var_type, output.dependency, first);
  }

  return SS_GET(ss);
}

}  // namespace webgpu
}  // namespace onnxruntime
