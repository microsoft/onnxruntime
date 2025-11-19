// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/tensor/depth_to_space.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_DEPTH_TO_SPACE_VERSIONED_KERNEL(start, end, domain, is_nhwc) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                        \
      DepthToSpace,                                                         \
      domain,                                                               \
      start,                                                                \
      end,                                                                  \
      kWebGpuExecutionProvider,                                             \
      (*KernelDefBuilder::Create())                                         \
          .TypeConstraint("T", WebGpuSupportedFloatTypes()),                \
      DepthToSpace<is_nhwc>);

#define WEBGPU_DEPTH_TO_SPACE_KERNEL(version, domain, is_nhwc) \
  ONNX_OPERATOR_KERNEL_EX(                                     \
      DepthToSpace,                                            \
      domain,                                                  \
      version,                                                 \
      kWebGpuExecutionProvider,                                \
      (*KernelDefBuilder::Create())                            \
          .TypeConstraint("T", WebGpuSupportedFloatTypes()),   \
      DepthToSpace<is_nhwc>);

WEBGPU_DEPTH_TO_SPACE_VERSIONED_KERNEL(11, 12, kOnnxDomain, false)
WEBGPU_DEPTH_TO_SPACE_KERNEL(13, kOnnxDomain, false)

WEBGPU_DEPTH_TO_SPACE_VERSIONED_KERNEL(11, 12, kMSInternalNHWCDomain, true)
WEBGPU_DEPTH_TO_SPACE_KERNEL(13, kMSInternalNHWCDomain, true)

void AppendPermFunction(std::ostream& os, const ShaderVariableHelper& input, const int64_t* perm) {
  os << "fn perm(i: input_indices_t) -> input_indices_t {\n"
     << "  var a: input_indices_t;\n";
  for (int idx = 0; idx < input.Rank(); ++idx) {
    os << "  " << input.IndicesSet("a", std::to_string(perm[idx]), "i[" + std::to_string(idx) + "]") << "\n";
  }
  os << "  return a;\n"
     << "}\n";
}

Status DepthToSpaceProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input = shader.AddInput("input");
  const ShaderVariableHelper& output = shader.AddOutput("output");

  AppendPermFunction(shader.AdditionalImplementation(), input, perm_);

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  let indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "  let aIndices = perm(indices);\n"
                            << "  " << output.SetByOffset("global_idx", input.GetByIndices("aIndices"));

  return Status::OK();
}

template <bool is_nhwc>
Status DepthToSpace<is_nhwc>::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* input = context.Input(0);
  const TensorShape input_shape = input->Shape();
  int64_t input_rank = input_shape.NumDimensions();
  ORT_ENFORCE(input_rank == 4, "Input must be rank 4.");

  int64_t n, c, h, w;
  int64_t shape[6];
  int64_t perm[6];
  if (is_nhwc) {
    n = input_shape[0];
    h = input_shape[1];
    w = input_shape[2];
    c = input_shape[3];

    if (is_dcr_) {
      int64_t shape_values[] = {n, h, w, blocksize_, blocksize_, c / (blocksize_ * blocksize_)};
      int64_t perm_values[] = {0, 1, 3, 2, 4, 5};
      std::copy(shape_values, shape_values + 6, shape);
      std::copy(perm_values, perm_values + 6, perm);
    } else {
      int64_t shape_values[] = {n, h, w, c / (blocksize_ * blocksize_), blocksize_, blocksize_};
      int64_t perm_values[] = {0, 1, 4, 2, 5, 3};
      std::copy(shape_values, shape_values + 6, shape);
      std::copy(perm_values, perm_values + 6, perm);
    }
  } else {
    n = input_shape[0];
    h = input_shape[2];
    w = input_shape[3];
    c = input_shape[1];

    if (is_dcr_) {
      int64_t shape_values[] = {n, blocksize_, blocksize_, c / (blocksize_ * blocksize_), h, w};
      int64_t perm_values[] = {0, 3, 4, 1, 5, 2};
      std::copy(shape_values, shape_values + 6, shape);
      std::copy(perm_values, perm_values + 6, perm);
    } else {
      int64_t shape_values[] = {n, c / (blocksize_ * blocksize_), blocksize_, blocksize_, h, w};
      int64_t perm_values[] = {0, 1, 4, 2, 5, 3};
      std::copy(shape_values, shape_values + 6, shape);
      std::copy(perm_values, perm_values + 6, perm);
    }
  }

  std::vector<int64_t> shape_vec(shape, shape + 6);
  TensorShape input_override_shape(shape_vec);

  // Calculate the final 4D output shape
  int64_t output_shape[4];
  if (is_nhwc) {
    int64_t output_shape_values[] = {n, h * blocksize_, w * blocksize_, c / (blocksize_ * blocksize_)};
    std::copy(output_shape_values, output_shape_values + 4, output_shape);
  } else {
    int64_t output_shape_values[] = {n, c / (blocksize_ * blocksize_), h * blocksize_, w * blocksize_};
    std::copy(output_shape_values, output_shape_values + 4, output_shape);
  }
  TensorShape final_output_shape(gsl::make_span(output_shape));

  auto* output = context.Output(0, final_output_shape);
  int64_t output_size = output->Shape().Size();

  if (output_size == 0) {
    return Status::OK();
  }

  std::vector<int64_t> shape_after_permutation_vec(6);
  for (int i = 0; i < 6; i++) {
    shape_after_permutation_vec[i] = shape[perm[i]];
  }
  TensorShape output_override_shape(shape_after_permutation_vec);

  DepthToSpaceProgram program{perm};
  program
      .AddInput({input, ProgramTensorMetadataDependency::TypeAndRank, input_override_shape, 1})
      .AddOutput({output, ProgramTensorMetadataDependency::None, output_override_shape, 1})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(absl::StrJoin(input_shape.GetDims(), "-"), blocksize_, is_dcr_ ? "DCR" : "CRD")
      .AddUniformVariable({static_cast<uint32_t>(output_size)});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime