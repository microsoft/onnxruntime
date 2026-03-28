// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/tensor/space_to_depth.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {

#define WEBGPU_SPACE_TO_DEPTH_VERSIONED_KERNEL(start, end, domain, is_nhwc) \
  ONNX_OPERATOR_VERSIONED_KERNEL_EX(                                        \
      SpaceToDepth,                                                         \
      domain,                                                               \
      start,                                                                \
      end,                                                                  \
      kWebGpuExecutionProvider,                                             \
      (*KernelDefBuilder::Create())                                         \
          .TypeConstraint("T", WebGpuSupportedFloatTypes()),                \
      SpaceToDepth<is_nhwc>);

#define WEBGPU_SPACE_TO_DEPTH_KERNEL(version, domain, is_nhwc) \
  ONNX_OPERATOR_KERNEL_EX(                                     \
      SpaceToDepth,                                            \
      domain,                                                  \
      version,                                                 \
      kWebGpuExecutionProvider,                                \
      (*KernelDefBuilder::Create())                            \
          .TypeConstraint("T", WebGpuSupportedFloatTypes()),   \
      SpaceToDepth<is_nhwc>);

WEBGPU_SPACE_TO_DEPTH_VERSIONED_KERNEL(1, 12, kOnnxDomain, false)
WEBGPU_SPACE_TO_DEPTH_KERNEL(13, kOnnxDomain, false)

WEBGPU_SPACE_TO_DEPTH_VERSIONED_KERNEL(1, 12, kMSInternalNHWCDomain, true)
WEBGPU_SPACE_TO_DEPTH_KERNEL(13, kMSInternalNHWCDomain, true)

Status SpaceToDepthProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const ShaderVariableHelper& input = shader.AddInput("input");
  const ShaderVariableHelper& output = shader.AddOutput("output");

  // Generate permutation function inline
  OStringStream& os = shader.AdditionalImplementation();
  os << "fn perm(i: input_indices_t) -> input_indices_t {\n"
     << "  var a: input_indices_t;\n";
  for (int idx = 0; idx < input.Rank(); ++idx) {
    os << "  " << input.IndicesSet("a", std::to_string(perm_[idx]), "i[" + std::to_string(idx) + "]") << "\n";
  }
  os << "  return a;\n"
     << "}\n";

  shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                            << "  let indices = " << output.OffsetToIndices("global_idx") << ";\n"
                            << "  let aIndices = perm(indices);\n"
                            << "  " << output.SetByOffset("global_idx", input.GetByIndices("aIndices"));

  return Status::OK();
}

template <bool is_nhwc>
Status SpaceToDepth<is_nhwc>::ComputeInternal(onnxruntime::webgpu::ComputeContext& context) const {
  const auto* input = context.Input(0);
  const TensorShape input_shape = input->Shape();
  int64_t input_rank = input_shape.NumDimensions();
  ORT_ENFORCE(input_rank == 4, "Input must be rank 4.");

  int64_t n, c, h, w;
  int64_t shape[6];
  int64_t perm[6];
  if (is_nhwc) {
    // NHWC: input is [N, H, W, C]
    n = input_shape[0];
    h = input_shape[1];
    w = input_shape[2];
    c = input_shape[3];
    ORT_ENFORCE(h % blocksize_ == 0, "H must be divisible by blocksize.");
    ORT_ENFORCE(w % blocksize_ == 0, "W must be divisible by blocksize.");

    // Reshape to [N, H/b, b, W/b, b, C]
    // Output 6D: [N, H/b, W/b, b, b, C] => flattened [N, H/b, W/b, C*b*b]
    // output[n, h', w', bh, bw, c] reads from input[n, h', bh, w', bw, c]
    // perm: {0, 1, 3, 2, 4, 5}
    int64_t shape_values[] = {n, h / blocksize_, blocksize_, w / blocksize_, blocksize_, c};
    int64_t perm_values[] = {0, 1, 3, 2, 4, 5};
    std::copy(shape_values, shape_values + 6, shape);
    std::copy(perm_values, perm_values + 6, perm);
  } else {
    // NCHW: input is [N, C, H, W]
    n = input_shape[0];
    c = input_shape[1];
    h = input_shape[2];
    w = input_shape[3];
    ORT_ENFORCE(h % blocksize_ == 0, "H must be divisible by blocksize.");
    ORT_ENFORCE(w % blocksize_ == 0, "W must be divisible by blocksize.");

    // Reshape to [N, C, H/b, b, W/b, b]
    // ONNX transpose perm for NCHW SpaceToDepth: {0, 3, 5, 1, 2, 4}
    // Output 6D shape: [N, b, b, C, H/b, W/b] => flattened [N, C*b*b, H/b, W/b]
    int64_t shape_values[] = {n, c, h / blocksize_, blocksize_, w / blocksize_, blocksize_};
    int64_t perm_values[] = {0, 3, 5, 1, 2, 4};
    std::copy(shape_values, shape_values + 6, shape);
    std::copy(perm_values, perm_values + 6, perm);
  }

  std::vector<int64_t> shape_vec(shape, shape + 6);
  TensorShape input_override_shape(shape_vec);

  // Calculate the final 4D output shape
  int64_t output_shape[4];
  if (is_nhwc) {
    int64_t output_shape_values[] = {n, h / blocksize_, w / blocksize_, c * blocksize_ * blocksize_};
    std::copy(output_shape_values, output_shape_values + 4, output_shape);
  } else {
    int64_t output_shape_values[] = {n, c * blocksize_ * blocksize_, h / blocksize_, w / blocksize_};
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

  SpaceToDepthProgram program{perm};
  program
      .AddInput({input, ProgramTensorMetadataDependency::TypeAndRank, input_override_shape, 1})
      .AddOutput({output, ProgramTensorMetadataDependency::None, output_override_shape, 1})
      .SetDispatchGroupSize((output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .CacheHint(absl::StrJoin(input_shape.GetDims(), "-"), blocksize_)
      .AddUniformVariable({static_cast<uint32_t>(output_size)});
  return context.RunProgram(program);
}

}  // namespace webgpu
}  // namespace onnxruntime
