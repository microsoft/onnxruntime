// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/span_utils.h"
#include "core/common/inlined_containers.h"
#include "core/providers/cpu/tensor/utils.h"
#include "core/providers/webgpu/tensor/transpose.h"
#include "core/providers/webgpu/shader_variable.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace {
bool AreSpansEqual(gsl::span<const size_t> a, gsl::span<const size_t> b) {
  if (a.size() != b.size()) {
    return false;
  }

  return std::equal(a.begin(), a.end(), b.begin());
}

auto SqueezeShape(const gsl::span<const int64_t>& shape,
                  const gsl::span<const size_t>& adjusted_perm,
                  onnxruntime::TensorShapeVector& new_shape,
                  onnxruntime::TensorShapeVector& new_perm) {
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] != 1) {
      new_shape.push_back(shape[i]);
    }
    if (shape[adjusted_perm[i]] != 1) {
      new_perm.push_back(adjusted_perm[i]);
    }
  }
};
}  // namespace

namespace onnxruntime {
namespace webgpu {
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    1, 12,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Transpose);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    13, 20,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Transpose);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    21, 22,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Transpose);

ONNX_OPERATOR_KERNEL_EX(
    Transpose,
    kOnnxDomain,
    23,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedNumberTypes()),
    Transpose);

Status OIHW2OHWIProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& src = shader.AddInput("src", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  return WGSL_TEMPLATE_APPLY(shader, "tensor/oihw_to_ohwi.wgsl.template",
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(src, src));
}

Status TransposeProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& input = shader.AddInput("a", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias);
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseIndicesTypeAlias | ShaderUsage::UseValueTypeAlias);

  if (use_shared_) {
    shader.AdditionalImplementation() << "var<workgroup> tile : array<array<output_value_t, tile_size + 1>, tile_size>;\n";
    shader.MainFunctionBody() << "  let stride = (uniforms.output_shape[1] - 1) / tile_size + 1;\n"
                                 "  let workgroup_id_x = workgroup_idx % stride;\n"
                                 "  let workgroup_id_y = workgroup_idx / stride;\n"
                                 "  let input_col = workgroup_id_y * tile_size + local_id.x;\n"
                                 "  let input_row = workgroup_id_x * tile_size + local_id.y;\n"
                                 "  if (input_row < uniforms.a_shape[0] && input_col < uniforms.a_shape[1]) {\n"
                              << "    tile[local_id.y][local_id.x] = " << input.GetByIndices("a_indices_t(input_row, input_col)") << ";\n"
                              << "  }\n"
                                 "  workgroupBarrier();\n"
                                 "  let output_col = workgroup_id_x * tile_size + local_id.x;\n"
                                 "  let output_row = workgroup_id_y * tile_size + local_id.y;\n"
                                 "  if (output_row < uniforms.output_shape[0] && output_col < uniforms.output_shape[1]) {\n"
                              << "    " << output.SetByIndices("output_indices_t(output_row, output_col)", "tile[local_id.x][local_id.y]") << "\n"
                              << "  }";
  } else {
    shader.AdditionalImplementation() << "fn perm(i: output_indices_t)->a_indices_t {\n"
                                         "  var a: a_indices_t;\n";
    for (size_t i = 0; i < perm_.size(); ++i) {
      shader.AdditionalImplementation() << "  a[" << perm_[i] << "] = i[" << i << "];\n";
    }
    shader.AdditionalImplementation() << "  return a;\n"
                                         "}\n";

    shader.MainFunctionBody() << shader.GuardAgainstOutOfBoundsWorkgroupSizes("uniforms.output_size")
                              << "  let indices = " << output.OffsetToIndices("global_idx")
                              << ";\n"
                                 "  let x_indices = perm(indices);\n"
                                 "  "
                              << output.SetByOffset("global_idx", input.GetByIndices("x_indices"));
  }
  return Status::OK();
}

Status Transpose::DoTranspose(onnxruntime::webgpu::ComputeContextBase& context,
                              gsl::span<const size_t> permutations,
                              const Tensor& input, Tensor& output) {
  const auto& input_shape = input.Shape();
  const auto& input_dims = input_shape.GetDims();
  int32_t rank = static_cast<int32_t>(input_shape.NumDimensions());
  TensorShapeVector output_dims(rank);

  for (int32_t i = 0; i < rank; i++) {
    output_dims[i] = input_dims[permutations[i]];
  }
  TensorShape output_shape(output_dims);

  // Check if `OIHW2OHWIProgram` can be applied.
  //
  // `OIHW2OHWIProgram` was originally designed to transpose 4D weights from OIHW
  // to OHWI format, utilizing workgroup tiling to maximize bandwidth through
  // coalesced reads and writes. While variable names reflect this origin for
  // simplicity, the shader is now generalized for broader use, supporting any
  // permutation equivalent to {0, 2, 3, 1}.
  //
  // TODO: Extend support to 2D and 3D transpositions.
  if (AreSpansEqual(permutations, AsSpan<const size_t>({0, 2, 3, 1}))) {
    const uint32_t channel_output = onnxruntime::narrow<uint32_t>(input_shape[0]);
    const uint32_t channel_input = onnxruntime::narrow<uint32_t>(input_shape[1]);
    const uint32_t kernel_height = onnxruntime::narrow<uint32_t>(input_shape[2]);
    const uint32_t kernel_width = onnxruntime::narrow<uint32_t>(input_shape[3]);

    // Calculate tiling for the input channel dimension (tiled by 64)
    const uint32_t input_channel_tiles = CeilDiv(channel_input, 64u);
    const uint32_t dispatch_size = channel_output * input_channel_tiles;

    // Threshold check: Only apply if the workload is large enough to saturate
    // GPU compute units. For small tensors, the overhead of the transpose
    // outweighs the gain.
    if (dispatch_size >= 128u) {
      OIHW2OHWIProgram transpose_program{};
      transpose_program.SetWorkgroupSize(64);
      transpose_program.SetDispatchGroupSize(dispatch_size);
      transpose_program.AddInput({&input,
                                  ProgramTensorMetadataDependency::TypeAndRank});
      transpose_program.AddOutput({&output,
                                   ProgramTensorMetadataDependency::TypeAndRank});
      transpose_program.AddUniformVariables({{channel_output},
                                             {channel_input},
                                             {kernel_height},
                                             {kernel_width},
                                             {input_channel_tiles},
                                             {CeilDiv(kernel_height * kernel_width, 4u)}});
      return context.RunProgram(transpose_program);
    }
  }

  TensorShapeVector new_shape{};
  TensorShapeVector new_perm{};
  SqueezeShape(input_shape.GetDims(), permutations, new_shape, new_perm);
  const bool channels_last = new_perm == TensorShapeVector({2, 3, 1});
  const bool channels_first = new_perm == TensorShapeVector({3, 1, 2});
  const bool use_shared = (new_shape.size() == 2 && new_perm[0] > new_perm[1]) || channels_last || channels_first;
  auto new_input_shape = input_shape;

  if (use_shared) {
    new_input_shape = channels_last
                          ? TensorShape({new_shape[0], new_shape[1] * new_shape[2]})
                      : channels_first
                          ? TensorShape({new_shape[0] * new_shape[1], new_shape[2]})
                          : new_shape;
    output_shape = TensorShape({new_input_shape[1], new_input_shape[0]});
  }

  uint32_t output_size = onnxruntime::narrow<uint32_t>(input_shape.Size());
  TransposeProgram program{permutations, use_shared};

  program
      .CacheHint(absl::StrJoin(permutations, "-"))
      .AddInputs({{&input, ProgramTensorMetadataDependency::TypeAndRank, new_input_shape, 1}})
      .AddOutputs({{&output, ProgramTensorMetadataDependency::None, output_shape, 1}})
      .AddUniformVariables({{output_size}});

  if (use_shared) {
    program.SetWorkgroupSize(TILE_SIZE, TILE_SIZE, 1);
    program.SetDispatchGroupSize(static_cast<uint32_t>((output_shape[1] + TILE_SIZE - 1) / TILE_SIZE),
                                 static_cast<uint32_t>(((output_shape[0] + TILE_SIZE - 1) / TILE_SIZE)));
  } else {
    program.SetWorkgroupSize(64u);

    uint32_t dispatch_x = CeilDiv(output_size, 64u);
    uint32_t dispatch_y = 1;
    uint32_t dispatch_z = 1;

    // This temporary workaround addresses a significant performance bottleneck
    // (~12x slower) for the input shape (1280, 2560, 3, 3) due to an issue with Intel's
    // GPU drivers. We manually normalize the dispatch group size to restore
    // performance.
    //
    // TODO: Revert this change once the driver issue is fixed.
    if (context.AdapterInfo().vendor == std::string_view{"intel"} && rank == 4) {
      uint32_t dispatch_size = dispatch_x;
      dispatch_x = 4;
      dispatch_y = 8;
      dispatch_z = CeilDiv(dispatch_size, dispatch_x * dispatch_y);
    }
    program.SetDispatchGroupSize(dispatch_x, dispatch_y, dispatch_z);
  }
  return context.RunProgram(program);
}

Status Transpose::ComputeInternal(ComputeContext& context) const {
  const auto* input_tensor = context.Input(0);
  const TensorShape& input_shape = input_tensor->Shape();
  int32_t rank = static_cast<int32_t>(input_shape.NumDimensions());

  TensorShapeVector output_dims(rank);
  InlinedVector<size_t> default_perm(rank);
  const InlinedVector<size_t>* p_perm = nullptr;
  ORT_RETURN_IF_ERROR(ComputeOutputShape(*input_tensor, output_dims, default_perm, p_perm));
  TensorShape output_shape(output_dims);
  auto* output_tensor = context.Output(0, output_shape);

  int64_t output_size = output_shape.Size();
  if (output_size == 0) {
    return Status::OK();
  }

  return DoTranspose(context, *p_perm, *input_tensor, *output_tensor);
}

}  // namespace webgpu
}  // namespace onnxruntime
