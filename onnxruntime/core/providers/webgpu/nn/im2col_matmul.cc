// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <string>
#include <utility>
#include <vector>

#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/nn/im2col_matmul.h"
#include "core/providers/webgpu/nn/conv.h"
#include "core/providers/webgpu/nn/activation_util.h"

namespace onnxruntime {
namespace webgpu {
namespace {
// Chooses the optimal tile size (M, N) for the im2col operation.
// This tile size is performance-tuned and varies depending on the target device.
std::pair<uint32_t, uint32_t> ChooseTileSize(uint32_t im2col_m, uint32_t im2col_n) {
  // Define a list of preferred (tile_m, tile_n) pairs in descending order of preference.
  const std::vector<std::pair<uint32_t, uint32_t>> kTileSizes = {
      std::make_pair(32, 64),
      std::make_pair(16, 64),
  };

  for (const auto& tile_pair : kTileSizes) {
    const uint32_t tile_m = tile_pair.first;
    const uint32_t tile_n = tile_pair.second;

    const uint32_t dispatch_m = CeilDiv(im2col_m, tile_m);
    const uint32_t dispatch_n = CeilDiv(im2col_n, tile_n);
    const uint32_t dispatch = dispatch_m * dispatch_n;

    if (dispatch >= 128) {
      return tile_pair;
    }
  }

  // If none of the tile sizes meet the dispatch >=128 requirement,
  return kTileSizes.back();
}

// Add support for more devices.
bool IsDeviceSupported(const ComputeContextBase& context) {
  const wgpu::AdapterInfo& adapter_info = context.AdapterInfo();

  if (adapter_info.vendor == std::string_view("intel")) {
    if (adapter_info.architecture == std::string_view("xe-2lpg")) {
      return true;
    }
  }

  return false;
}

}  // namespace

Status Im2ColMatMulProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& src = shader.AddInput("src", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  const auto& weight = shader.AddInput("weight", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);

  ORT_ENFORCE(tile_m_ == 16 || tile_m_ == 32, "tile_m must be 16 or 32.");
  ORT_ENFORCE(tile_n_ == 64, "tile_n must be 64.");

  return WGSL_TEMPLATE_APPLY(shader, "nn/im2col_matmul.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                             WGSL_TEMPLATE_PARAMETER(tile_m, tile_m_),
                             WGSL_TEMPLATE_PARAMETER(tile_n, tile_n_),
                             WGSL_TEMPLATE_PARAMETER(use_subgroup, use_subgroup_),
                             WGSL_TEMPLATE_VARIABLE(output, output),
                             WGSL_TEMPLATE_VARIABLE(src, src),
                             WGSL_TEMPLATE_VARIABLE(weight, weight));
}

Status ApplyIm2ColMatMulProgram(ComputeContext& context,
                                bool is_channels_last,
                                const std::vector<uint32_t>& dilations,
                                const std::vector<uint32_t>& pads,
                                const std::vector<uint32_t>& strides,
                                Tensor* output) {
  const auto* src = context.Input<Tensor>(0);
  const auto* weight = context.Input<Tensor>(1);
  const bool has_bias = context.InputCount() > 2;
  const auto* bias = has_bias ? context.Input<Tensor>(2) : nullptr;

  TensorShape weight_shape = weight->Shape();
  const uint32_t channel_output = onnxruntime::narrow<uint32_t>(weight_shape[0]);
  const uint32_t channel_input = onnxruntime::narrow<uint32_t>(weight_shape[1]);
  const uint32_t kernel_height = onnxruntime::narrow<uint32_t>(weight_shape[2]);
  const uint32_t kernel_width = onnxruntime::narrow<uint32_t>(weight_shape[3]);

  // Transpose OIHW Weight to OHWI
  // TODO: Use prepack
  Tensor ohwi_weight;
  ORT_RETURN_IF_ERROR(TransposeKernel(context, weight, weight->Shape(), &ohwi_weight, {0, 2, 3, 1}));

  // im2col-matmul
  const TensorShape src_shape = src->Shape();
  const TensorShape output_shape = output->Shape();

  const uint32_t batch = onnxruntime::narrow<uint32_t>(src_shape[0]);
  const uint32_t src_height = onnxruntime::narrow<uint32_t>(src_shape[is_channels_last ? 1 : 2]);
  const uint32_t src_width = onnxruntime::narrow<uint32_t>(src_shape[is_channels_last ? 2 : 3]);
  const uint32_t output_height = onnxruntime::narrow<uint32_t>(output_shape[is_channels_last ? 1 : 2]);
  const uint32_t output_width = onnxruntime::narrow<uint32_t>(output_shape[is_channels_last ? 2 : 3]);

  const uint32_t im2col_m = output_height * output_width;
  const uint32_t im2col_k = kernel_height * kernel_width * channel_input;
  const uint32_t im2col_n = channel_output;

  const auto [tile_m, tile_n] = ChooseTileSize(im2col_m, im2col_n);
  const uint32_t workgroup_size = tile_n;

  // Check the device's subgroup size before shader compilation to avoid potential performance penalties
  // associated with conditional checks in the shader runtime.
  //
  // Ensure the subgroup size must be greater than or equal to `tile_m` to safely enable `use_subgroup`.
  // If the status of this condition is uncertain, the feature must be disabled.
  const bool use_subgroup = false;
  Im2ColMatMulProgram im2col_mm_program{has_bias, tile_m, tile_n, use_subgroup};
  im2col_mm_program.SetWorkgroupSize(workgroup_size);

  const uint32_t M_tiles = CeilDiv(im2col_m, tile_m);
  const uint32_t N_tiles = CeilDiv(im2col_n, tile_n);
  im2col_mm_program.SetDispatchGroupSize(M_tiles, N_tiles, batch);

  im2col_mm_program.AddInput({src,
                              ProgramTensorMetadataDependency::TypeAndRank,
                              4});
  im2col_mm_program.AddInput({&ohwi_weight,
                              ProgramTensorMetadataDependency::TypeAndRank,
                              4});
  if (has_bias) {
    im2col_mm_program.AddInput({bias,
                                ProgramTensorMetadataDependency::TypeAndRank});
  }
  im2col_mm_program.AddOutput({output,
                               ProgramTensorMetadataDependency::TypeAndRank});
  im2col_mm_program.AddUniformVariables({{batch},
                                         {src_height},
                                         {src_width},
                                         {channel_input},
                                         {kernel_height},
                                         {kernel_width},
                                         {output_height},
                                         {output_width},
                                         {im2col_m},
                                         {im2col_k},
                                         {im2col_n},
                                         {M_tiles},
                                         {N_tiles},
                                         {CeilDiv(CeilDiv(im2col_k, 4u), 4u)},
                                         {dilations},
                                         {pads},
                                         {strides}});
  im2col_mm_program.CacheHint(has_bias, tile_m, tile_n, use_subgroup);

  return context.RunProgram(im2col_mm_program);
}

bool CanApplyIm2ColMatMulProgram(ComputeContextBase& context,
                                 const bool is_channels_last,
                                 const bool is_fused,
                                 const TensorShape weight_shape,
                                 const uint32_t group) {
  if (!IsDeviceSupported(context)) {
    return false;
  }

  // TODO: Support !is_channels_last
  // TODO: Support fuse
  // TODO: Support group conv
  if (!is_channels_last || is_fused || group != 1) {
    return false;
  }

  // TODO: Support conv1d
  // TODO: Support conv2d_1x1
  const uint32_t kernel_height = onnxruntime::narrow<uint32_t>(weight_shape[2]);
  const uint32_t kernel_width = onnxruntime::narrow<uint32_t>(weight_shape[3]);
  if (kernel_height == 1 || kernel_width == 1) {
    return false;
  }

  // TODO: Support channel input vec1
  const uint32_t channel_input = onnxruntime::narrow<uint32_t>(weight_shape[1]);
  if (channel_input % 4 != 0) {
    return false;
  }

  return true;
}

}  // namespace webgpu
}  // namespace onnxruntime
