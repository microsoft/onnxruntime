// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/providers/webgpu/webgpu_supported_types.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"
#include "contrib_ops/webgpu/moe/block_fp8_expert_matmul.h"
#include "contrib_ops/webgpu/moe/qmoe.h"
#include "contrib_ops/webgpu/moe/gate_1token.h"
#include "contrib_ops/cpu/moe/moe_helper.h"
#include "contrib_ops/webgpu/quantization/matmul_nbits.h"
#include "core/providers/webgpu/math/gemm_packed.h"
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/platform/env_var_utils.h"
#if !defined(DISABLE_FLOAT8_TYPES)
#include "core/common/float8.h"
#endif

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include "gsl/gsl"
#include <atomic>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <string_view>

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;
using onnxruntime::webgpu::ComputeContext;

namespace {
std::atomic<uint64_t> block_fp8_matrix_dispatch_count{0};
}

uint64_t BlockFp8MatrixDispatchCount() {
  return block_fp8_matrix_dispatch_count.load(std::memory_order_relaxed);
}

namespace {

const std::string& Fp8E4M3DequantLutWgsl() {
  static const std::string lut = [] {
    std::ostringstream oss;
    oss << "const kFp8DequantLutBits = array<u32, 256>(";
    for (int i = 0; i < 256; ++i) {
      if (i > 0) {
        oss << ", ";
      }
#if !defined(DISABLE_FLOAT8_TYPES)
      const float value = Float8E4M3FN(static_cast<uint8_t>(i), Float8E4M3FN::FromBits()).ToFloat();
      uint32_t bits;
      std::memcpy(&bits, &value, sizeof(bits));
      oss << bits << "u";
#else
      oss << "0u";
#endif
    }
    oss << ");\n";
    return oss.str();
  }();
  return lut;
}

class QmoeGroupedCountProgram final : public Program<QmoeGroupedCountProgram> {
 public:
  QmoeGroupedCountProgram() : Program{"QmoeGroupedExpertCount"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"rows", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& experts = shader.AddInput("experts", ShaderUsage::UseUniform);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);
    shader.AdditionalImplementation() << "var<workgroup> counts: array<u32, 64>;\n";
    shader.MainFunctionBody()
        << "var count = 0u;\n"
        << "for (var row = local_idx; row < uniforms.rows; row += 64u) {\n"
        << "  if (" << experts.GetByOffset("row") << " == workgroup_idx) { count++; }\n"
        << "}\n"
        << "counts[local_idx] = count;\nworkgroupBarrier();\n"
        << "for (var width = 32u; width > 0u; width >>= 1u) {\n"
        << "  if (local_idx < width) { counts[local_idx] += counts[local_idx + width]; }\n"
        << "  workgroupBarrier();\n"
        << "}\n"
        << "if (local_idx == 0u) { " << output.SetByOffset("workgroup_idx", "counts[0]") << " }\n";
    return Status::OK();
  }
};

class QmoeGroupedOffsetProgram final : public Program<QmoeGroupedOffsetProgram> {
 public:
  QmoeGroupedOffsetProgram() : Program{"QmoeGroupedExpertOffsets"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"experts", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& counts = shader.AddInput("counts", ShaderUsage::UseUniform);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);
    shader.MainFunctionBody()
        << "if (local_idx != 0u) { return; }\n"
        << "var offset = 0u;\n"
        << output.SetByOffset("0u", "0u") << "\n"
        << "for (var expert = 0u; expert < uniforms.experts; expert++) {\n"
        << "  offset += " << counts.GetByOffset("expert") << ";\n"
        << "  " << output.SetByOffset("expert + 1u", "offset") << "\n"
        << "}\n";
    return Status::OK();
  }
};

class QmoeGroupedPermutationProgram final : public Program<QmoeGroupedPermutationProgram> {
 public:
  QmoeGroupedPermutationProgram() : Program{"QmoeGroupedExpertPermutation"} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"rows", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& experts = shader.AddInput("experts", ShaderUsage::UseUniform);
    const auto& offsets = shader.AddInput("offsets", ShaderUsage::UseUniform);
    const auto& rows = shader.AddOutput("rows", ShaderUsage::UseUniform);
    const auto& sorted = shader.AddOutput("sorted", ShaderUsage::UseUniform);
    shader.AdditionalImplementation() << "var<workgroup> cursor: atomic<u32>;\n";
    shader.MainFunctionBody()
        << "if (local_idx == 0u) { atomicStore(&cursor, 0u); }\n"
        << "workgroupBarrier();\n"
        << "for (var row = local_idx; row < uniforms.rows; row += 64u) {\n"
        << "  if (" << experts.GetByOffset("row") << " == workgroup_idx) {\n"
        << "    let position = " << offsets.GetByOffset("workgroup_idx") << " + atomicAdd(&cursor, 1u);\n"
        << "    " << rows.SetByOffset("position", "row") << "\n"
        << "    " << sorted.SetByOffset("position", "workgroup_idx") << "\n"
        << "  }\n"
        << "}\n";
    return Status::OK();
  }
};

class QmoeGroupedCopyProgram final : public Program<QmoeGroupedCopyProgram> {
 public:
  explicit QmoeGroupedCopyProgram(bool scatter)
      : Program{scatter ? "QmoeGroupedOutputScatter" : "QmoeGroupedHiddenGather"}, scatter_{scatter} {}
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"hidden", ProgramUniformVariableDataType::Uint32},
      {"k", ProgramUniformVariableDataType::Uint32});
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& data = shader.AddInput("data", ShaderUsage::UseUniform);
    const auto& rows = shader.AddInput("rows", ShaderUsage::UseUniform);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform);
    shader.MainFunctionBody()
        << "if (global_idx >= uniforms.rows * uniforms.hidden) { return; }\n"
        << "let sorted_row = global_idx / uniforms.hidden;\n"
        << "let column = global_idx % uniforms.hidden;\n"
        << "let original_row = " << rows.GetByOffset("sorted_row") << ";\n"
        << output.SetByOffset(scatter_ ? "original_row * uniforms.hidden + column" : "global_idx",
                              data.GetByOffset(scatter_ ? "global_idx" :
                                               "(original_row / uniforms.k) * uniforms.hidden + column")) << "\n";
    return Status::OK();
  }
 private:
  bool scatter_;
};

class BlockFp8ExpertMatMulProgram final : public Program<BlockFp8ExpertMatMulProgram> {
 public:
  BlockFp8ExpertMatMulProgram(bool has_bias, bool has_indirect_experts, bool broadcast_input, bool use_matrix,
                              bool use_cooperative)
      : Program<BlockFp8ExpertMatMulProgram>{"QMoEBlockFp8ExpertMatMul"},
        has_bias_{has_bias},
        has_indirect_experts_{has_indirect_experts},
        broadcast_input_{broadcast_input},
        use_matrix_{use_matrix},
        use_cooperative_{use_cooperative} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& input = shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
    const auto& weights = shader.AddInput("weights", ShaderUsage::UseElementTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    const ShaderVariableHelper* bias = &input;
    if (has_bias_) {
      bias = &shader.AddInput("bias", ShaderUsage::UseElementTypeAlias);
    }
    const ShaderVariableHelper* indirect_experts = &weights;
    if (has_indirect_experts_) {
      indirect_experts = &shader.AddInput("indirect_experts", ShaderUsage::UseElementTypeAlias);
    }
    const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    shader.AdditionalImplementation() << Fp8E4M3DequantLutWgsl();
    return WGSL_TEMPLATE_APPLY(shader, "moe/block_fp8_expert_matmul.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(broadcast_input, broadcast_input_),
                               WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                               WGSL_TEMPLATE_PARAMETER(has_indirect_experts, has_indirect_experts_),
                               WGSL_TEMPLATE_PARAMETER(use_cooperative, use_cooperative_),
                               WGSL_TEMPLATE_PARAMETER(use_matrix, use_matrix_),
                               WGSL_TEMPLATE_VARIABLE(bias, *bias),
                               WGSL_TEMPLATE_VARIABLE(indirect_experts, *indirect_experts),
                               WGSL_TEMPLATE_VARIABLE(input, input),
                               WGSL_TEMPLATE_VARIABLE(output, output),
                               WGSL_TEMPLATE_VARIABLE(scales, scales),
                               WGSL_TEMPLATE_VARIABLE(weights, weights));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32},
      {"inner", ProgramUniformVariableDataType::Uint32},
      {"weight_byte_offset", ProgramUniformVariableDataType::Uint32},
      {"scale_n_blocks", ProgramUniformVariableDataType::Uint32},
      {"scale_k_blocks", ProgramUniformVariableDataType::Uint32},
      {"experts_per_token", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_bias_;
  bool has_indirect_experts_;
  bool broadcast_input_;
  bool use_matrix_;
  bool use_cooperative_;
};

Status ApplyBlockFp8ExpertMatMul(ComputeContext& context,
                                 const Tensor* input,
                                 const Tensor* weights,
                                 const Tensor* scales,
                                 const Tensor* bias,
                                 Tensor* output,
                                 uint32_t rows,
                                 uint32_t cols,
                                 uint32_t inner,
                                 uint32_t expert_idx,
                                 const Tensor* indirect_experts = nullptr,
                                 bool broadcast_input = false,
                                 uint32_t experts_per_token = 1) {
  auto memory_info = OrtMemoryInfo{
      WEBGPU_BUFFER,
      OrtDeviceAllocator,
      OrtDevice{OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, 0}};
  Tensor raw_weights(DataTypeImpl::GetType<uint8_t>(), weights->Shape(),
                     const_cast<void*>(weights->DataRaw()), memory_info);

  const uint32_t scale_n_blocks = (cols - 1) / 128 + 1;
  const uint32_t scale_k_blocks = (inner - 1) / 128 + 1;
  uint32_t byte_offset = 0;
  const bool use_matrix = !indirect_experts && rows >= 128 && cols >= 256 && inner >= 256 &&
                          BlockFp8MatrixDispatchFits(rows, cols,
                                                     context.DeviceLimits().maxComputeWorkgroupsPerDimension) &&
                          input->DataType() == DataTypeImpl::GetType<MLFloat16>() &&
                          SelectSubgroupMatrixConfig(context, {{wgpu::SubgroupMatrixComponentType::F16,
                                                                wgpu::SubgroupMatrixComponentType::F32,
                                                                16, 16, 16, 32, false}})
                              .has_value();
  const bool use_cooperative = !use_matrix && rows <= 128 && inner >= 512 && cols >= 256 &&
                               cols <= context.DeviceLimits().maxComputeWorkgroupsPerDimension &&
                               ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_FP8_COOPERATIVE", 0) != 0;
  BlockFp8ExpertMatMulProgram program{bias != nullptr, indirect_experts != nullptr, broadcast_input, use_matrix,
                                      use_cooperative};
  program.AddInputs({{input, ProgramTensorMetadataDependency::Type}});
  if (indirect_experts) {
    program.AddInputs({{&raw_weights, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, 4}})
        .AddInputs({{scales, ProgramTensorMetadataDependency::Type}});
    if (bias) {
      program.AddInputs({{bias, ProgramTensorMetadataDependency::Type}});
    }
    program.AddInputs({{indirect_experts, ProgramTensorMetadataDependency::Type}});
  } else {
    const uint64_t weight_elements = uint64_t{cols} * inner;
    const uint64_t weight_offset = uint64_t{expert_idx} * weight_elements;
    const uint32_t first_word = static_cast<uint32_t>(weight_offset / 4);
    byte_offset = static_cast<uint32_t>(weight_offset % 4);
    // Include the shared boundary word when an expert starts inside it.
    const uint32_t view_words = static_cast<uint32_t>((byte_offset + weight_elements + 3) / 4);
    program.AddInputs({ProgramInput::BufferView(&raw_weights,
                                                ProgramTensorMetadataDependency::Type,
                                                TensorShape({view_words}),
                                                first_word,
                                                4)})
        .AddInputs({ProgramInput::BufferView(scales,
                                             ProgramTensorMetadataDependency::Type,
                                             TensorShape({scale_n_blocks * scale_k_blocks}),
                                             expert_idx * scale_n_blocks * scale_k_blocks)});
    if (bias) {
      program.AddInputs({ProgramInput::BufferView(bias,
                                                  ProgramTensorMetadataDependency::Type,
                                                  TensorShape({cols}),
                                                  expert_idx * cols)});
    }
  }
  program.AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(use_matrix ? 128 : 64)
      .SetDispatchGroupSize(use_matrix ? (cols - 1) / 16 + 1 : use_cooperative ? cols
                                                                               : (cols - 1) / 64 + 1,
                            use_matrix ? (rows - 1) / 64 + 1 : rows)
      .AddUniformVariables({rows, cols, inner, byte_offset,
                            scale_n_blocks, scale_k_blocks, experts_per_token})
      .CacheHint(bias != nullptr, indirect_experts != nullptr, broadcast_input, use_cooperative);
  if (use_matrix && context.HasFeature(wgpu::FeatureName::SubgroupSizeControl)) {
    program.SetSubgroupSize(32);
  }
  ORT_RETURN_IF_ERROR(context.RunProgram(program));
  if (use_matrix) {
    block_fp8_matrix_dispatch_count.fetch_add(1, std::memory_order_relaxed);
  }
  return Status::OK();
}

Status ValidateBlockFp8Projection(const Tensor* weights, const Tensor* scales, const Tensor* bias,
                                  const char* name, uint32_t rows, uint32_t cols, uint32_t inner,
                                  uint32_t max_workgroups_per_dimension) {
  constexpr uint64_t max_index = std::numeric_limits<uint32_t>::max();
  const uint64_t weight_elements = static_cast<uint64_t>(weights->Shape().Size());
  const uint64_t scale_elements = static_cast<uint64_t>(scales->Shape().Size());
  const uint64_t bias_elements = bias ? static_cast<uint64_t>(bias->Shape().Size()) : 0;
  ORT_RETURN_IF_NOT(cols != 0 && inner != 0 && rows != 0 &&
                        (uint64_t{cols} - 1) / 64 < max_workgroups_per_dimension &&
                        rows <= max_workgroups_per_dimension,
                    name, " block-FP8 dispatch exceeds WebGPU's per-dimension workgroup limit.");
  // Both the direct expert view and indirect shader use u32 byte/element offsets.
  // Reject matrices that could wrap those offsets even when their buffers are valid.
  ORT_RETURN_IF_NOT(weight_elements <= max_index && scale_elements <= max_index &&
                        bias_elements <= max_index &&
                        uint64_t{rows} * cols <= max_index,
                    name, " block-FP8 projection exceeds 32-bit shader address range.");
  return Status::OK();
}

Status ValidateBlockFp8Scales(const Tensor* scales, const char* name,
                              int64_t experts, int64_t output_size, int64_t input_size) {
  ORT_RETURN_IF_NOT(scales != nullptr, name, " is required for block-scaled FP8 QMoE.");
  ORT_RETURN_IF_NOT(scales->DataType() == DataTypeImpl::GetType<float>(),
                    name, " must have float32 elements for block-scaled FP8 QMoE.");
  const TensorShape expected({experts, (output_size + 127) / 128, (input_size + 127) / 128});
  ORT_RETURN_IF_NOT(scales->Shape() == expected, name, " must have shape ", expected,
                    " for block-scaled FP8 QMoE, got ", scales->Shape(), ".");
  return Status::OK();
}

class BlockQuantizedExpertMatMulProgram final : public Program<BlockQuantizedExpertMatMulProgram> {
 public:
  BlockQuantizedExpertMatMulProgram(bool has_bias, bool indirect, bool broadcast_input, bool k_packed, bool nvfp4)
      : Program<BlockQuantizedExpertMatMulProgram>{nvfp4 ? "QMoENvfp4ExpertMatMul" : "QMoEInt4ExpertMatMul"},
        has_bias_{has_bias},
        indirect_{indirect},
        broadcast_input_{broadcast_input},
        k_packed_{k_packed},
        nvfp4_{nvfp4} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& input = shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
    const auto& weights = shader.AddInput("weights", ShaderUsage::UseElementTypeAlias);
    const auto& scales = shader.AddInput("scales", ShaderUsage::UseElementTypeAlias);
    const ShaderVariableHelper* global_scales = &input;
    if (nvfp4_) {
      global_scales = &shader.AddInput("global_scales", ShaderUsage::UseElementTypeAlias);
    }
    const ShaderVariableHelper* bias = &input;
    if (has_bias_) {
      bias = &shader.AddInput("bias", ShaderUsage::UseElementTypeAlias);
    }
    const ShaderVariableHelper* indirect_experts = &weights;
    if (indirect_) {
      indirect_experts = &shader.AddInput("indirect_experts", ShaderUsage::UseElementTypeAlias);
    }
    const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    if (nvfp4_) {
      shader.AdditionalImplementation() << Fp8E4M3DequantLutWgsl();
    }
    return WGSL_TEMPLATE_APPLY(shader, "moe/block_quantized_expert_matmul.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(broadcast_input, broadcast_input_),
                               WGSL_TEMPLATE_PARAMETER(has_bias, has_bias_),
                               WGSL_TEMPLATE_PARAMETER(has_indirect_experts, indirect_),
                               WGSL_TEMPLATE_PARAMETER(k_packed, k_packed_),
                               WGSL_TEMPLATE_PARAMETER(nvfp4, nvfp4_),
                               WGSL_TEMPLATE_VARIABLE(bias, *bias),
                               WGSL_TEMPLATE_VARIABLE(global_scales, *global_scales),
                               WGSL_TEMPLATE_VARIABLE(indirect_experts, *indirect_experts),
                               WGSL_TEMPLATE_VARIABLE(input, input),
                               WGSL_TEMPLATE_VARIABLE(output, output),
                               WGSL_TEMPLATE_VARIABLE(scales, scales),
                               WGSL_TEMPLATE_VARIABLE(weights, weights));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"cols", ProgramUniformVariableDataType::Uint32},
      {"inner", ProgramUniformVariableDataType::Uint32},
      {"expert_idx", ProgramUniformVariableDataType::Uint32},
      {"experts_per_token", ProgramUniformVariableDataType::Uint32});

 private:
  bool has_bias_;
  bool indirect_;
  bool broadcast_input_;
  bool k_packed_;
  bool nvfp4_;
};

Status ApplyBlockQuantizedExpertMatMul(ComputeContext& context, const Tensor* input, const Tensor* weights,
                              const Tensor* scales, const Tensor* global_scales, const Tensor* bias,
                              Tensor* output, uint32_t rows, uint32_t cols, uint32_t inner,
                              uint32_t expert_idx, bool nvfp4, bool k_packed, const Tensor* indirect_experts = nullptr,
                              bool broadcast_input = false, uint32_t experts_per_token = 1) {
  auto memory_info = OrtMemoryInfo{
      WEBGPU_BUFFER, OrtDeviceAllocator,
      OrtDevice{OrtDevice::GPU, OrtDevice::MemType::DEFAULT, OrtDevice::VendorIds::NONE, 0}};
  std::optional<Tensor> raw_scales;
  BlockQuantizedExpertMatMulProgram program{bias != nullptr, indirect_experts != nullptr, broadcast_input, k_packed, nvfp4};
  program.AddInputs({{input, ProgramTensorMetadataDependency::Type},
                     {weights, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, 4}});
  if (nvfp4) {
    raw_scales.emplace(DataTypeImpl::GetType<uint8_t>(), scales->Shape(),
                       const_cast<void*>(scales->DataRaw()), memory_info);
    program.AddInputs({{&*raw_scales, ProgramTensorMetadataDependency::Type, ProgramInput::Flatten, 4},
                       {global_scales, ProgramTensorMetadataDependency::Type}});
  } else {
    program.AddInputs({{scales, ProgramTensorMetadataDependency::Type}});
  }
  if (bias) {
    program.AddInputs({{bias, ProgramTensorMetadataDependency::Type}});
  }
  if (indirect_experts) {
    program.AddInputs({{indirect_experts, ProgramTensorMetadataDependency::Type}});
  }
  program.AddOutput({output, ProgramTensorMetadataDependency::None})
      .SetWorkgroupSize(64)
      .SetDispatchGroupSize((cols + (k_packed ? 0u : 7u)) / (k_packed ? 1u : 8u), rows)
      .AddUniformVariables({cols, inner, expert_idx, experts_per_token})
      .CacheHint(bias != nullptr, indirect_experts != nullptr, broadcast_input, k_packed, nvfp4);
  return context.RunProgram(program);
}

}  // namespace

class GateProgram final : public Program<GateProgram> {
 public:
  GateProgram(int k, bool is_fp16, bool has_router_weights, bool normalize_routing_weights)
      : Program<GateProgram>{"QmoeGate"},
        k_{k},
        is_fp16_{is_fp16},
        has_router_weights_{has_router_weights},
        normalize_routing_weights_{normalize_routing_weights} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& router_logits = shader.AddInput("router_logits", ShaderUsage::UseElementTypeAlias);
    const ShaderVariableHelper* router_weights = &router_logits;
    if (has_router_weights_) {
      router_weights = &shader.AddInput("router_weights", ShaderUsage::UseElementTypeAlias);
    }
    const auto& topk_values = shader.AddOutput("topk_values");
    const auto& hiddenstate_for_expert = shader.AddOutput("hiddenstate_for_expert");
    shader.AddOutput("tokencount_for_expert");

    return WGSL_TEMPLATE_APPLY(shader, "moe/gate.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(has_router_weights, has_router_weights_),
                               WGSL_TEMPLATE_PARAMETER(is_fp16, is_fp16_),
                               WGSL_TEMPLATE_PARAMETER(k, k_),
                               WGSL_TEMPLATE_PARAMETER(normalize_routing_weights, normalize_routing_weights_),
                               WGSL_TEMPLATE_VARIABLE(hiddenstate_for_expert, hiddenstate_for_expert),
                               WGSL_TEMPLATE_VARIABLE(router_logits, router_logits),
                               WGSL_TEMPLATE_VARIABLE(router_weights, *router_weights),
                               WGSL_TEMPLATE_VARIABLE(topk_values, topk_values));
  };

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32},
      {"token_offset", ProgramUniformVariableDataType::Uint32});

 private:
  int k_;
  bool is_fp16_;
  bool has_router_weights_;
  bool normalize_routing_weights_;
};

Gate1TokenProgram::Gate1TokenProgram(int k, bool is_fp16, bool has_router_weights, bool normalize_routing_weights,
                                    bool sort_experts, bool use_subgroups)
    : Program<Gate1TokenProgram>{"QmoeGate1Token"},
      k_{k},
      is_fp16_{is_fp16},
      has_router_weights_{has_router_weights},
      normalize_routing_weights_{normalize_routing_weights},
      sort_experts_{sort_experts},
      use_subgroups_{use_subgroups} {}

Status Gate1TokenProgram::GenerateShaderCode(ShaderHelper& shader) const {
  const auto& router_logits = shader.AddInput("router_logits", ShaderUsage::UseElementTypeAlias);
  const ShaderVariableHelper* router_weights = &router_logits;
  if (has_router_weights_) {
    router_weights = &shader.AddInput("router_weights", ShaderUsage::UseElementTypeAlias);
  }
  const auto& topk_values = shader.AddOutput("topk_values");
  const auto& indirect_experts = shader.AddOutput("indirect_experts");

  return WGSL_TEMPLATE_APPLY(shader, "moe/gate_1token.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_router_weights, has_router_weights_),
                             WGSL_TEMPLATE_PARAMETER(is_fp16, is_fp16_),
                             WGSL_TEMPLATE_PARAMETER(k, k_),
                             WGSL_TEMPLATE_PARAMETER(normalize_routing_weights, normalize_routing_weights_),
                             WGSL_TEMPLATE_PARAMETER(sort_experts, sort_experts_),
                             WGSL_TEMPLATE_PARAMETER(use_subgroups, use_subgroups_),
                             WGSL_TEMPLATE_VARIABLE(indirect_experts, indirect_experts),
                             WGSL_TEMPLATE_VARIABLE(router_logits, router_logits),
                             WGSL_TEMPLATE_VARIABLE(router_weights, *router_weights),
                             WGSL_TEMPLATE_VARIABLE(topk_values, topk_values));
}

class ExpandMoEInputProgram final : public Program<ExpandMoEInputProgram> {
 public:
  ExpandMoEInputProgram() : Program<ExpandMoEInputProgram>{"QmoeExpandInput"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& input = shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
    const auto& expanded = shader.AddOutput("expanded", ShaderUsage::UseElementTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "moe/expand_input.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(expanded, expanded),
                               WGSL_TEMPLATE_VARIABLE(input, input));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"k", ProgramUniformVariableDataType::Uint32},
      {"rows", ProgramUniformVariableDataType::Uint32});
};

class HiddenStateGatherProgram final : public Program<HiddenStateGatherProgram> {
 public:
  HiddenStateGatherProgram() : Program<HiddenStateGatherProgram>{"QmoeHiddenStateGather"} {};

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& hiddenstate_for_expert = shader.AddInput("hiddenstate_for_expert", ShaderUsage::UseElementTypeAlias);
    const auto& hidden_state = shader.AddInput("hidden_state", ShaderUsage::UseElementTypeAlias);
    const auto& new_hidden_state = shader.AddOutput("new_hidden_state");
    const auto& tokens = shader.AddOutput("tokens");

    return WGSL_TEMPLATE_APPLY(shader, "moe/hidden_state_gather.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(hidden_state, hidden_state),
                               WGSL_TEMPLATE_VARIABLE(hiddenstate_for_expert, hiddenstate_for_expert),
                               WGSL_TEMPLATE_VARIABLE(new_hidden_state, new_hidden_state),
                               WGSL_TEMPLATE_VARIABLE(tokens, tokens));
  };

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"expert_idx", ProgramUniformVariableDataType::Uint32},
      {"num_experts", ProgramUniformVariableDataType::Uint32},
      {"num_tokens", ProgramUniformVariableDataType::Uint32},
      {"hidden_size", ProgramUniformVariableDataType::Uint32});

 private:
};

class ZeroTensorProgram final : public Program<ZeroTensorProgram> {
 public:
  ZeroTensorProgram() : Program<ZeroTensorProgram>{"QmoeZeroTensor"} {};

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& tensor = shader.AddOutput("tensor", ShaderUsage::UseElementTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "moe/zero_tensor.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(tensor, tensor));
  };

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"size", ProgramUniformVariableDataType::Uint32});

 private:
};

class ZeroU32Program final : public Program<ZeroU32Program> {
 public:
  ZeroU32Program() : Program<ZeroU32Program>{"QmoeZeroU32"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& output = shader.AddOutput("output");
    return WGSL_TEMPLATE_APPLY(shader, "moe/zero_u32.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(output, output));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"size", ProgramUniformVariableDataType::Uint32});
};

class MoEActivationProgram final : public Program<MoEActivationProgram> {
 public:
  MoEActivationProgram(MoEActivationType activation_type, int swiglu_fusion, bool has_fc3)
      : Program<MoEActivationProgram>{"MoEActivation"},
        activation_type_{activation_type},
        swiglu_fusion_{swiglu_fusion},
        has_fc3_{has_fc3} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& input = shader.AddInput("input", ShaderUsage::UseElementTypeAlias);
    const ShaderVariableHelper* fc3_input = &input;
    if (has_fc3_) {
      fc3_input = &shader.AddInput("fc3_input", ShaderUsage::UseElementTypeAlias);
    }
    const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);

    return WGSL_TEMPLATE_APPLY(shader, "moe/activation.wgsl.template",
                               WGSL_TEMPLATE_PARAMETER(activation, static_cast<int>(activation_type_)),
                               WGSL_TEMPLATE_PARAMETER(has_fc3, has_fc3_),
                               WGSL_TEMPLATE_PARAMETER(swiglu_fusion, swiglu_fusion_),
                               WGSL_TEMPLATE_VARIABLE(fc3_input, *fc3_input),
                               WGSL_TEMPLATE_VARIABLE(input, input),
                               WGSL_TEMPLATE_VARIABLE(output, output));
  };

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"rows", ProgramUniformVariableDataType::Uint32},
      {"cols", ProgramUniformVariableDataType::Uint32},
      {"alpha", ProgramUniformVariableDataType::Float32},
      {"beta", ProgramUniformVariableDataType::Float32},
      {"swiglu_limit", ProgramUniformVariableDataType::Float32});

 private:
  MoEActivationType activation_type_;
  int swiglu_fusion_;
  bool has_fc3_;
};

class FusedFinalMix1TokenProgram final : public Program<FusedFinalMix1TokenProgram> {
 public:
  FusedFinalMix1TokenProgram() : Program<FusedFinalMix1TokenProgram>{"QmoeFusedFinalMix1Token"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& fc2_outputs = shader.AddInput("fc2_outputs", ShaderUsage::UseElementTypeAlias);
    const auto& router_values = shader.AddInput("router_values", ShaderUsage::UseElementTypeAlias);
    const auto& indirect_experts = shader.AddInput("indirect_experts", ShaderUsage::UseElementTypeAlias);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "moe/fused_final_mix_1token.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(fc2_outputs, fc2_outputs),
                               WGSL_TEMPLATE_VARIABLE(indirect_experts, indirect_experts),
                               WGSL_TEMPLATE_VARIABLE(output, output),
                               WGSL_TEMPLATE_VARIABLE(router_values, router_values));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"k", ProgramUniformVariableDataType::Uint32},
      {"num_experts", ProgramUniformVariableDataType::Uint32});
};

class QMoEFinalMixProgram final : public Program<QMoEFinalMixProgram> {
 public:
  QMoEFinalMixProgram() : Program<QMoEFinalMixProgram>{"QMoEFinalMix"} {}

  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& fc2_outputs = shader.AddInput("fc2_outputs", ShaderUsage::UseElementTypeAlias);
    const auto& router_values = shader.AddInput("router_values", ShaderUsage::UseElementTypeAlias);
    const auto& expert_tokens = shader.AddInput("expert_tokens", ShaderUsage::UseElementTypeAlias);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseElementTypeAlias);

    return WGSL_TEMPLATE_APPLY(shader, "moe/final_mix.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(expert_tokens, expert_tokens),
                               WGSL_TEMPLATE_VARIABLE(fc2_outputs, fc2_outputs),
                               WGSL_TEMPLATE_VARIABLE(output, output),
                               WGSL_TEMPLATE_VARIABLE(router_values, router_values));
  }

  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(
      {"hidden_size", ProgramUniformVariableDataType::Uint32},
      {"num_experts", ProgramUniformVariableDataType::Uint32},
      {"expert_idx", ProgramUniformVariableDataType::Uint32},
      {"token_offset", ProgramUniformVariableDataType::Uint32});
};

QMoE::~QMoE() {
  const char* path = std::getenv("ORT_WEBGPU_QMOE_COMMAND_REPLAY_STATS_FILE");
  if (path == nullptr || replay_requests_ == 0) {
    return;
  }
  static std::mutex statistics_mutex;
  std::lock_guard<std::mutex> lock(statistics_mutex);
  std::ofstream output(path, std::ios::app);
  output << "{\"kernel_identity\":" << reinterpret_cast<uintptr_t>(this)
         << ",\"requests\":" << replay_requests_
         << ",\"hits\":" << replay_hits_
         << ",\"captures\":" << replay_captures_
         << ",\"entries\":" << replay_entries_.size()
         << ",\"evictions\":" << replay_evictions_
         << ",\"shape_invalidations\":" << replay_shape_invalidations_
         << ",\"capacity_fallbacks\":" << replay_capacity_fallbacks_
         << ",\"capacity\":8}\n";
  if (!output) {
    LOGS_DEFAULT(ERROR) << "Cannot write QMoE command replay statistics to " << path;
  }
}

#include "qmoe_fc1_activation.inc"
#include "qmoe_fc2_final_mix.inc"
#include "qmoe_hierarchical_gate.inc"

Status QMoE::ComputeInternal(ComputeContext& context) const {
  const int expert_bitcast = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_EXPERT_BITCAST_UNPACK", 0);
  ORT_RETURN_IF_NOT(expert_bitcast == 0 || expert_bitcast == 1, "ORT_WEBGPU_EXPERT_BITCAST_UNPACK must be 0 or 1");
  const int extended_prefill = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_EXTENDED_PREFILL", 0);
  ORT_RETURN_IF_NOT(extended_prefill == 0 || extended_prefill == 1, "ORT_WEBGPU_QMOE_EXTENDED_PREFILL must be 0 or 1");
  const int prefill256 = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_PREFILL256", 0);
  ORT_RETURN_IF_NOT(prefill256 == 0 || prefill256 == 1, "ORT_WEBGPU_QMOE_PREFILL256 must be 0 or 1");
  const int prefill512 = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_PREFILL512", 0);
  ORT_RETURN_IF_NOT(prefill512 == 0 || prefill512 == 1, "ORT_WEBGPU_QMOE_PREFILL512 must be 0 or 1");
  const int prefill1024 = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_PREFILL1024", 0);
  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, "ORT_WEBGPU_QMOE_PREFILL1024 must be 0 or 1");
  const int grouped_prefill = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_GROUPED_PREFILL", 0);
  ORT_RETURN_IF_NOT(grouped_prefill == 0 || grouped_prefill == 1, "ORT_WEBGPU_QMOE_GROUPED_PREFILL must be 0 or 1");

  const int hierarchical_gate = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_HIERARCHICAL_GATE", 0);
  ORT_RETURN_IF_NOT(hierarchical_gate == 0 || hierarchical_gate == 1, "ORT_WEBGPU_QMOE_HIERARCHICAL_GATE must be 0 or 1");
  const int fc2_mix = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_FC2_FINAL_MIX", 0);
  ORT_RETURN_IF_NOT(fc2_mix == 0 || fc2_mix == 1, "ORT_WEBGPU_QMOE_FC2_FINAL_MIX must be 0 or 1");

  const int fc1_activation = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_FC1_ACTIVATION", 0);
  ORT_RETURN_IF_NOT(fc1_activation == 0 || fc1_activation == 1, "ORT_WEBGPU_QMOE_FC1_ACTIVATION must be 0 or 1");

  const int enabled = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_COMMAND_REPLAY", 0);
  ORT_RETURN_IF_NOT(enabled == 0 || enabled == 1, "ORT_WEBGPU_QMOE_COMMAND_REPLAY must be 0 or 1");
  if (enabled == 0) {
    return ComputeWithWorkspace(context, nullptr);
  }
  std::lock_guard<std::mutex> lock(replay_mutex_);
  const Tensor* input = context.Input<Tensor>(0);
  const Tensor* routing = context.Input<Tensor>(1);
  if (!context.CanCaptureNodeCommands() || context.NodeReplayProfilingEnabled() ||
      input == nullptr || routing == nullptr ||
      input->Shape().NumDimensions() == 0 || input->Shape().GetDims().back() <= 0 ||
      input->Shape().Size() != input->Shape().GetDims().back() ||
      !input->IsDataType<MLFloat16>() || routing->Shape().Size() != 512 ||
      is_block_fp8_ || is_nvfp4_ || fc1_expert_weight_bits_ != 4 ||
      fc2_expert_weight_bits_ != 4 || block_size_ != 32 || k_ != 10 ||
      !normalize_routing_weights_ || context.Input<Tensor>(8) != nullptr ||
      context.Input<Tensor>(11) != nullptr || context.Input<Tensor>(12) != nullptr ||
      context.Input<Tensor>(13) != nullptr || context.Input<Tensor>(14) != nullptr) {
    return ComputeWithWorkspace(context, nullptr);
  }
  ++replay_requests_;
  std::vector<uint64_t> key;
  std::vector<uint64_t> workspace_key;
  key.reserve(128);
  workspace_key.reserve(96);
  key.push_back(context.NodeReplayProfilingEnabled());
  key.push_back(context.EnableMatmulFp32Accumulation());
  key.push_back(fc1_activation);
  key.push_back(fc2_mix);
  key.push_back(hierarchical_gate);
  key.push_back(expert_bitcast);
  key.push_back(extended_prefill);
  key.push_back(prefill256);
  key.push_back(prefill512);
  key.push_back(prefill1024);
  key.push_back(grouped_prefill);
  for (const char* flag : {"ORT_WEBGPU_QMOE_INT4_COOPERATIVE", "ORT_WEBGPU_QMOE_SUBGROUP_GATE",
                           "ORT_WEBGPU_QMOE_INT4_PREFILL", "ORT_WEBGPU_MATMUL_NBITS_SGM_N_TAIL",
                           "ORT_WEBGPU_MATMUL_NBITS_SKINNY", "ORT_WEBGPU_MATMUL_GEMV_N96"}) {
    key.push_back(ParseEnvironmentVariableWithDefault<int>(flag, 0) != 0);
  }
  auto append_tensor = [&key, &workspace_key](const Tensor* tensor) {
    key.push_back(tensor == nullptr ? 0 : reinterpret_cast<uintptr_t>(tensor->DataRaw()));
    workspace_key.push_back(tensor != nullptr);
    if (tensor != nullptr) {
      key.push_back(tensor->GetElementType());
      key.push_back(tensor->Shape().NumDimensions());
      workspace_key.push_back(tensor->GetElementType());
      workspace_key.push_back(tensor->Shape().NumDimensions());
      for (int64_t dimension : tensor->Shape().GetDims()) {
        key.push_back(static_cast<uint64_t>(dimension));
        workspace_key.push_back(static_cast<uint64_t>(dimension));
      }
    }
  };
  key.push_back(context.InputCount());
  for (int index = 0; index < context.InputCount(); ++index) {
    append_tensor(context.Input<Tensor>(index));
  }
  append_tensor(context.Output(0, input->Shape()));
  if (!replay_workspace_key_.empty() && replay_workspace_key_ != workspace_key) {
    ORT_RETURN_IF_ERROR(context.FlushNodeCommands());
    replay_entries_.clear();
    for (auto& tensor : replay_scratch_) {
      tensor.reset();
    }
    ++replay_shape_invalidations_;
  }
  replay_workspace_key_ = std::move(workspace_key);
  for (auto& entry : replay_entries_) {
    if (entry.key == key) {
      ORT_RETURN_IF_ERROR(context.AppendNodeCommands(entry.commands));
      entry.last_used = ++replay_clock_;
      ++replay_hits_;
      return Status::OK();
    }
  }
  if (replay_entries_.size() == 8) {
    auto oldest = std::min_element(replay_entries_.begin(), replay_entries_.end(),
                                   [](const auto& a, const auto& b) { return a.last_used < b.last_used; });
    replay_entries_.erase(oldest);
    ++replay_evictions_;
  }
  ReplayEntry entry;
  entry.key = std::move(key);
  context.BeginNodeCapture(entry.commands);
  auto end_capture = gsl::finally([&context]() { context.EndNodeCapture(); });
  ORT_RETURN_IF_ERROR(ComputeWithWorkspace(context, &replay_scratch_));
  ORT_RETURN_IF_ERROR(context.FlushNodeCommands());
  const size_t expected_commands = (CanFuseFc1Activation(context, fc1_activation == 1) ? 4 : 5) -
      (CanFuseFc2FinalMix(context, fc2_mix == 1) ? 1 : 0);
  ORT_RETURN_IF_NOT(entry.commands.size() == expected_commands,
                    "QMoE node capture command count mismatch, got ", entry.commands.size());
  entry.last_used = ++replay_clock_;
  ++replay_captures_;
  replay_entries_.push_back(std::move(entry));
  return Status::OK();
}

Status QMoE::ComputeWithWorkspace(ComputeContext& context,
                                 std::array<std::optional<Tensor>, 5>* scratch) const {

  const int expert_bitcast = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_EXPERT_BITCAST_UNPACK", 0);
  ORT_RETURN_IF_NOT(expert_bitcast == 0 || expert_bitcast == 1, "ORT_WEBGPU_EXPERT_BITCAST_UNPACK must be 0 or 1");
  const int extended_prefill = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_EXTENDED_PREFILL", 0);
  ORT_RETURN_IF_NOT(extended_prefill == 0 || extended_prefill == 1, "ORT_WEBGPU_QMOE_EXTENDED_PREFILL must be 0 or 1");
  const int prefill256 = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_PREFILL256", 0);
  ORT_RETURN_IF_NOT(prefill256 == 0 || prefill256 == 1, "ORT_WEBGPU_QMOE_PREFILL256 must be 0 or 1");
  const int prefill512 = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_PREFILL512", 0);
  ORT_RETURN_IF_NOT(prefill512 == 0 || prefill512 == 1, "ORT_WEBGPU_QMOE_PREFILL512 must be 0 or 1");
  const int prefill1024 = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_PREFILL1024", 0);
  ORT_RETURN_IF_NOT(prefill1024 == 0 || prefill1024 == 1, "ORT_WEBGPU_QMOE_PREFILL1024 must be 0 or 1");
  const int grouped_prefill = ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_GROUPED_PREFILL", 0);
  ORT_RETURN_IF_NOT(grouped_prefill == 0 || grouped_prefill == 1, "ORT_WEBGPU_QMOE_GROUPED_PREFILL must be 0 or 1");
  const Tensor* hidden_state = context.Input<Tensor>(0);
  const Tensor* router_logits = context.Input<Tensor>(1);
  // fc1 is gate_up_proj
  const Tensor* fc1_experts_weights = context.Input<Tensor>(2);
  const Tensor* fc1_scales = context.Input<Tensor>(3);
  const Tensor* fc1_experts_bias_optional = context.Input<Tensor>(4);
  // fc2 is gate_down_proj
  const Tensor* fc2_experts_weights = context.Input<Tensor>(5);
  const Tensor* fc2_scales = context.Input<Tensor>(6);
  const Tensor* fc2_experts_bias_optional = context.Input<Tensor>(7);
  const Tensor* fc3_experts_weights_optional = context.Input<Tensor>(8);
  const Tensor* fc3_scales_optional = context.Input<Tensor>(9);
  const Tensor* fc3_experts_bias_optional = context.Input<Tensor>(10);
  const Tensor* fc1_zero_points = context.Input<Tensor>(11);
  const Tensor* fc2_zero_points = context.Input<Tensor>(12);
  const Tensor* fc3_zero_points = context.Input<Tensor>(13);
  const Tensor* router_weights = context.Input<Tensor>(14);
  const Tensor* fc1_global_scale = context.Input<Tensor>(15);
  const Tensor* fc2_global_scale = context.Input<Tensor>(16);
  const Tensor* fc1_act_scale = context.Input<Tensor>(17);
  const Tensor* fc2_act_scale = context.Input<Tensor>(18);
  const Tensor* fc1_act_block_scale = context.Input<Tensor>(19);
  const Tensor* fc2_act_block_scale = context.Input<Tensor>(20);

  MoEParameters moe_params;

  int swiglu_fusion = swiglu_fusion_;
  if (activation_type_ == MoEActivationType::SwiGLU && swiglu_fusion == 0 &&
      fc3_experts_weights_optional == nullptr) {
    swiglu_fusion = 1;
  }
  const bool is_swiglu = activation_type_ == MoEActivationType::SwiGLU;
  const bool is_fused_swiglu = is_swiglu && swiglu_fusion != 0 && fc3_experts_weights_optional == nullptr;
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, hidden_state, router_logits,
      fc1_experts_weights, fc1_experts_bias_optional, is_block_fp8_ ? nullptr : fc1_scales, fc1_zero_points,
      fc2_experts_weights, fc2_experts_bias_optional, is_block_fp8_ ? nullptr : fc2_scales, fc2_zero_points,
      fc3_experts_weights_optional, fc3_experts_bias_optional, is_block_fp8_ ? nullptr : fc3_scales_optional, fc3_zero_points,
      moe_helper::MoEWeightBits{fc1_expert_weight_bits_,
                                fc2_expert_weight_bits_,
                                fc3_expert_weight_bits_},
      is_fused_swiglu, block_size_));
  ORT_RETURN_IF(router_weights && router_weights->Shape() != router_logits->Shape(),
                "router_weights must have the same shape as router_probs; got ",
                router_weights->Shape(), " and ", router_logits->Shape());
  ORT_RETURN_IF_NOT(k_ > 0 && k_ <= moe_params.num_experts,
                    "QMoE requires 0 < k <= num_experts, got k=", k_,
                    " and num_experts=", moe_params.num_experts);
  ORT_RETURN_IF_NOT(block_size_ > 0 ||
                        (fc1_zero_points == nullptr && fc2_zero_points == nullptr && fc3_zero_points == nullptr),
                    "WebGPU QMoE row-wise quantization does not support explicit zero points. "
                    "Use block-wise quantization or omit the zero-point inputs.");
  ORT_RETURN_IF_NOT(moe_params.num_rows != 1 ||
                        (fc1_zero_points == nullptr && fc2_zero_points == nullptr && fc3_zero_points == nullptr),
                    "WebGPU QMoE does not support explicit zero points on the optimized single-token path.");

  if (is_nvfp4_) {
#if defined(DISABLE_FLOAT8_TYPES)
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "NVFP4 QMoE requires float8 types enabled.");
#endif
    ORT_RETURN_IF_NOT(fc3_experts_weights_optional == nullptr && fc3_scales_optional == nullptr &&
                          fc3_experts_bias_optional == nullptr,
                      "WebGPU NVFP4 QMoE does not support a separate FC3 projection.");
    ORT_RETURN_IF_NOT(fc1_zero_points == nullptr && fc2_zero_points == nullptr && fc3_zero_points == nullptr &&
                          fc1_act_scale == nullptr && fc2_act_scale == nullptr &&
                          fc1_act_block_scale == nullptr && fc2_act_block_scale == nullptr,
                      "WebGPU NVFP4 QMoE does not support zero points or activation scales.");
    const auto validate_projection = [&](const Tensor* weights, const Tensor* scales, const Tensor* global_scales,
                                         int64_t cols, int64_t inner, const char* name) -> Status {
      ORT_RETURN_IF_NOT(cols > 0 && cols % 2 == 0 && inner > 0 && inner % 16 == 0 &&
                            weights->Shape() == TensorShape({moe_params.num_experts, inner, cols / 2}),
                        name, " NVFP4 weights must have logical shape [E,K,N/2], with K divisible by 16.");
      ORT_RETURN_IF_NOT(weights->IsDataType<uint8_t>() && scales != nullptr &&
                            scales->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN &&
                            scales->Shape() == TensorShape({moe_params.num_experts, cols, inner / 16}) &&
                            global_scales != nullptr && global_scales->IsDataType<float>() &&
                            global_scales->Shape() == TensorShape({moe_params.num_experts}),
                        name, " NVFP4 requires uint8 weights, E4M3FN [E,N,K/16] scales, and float32 [E] global scales.");
      ORT_RETURN_IF_NOT(weights->Shape().Size() <= std::numeric_limits<uint32_t>::max() &&
                            scales->Shape().Size() <= std::numeric_limits<uint32_t>::max() &&
                            static_cast<uint64_t>(moe_params.num_rows) * k_ * cols <=
                                std::numeric_limits<uint32_t>::max(),
                        name, " NVFP4 projection exceeds 32-bit shader addressing.");
      const uint64_t column_groups = nvfp4_k_packed_ ? static_cast<uint64_t>(cols)
                                                     : (static_cast<uint64_t>(cols) + 7) / 8;
      ORT_RETURN_IF_NOT(column_groups <= context.DeviceLimits().maxComputeWorkgroupsPerDimension,
                        name, " NVFP4 projection exceeds WebGPU's per-dimension workgroup limit.");
      return Status::OK();
    };
    ORT_RETURN_IF_ERROR(validate_projection(fc1_experts_weights, fc1_scales, fc1_global_scale,
                                            is_fused_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size,
                                            moe_params.hidden_size, "FC1"));
    ORT_RETURN_IF_ERROR(validate_projection(fc2_experts_weights, fc2_scales, fc2_global_scale,
                                            moe_params.hidden_size, moe_params.inter_size, "FC2"));
  } else if (is_block_fp8_) {
#if defined(DISABLE_FLOAT8_TYPES)
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "Block-scaled FP8 QMoE requires a build with float8 types enabled.");
#endif
    const int64_t fc1_size = is_fused_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size;
    ORT_RETURN_IF_NOT(fc1_experts_weights->Shape() ==
                              TensorShape({moe_params.num_experts, fc1_size, moe_params.hidden_size}) &&
                          fc2_experts_weights->Shape() ==
                              TensorShape({moe_params.num_experts, moe_params.hidden_size, moe_params.inter_size}) &&
                          (fc3_experts_weights_optional == nullptr ||
                           fc3_experts_weights_optional->Shape() ==
                               TensorShape({moe_params.num_experts, moe_params.inter_size, moe_params.hidden_size})),
                      "Block-scaled FP8 QMoE requires canonical [E,N,K] expert weight shapes.");
    ORT_RETURN_IF_NOT(fc1_zero_points == nullptr && fc2_zero_points == nullptr && fc3_zero_points == nullptr,
                      "Block-scaled FP8 QMoE does not support zero points.");
    ORT_RETURN_IF_NOT(fc1_global_scale == nullptr && fc2_global_scale == nullptr,
                      "Block-scaled FP8 QMoE uses fc1_scales/fc2_scales directly; global scales must be omitted.");
    ORT_RETURN_IF_NOT(fc1_act_scale == nullptr && fc2_act_scale == nullptr &&
                          fc1_act_block_scale == nullptr && fc2_act_block_scale == nullptr,
                      "Block-scaled FP8 QMoE does not support activation scales.");
    ORT_RETURN_IF_NOT(fc1_experts_weights->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN &&
                          fc2_experts_weights->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN &&
                          (fc3_experts_weights_optional == nullptr ||
                           fc3_experts_weights_optional->GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN),
                      "Block-scaled FP8 QMoE weights must use float8e4m3fn elements.");
    ORT_RETURN_IF_ERROR(ValidateBlockFp8Scales(fc1_scales, "fc1_scales", moe_params.num_experts,
                                               fc1_size, moe_params.hidden_size));
    ORT_RETURN_IF_ERROR(ValidateBlockFp8Scales(fc2_scales, "fc2_scales", moe_params.num_experts,
                                               moe_params.hidden_size, moe_params.inter_size));
    ORT_RETURN_IF(fc3_scales_optional != nullptr && fc3_experts_weights_optional == nullptr,
                  "Block-scaled FP8 QMoE fc3_scales requires fc3_experts_weights.");
    if (fc3_experts_weights_optional) {
      ORT_RETURN_IF_ERROR(ValidateBlockFp8Scales(fc3_scales_optional, "fc3_scales", moe_params.num_experts,
                                                 moe_params.inter_size, moe_params.hidden_size));
    }
  } else {
    ORT_RETURN_IF_NOT(fc1_experts_weights->DataType() == DataTypeImpl::GetType<uint8_t>() &&
                          fc2_experts_weights->DataType() == DataTypeImpl::GetType<uint8_t>() &&
                          (fc3_experts_weights_optional == nullptr ||
                           fc3_experts_weights_optional->DataType() == DataTypeImpl::GetType<uint8_t>()),
                      "Integer QMoE weights must use uint8 elements.");
    // MatMulNBits addresses packed rows and blocks as u32 words; partial words would drop weights.
    const auto has_word_aligned_rows = [&](int64_t row_elems, int64_t col_elems, int64_t bits) {
      // A single scale for the matrix makes each row the effective packed block.
      const int64_t packed_block = block_size_ != 0 && block_size_ != row_elems * col_elems
                                       ? block_size_
                                       : row_elems;
      return (row_elems * bits) % 32 == 0 && (packed_block * bits) % 32 == 0;
    };
    ORT_RETURN_IF_NOT(has_word_aligned_rows(moe_params.hidden_size,
                                            is_fused_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size,
                                            fc1_expert_weight_bits_),
                      "WebGPU QMoE FC1 packed rows and blocks must be 32-bit aligned.");
    ORT_RETURN_IF_NOT(has_word_aligned_rows(moe_params.inter_size, moe_params.hidden_size, fc2_expert_weight_bits_),
                      "WebGPU QMoE FC2 packed rows and blocks must be 32-bit aligned.");
    ORT_RETURN_IF_NOT(fc3_experts_weights_optional == nullptr ||
                          has_word_aligned_rows(moe_params.hidden_size, moe_params.inter_size, fc3_expert_weight_bits_),
                      "WebGPU QMoE FC3 packed rows and blocks must be 32-bit aligned.");
  }

  const auto& input_shape = hidden_state->Shape();

  // process tokens in chunks of max_tokens to put some cap on memory usage
  const int max_tokens = 2 * 1024;

  const uint32_t num_experts = static_cast<uint32_t>(moe_params.num_experts);
  Tensor* output_tensor = context.Output(0, input_shape);
  if (moe_params.num_rows == 0) {
    return Status::OK();
  }
  const auto& device_limits = context.DeviceLimits();
  ORT_RETURN_IF_NOT(num_experts <= device_limits.maxComputeWorkgroupSizeX &&
                        num_experts <= device_limits.maxComputeInvocationsPerWorkgroup,
                    "WebGPU QMoE requires num_experts to fit in one workgroup; got ", num_experts,
                    ", maxComputeWorkgroupSizeX=", device_limits.maxComputeWorkgroupSizeX,
                    ", maxComputeInvocationsPerWorkgroup=", device_limits.maxComputeInvocationsPerWorkgroup, ".");
  const uint32_t hidden_size = static_cast<uint32_t>(moe_params.hidden_size);
  const bool use_nvfp4_prefill = is_nvfp4_ &&
                                 ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_NVFP4_PREFILL", 0) != 0;
  const bool is_symmetric_int4 = !is_block_fp8_ && !is_nvfp4_ &&
                                fc1_expert_weight_bits_ == 4 && fc2_expert_weight_bits_ == 4 &&
                                (fc3_experts_weights_optional == nullptr || fc3_expert_weight_bits_ == 4) &&
                                fc1_zero_points == nullptr && fc2_zero_points == nullptr && fc3_zero_points == nullptr;
  const bool use_int4_prefill = is_symmetric_int4 &&
                                ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_INT4_PREFILL", 0) != 0;
  const bool use_int3_prefill = fc1_expert_weight_bits_ == 3 && fc2_expert_weight_bits_ == 3 &&
                                (fc3_experts_weights_optional == nullptr || fc3_expert_weight_bits_ == 3) &&
                                block_size_ == 32 && fc1_zero_points == nullptr &&
                                fc2_zero_points == nullptr && fc3_zero_points == nullptr &&
                                ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_INT3_PREFILL", 0) != 0;
  const bool use_integer_prefill = use_int4_prefill || use_int3_prefill;
  const bool use_extended_int4_prefill =
      extended_prefill == 1 && use_int4_prefill && block_size_ == 32 &&
      hidden_state->IsDataType<MLFloat16>() && is_fused_swiglu &&
      moe_params.hidden_size == 2560 && moe_params.inter_size == 640 &&
      num_experts == 512 && k_ == 10 && fc3_experts_weights_optional == nullptr &&
      context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
      !context.EnableMatmulFp32Accumulation() &&
      device_limits.maxComputeWorkgroupsPerDimension >= 2560;
  const bool use_prefill256 =
      prefill256 == 1 && use_extended_int4_prefill && moe_params.num_rows == 256 &&
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_INT4_COOPERATIVE", 0) == 0 &&
      device_limits.maxComputeWorkgroupsPerDimension >= 12800;
  const bool use_prefill512 =
      prefill512 == 1 && use_extended_int4_prefill && moe_params.num_rows == 512 &&
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_INT4_COOPERATIVE", 0) == 0 &&
      device_limits.maxComputeWorkgroupsPerDimension >= 25600;
  const bool use_prefill1024 =
      prefill1024 == 1 && use_extended_int4_prefill && moe_params.num_rows == 1024 &&
      ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_INT4_COOPERATIVE", 0) == 0 &&
      device_limits.maxComputeWorkgroupsPerDimension >= 51200;
  const int64_t max_indirect_tokens =
      use_prefill1024 ? 1024 : use_prefill512 ? 512 : use_prefill256 ? 256 : use_nvfp4_prefill || use_extended_int4_prefill ? 128 : use_integer_prefill ? 16 : 8;
  const int64_t max_indirect_rows =
      use_prefill1024 ? 10240 : use_prefill512 ? 5120 : use_prefill256 ? 2560 : use_nvfp4_prefill ? 4096 : use_extended_int4_prefill ? 1280 : use_integer_prefill ? 256 : 128;
  const bool use_small_batch = (is_block_fp8_ || is_nvfp4_ || use_integer_prefill) &&
                               moe_params.num_rows <= max_indirect_tokens &&
                               moe_params.num_rows * k_ <= max_indirect_rows && router_weights == nullptr &&
                               normalize_routing_weights_ &&
                               (is_nvfp4_ || use_integer_prefill ||
                                ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_SMALL_BATCH", 0) != 0);
  const bool use_indirect_experts = (moe_params.num_rows == 1 && router_weights == nullptr) || use_small_batch;
  const int64_t fc1_output_size = is_fused_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size;
  const bool use_int4_cooperative = is_symmetric_int4 && block_size_ == 32 &&
                                    moe_params.hidden_size % 32 == 0 && moe_params.inter_size % 32 == 0 &&
                                    moe_params.hidden_size <= device_limits.maxComputeWorkgroupsPerDimension &&
                                    fc1_output_size <= device_limits.maxComputeWorkgroupsPerDimension &&
                                    fc1_experts_weights->Shape().Size() <= std::numeric_limits<uint32_t>::max() &&
                                    fc2_experts_weights->Shape().Size() <= std::numeric_limits<uint32_t>::max() &&
                                    (fc3_experts_weights_optional == nullptr ||
                                     fc3_experts_weights_optional->Shape().Size() <= std::numeric_limits<uint32_t>::max()) &&
                                    ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_INT4_COOPERATIVE", 0) != 0;
  if (is_block_fp8_) {
    const auto max_groups = device_limits.maxComputeWorkgroupsPerDimension;
    const uint32_t max_rows = static_cast<uint32_t>(
        use_indirect_experts ? moe_params.num_rows * k_ : std::min<int64_t>(moe_params.num_rows, max_tokens));
    ORT_RETURN_IF_ERROR(ValidateBlockFp8Projection(
        fc1_experts_weights, fc1_scales, fc1_experts_bias_optional, "FC1", max_rows,
        static_cast<uint32_t>(fc1_output_size), hidden_size, max_groups));
    ORT_RETURN_IF_ERROR(ValidateBlockFp8Projection(
        fc2_experts_weights, fc2_scales, fc2_experts_bias_optional, "FC2", max_rows,
        hidden_size, static_cast<uint32_t>(moe_params.inter_size), max_groups));
    if (fc3_experts_weights_optional) {
      ORT_RETURN_IF_ERROR(ValidateBlockFp8Projection(
          fc3_experts_weights_optional, fc3_scales_optional, fc3_experts_bias_optional, "FC3", max_rows,
          static_cast<uint32_t>(moe_params.inter_size), hidden_size, max_groups));
    }
  }
  const bool is_fp16 = hidden_state->DataType() == DataTypeImpl::GetType<MLFloat16>();
  const auto dtype = is_fp16 ? DataTypeImpl::GetType<MLFloat16>() : DataTypeImpl::GetType<float>();
  const auto dtype_uint32 = DataTypeImpl::GetType<uint32_t>();

  const int64_t K_fc1 = moe_params.hidden_size;
  const int64_t N_fc1 = fc1_output_size;
  const int64_t K_fc2 = moe_params.inter_size;
  const int64_t N_fc2 = moe_params.hidden_size;
  const int64_t accuracy_level = 4;
  const int64_t block_size_fc1 = (block_size_ != 0) ? block_size_ : K_fc1;
  const int64_t block_size_fc2 = (block_size_ != 0) ? block_size_ : K_fc2;
  const int64_t block_size_fc3 = block_size_fc1;
  const bool use_grouped_prefill = grouped_prefill == 1 && use_extended_int4_prefill &&
      (use_prefill512 || use_prefill1024) && use_small_batch && !use_int4_cooperative;
  Status status;

  if (use_indirect_experts) {
    std::array<std::optional<Tensor>, 5> local_scratch;
    auto temporary = [&](size_t index, MLDataType type, const TensorShape& shape) -> Tensor& {
      auto& tensor = scratch == nullptr ? local_scratch[index] : (*scratch)[index];
      if (!tensor) {
        tensor.emplace(context.CreateGPUTensor(type, shape));
      } else {
        ORT_ENFORCE(tensor->DataType() == type && tensor->Shape() == shape,
                    "QMoE replay workspace shape/type changed.");
      }
      return *tensor;
    };
    const uint32_t k = static_cast<uint32_t>(k_);
    const uint32_t num_tokens = static_cast<uint32_t>(moe_params.num_rows);
    const uint32_t routed_rows = num_tokens * k;
    TensorShape gate_value_shape({num_tokens, num_experts});
    TensorShape indirect_experts_shape({routed_rows});

    Tensor& router_values = temporary(0, dtype, gate_value_shape);
    Tensor& indirect_experts = temporary(1, dtype_uint32, indirect_experts_shape);

    // Step 1: Gate — select top-k experts
    const bool sort_experts = !is_block_fp8_ && !is_nvfp4_ && num_tokens > 1;
    const bool use_subgroups = num_tokens == 1 && num_experts == 512 && k_ == 10 &&
                               is_symmetric_int4 && is_fp16 && router_weights == nullptr &&
                               normalize_routing_weights_ &&
                               context.AdapterInfo().vendor == std::string_view{"nvidia"} &&
                               context.HasFeature(wgpu::FeatureName::Subgroups) &&
                               ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_SUBGROUP_GATE", 0) != 0;
    const bool use_hierarchical = use_subgroups &&
        context.AdapterInfo().subgroupMinSize >= 32 &&
        ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_HIERARCHICAL_GATE", 0) == 1;
    auto run_gate = [&](auto& gate) -> Status {
    gate
        .AddInputs({{router_logits, ProgramTensorMetadataDependency::Type}});
    if (router_weights) {
      gate.AddInputs({{router_weights, ProgramTensorMetadataDependency::Type}});
    }
    gate.AddOutput({&router_values, ProgramTensorMetadataDependency::None})
        .AddOutput({&indirect_experts, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(num_experts)
        .SetDispatchGroupSize(num_tokens)
        .AddUniformVariables({num_tokens, num_experts})
        .CacheHint(k_, is_fp16 ? "fp16" : "fp32", router_weights != nullptr, normalize_routing_weights_, sort_experts, use_subgroups);
    return context.RunProgram(gate);
    };
    if (use_hierarchical) {
      QmoeHierarchicalGateProgram gate;
      ORT_RETURN_IF_ERROR(run_gate(gate));
    } else {
      Gate1TokenProgram gate{k_, is_fp16, router_weights != nullptr, normalize_routing_weights_, sort_experts, use_subgroups};
      ORT_RETURN_IF_ERROR(run_gate(gate));
    }


    std::optional<Tensor> group_counts, group_offsets, group_rows, sorted_experts;
    if (use_grouped_prefill) {
      group_counts.emplace(context.CreateGPUTensor(dtype_uint32, TensorShape({num_experts})));
      group_offsets.emplace(context.CreateGPUTensor(dtype_uint32, TensorShape({num_experts + 1u})));
      group_rows.emplace(context.CreateGPUTensor(dtype_uint32, TensorShape({routed_rows})));
      sorted_experts.emplace(context.CreateGPUTensor(dtype_uint32, TensorShape({routed_rows})));
      QmoeGroupedCountProgram count;
      count.AddInput({&indirect_experts, ProgramTensorMetadataDependency::Type})
          .AddOutput({&*group_counts, ProgramTensorMetadataDependency::Type})
          .AddUniformVariables({{routed_rows}})
          .SetWorkgroupSize(64).SetDispatchGroupSize(num_experts);
      ORT_RETURN_IF_ERROR(context.RunProgram(count));
      QmoeGroupedOffsetProgram offsets;
      offsets.AddInput({&*group_counts, ProgramTensorMetadataDependency::Type})
          .AddOutput({&*group_offsets, ProgramTensorMetadataDependency::Type})
          .AddUniformVariables({{num_experts}})
          .SetWorkgroupSize(64).SetDispatchGroupSize(1);
      ORT_RETURN_IF_ERROR(context.RunProgram(offsets));
      QmoeGroupedPermutationProgram permutation;
      permutation.AddInputs({{&indirect_experts, ProgramTensorMetadataDependency::Type},
                             {&*group_offsets, ProgramTensorMetadataDependency::Type}})
          .AddOutputs({{&*group_rows, ProgramTensorMetadataDependency::Type},
                       {&*sorted_experts, ProgramTensorMetadataDependency::Type}})
          .AddUniformVariables({{routed_rows}})
          .SetWorkgroupSize(64).SetDispatchGroupSize(num_experts);
      ORT_RETURN_IF_ERROR(context.RunProgram(permutation));
    }
    const Tensor* projection_experts = use_grouped_prefill ? &*sorted_experts : &indirect_experts;
    std::optional<Tensor> expanded_input;
    const Tensor* integer_input = hidden_state;
    if (!is_block_fp8_ && !is_nvfp4_ && !use_int4_cooperative && num_tokens > 1) {
      expanded_input.emplace(context.CreateGPUTensor(dtype, TensorShape({routed_rows, hidden_size})));
      if (use_grouped_prefill) {
        QmoeGroupedCopyProgram gather{false};
        gather.AddInputs({{hidden_state, ProgramTensorMetadataDependency::Type},
                          {&*group_rows, ProgramTensorMetadataDependency::Type}})
            .AddOutput({&*expanded_input, ProgramTensorMetadataDependency::Type})
            .AddUniformVariables({{routed_rows}, {hidden_size}, {k}})
            .SetWorkgroupSize(64).SetDispatchGroupSize((hidden_size + 63) / 64, routed_rows);
        ORT_RETURN_IF_ERROR(context.RunProgram(gather));
      } else {
      ExpandMoEInputProgram expand;
      expand.AddInput({hidden_state, ProgramTensorMetadataDependency::Type})
          .AddOutput({&*expanded_input, ProgramTensorMetadataDependency::None})
          .SetWorkgroupSize(64)
          .SetDispatchGroupSize((hidden_size + 63) / 64, routed_rows)
          .AddUniformVariables({hidden_size, k, routed_rows});
      ORT_RETURN_IF_ERROR(context.RunProgram(expand));
      }
      integer_input = &*expanded_input;
    }

    TensorShape fc1_output_shape({static_cast<int64_t>(routed_rows), fc1_output_size});
    Tensor& fc1_outputs = temporary(2, dtype, fc1_output_shape);
    TensorShape fc1_activated_shape({static_cast<int64_t>(routed_rows), moe_params.inter_size});
    Tensor& fc1_activated = temporary(3, dtype, fc1_activated_shape);

    if (CanFuseFc1Activation(context,
          ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_FC1_ACTIVATION", 0) == 1)) {
      QmoeFC1SwiGLUProgram program{fc1_experts_bias_optional != nullptr, expert_bitcast == 1};
      program.AddInput({hidden_state, ProgramTensorMetadataDependency::TypeAndRank, 4})
          .AddInput({fc1_experts_weights, ProgramTensorMetadataDependency::TypeAndRank, 16})
          .AddInput({fc1_scales, ProgramTensorMetadataDependency::TypeAndRank});
      if (fc1_experts_bias_optional) {
        program.AddInput({fc1_experts_bias_optional, ProgramTensorMetadataDependency::None});
      }
      const auto cols = onnxruntime::narrow<uint32_t>(moe_params.inter_size);
      const auto tiles = CeilDiv(cols * 2u, 8u);
      program.AddInput({&indirect_experts, ProgramTensorMetadataDependency::None})
          .AddOutput({&fc1_activated, ProgramTensorMetadataDependency::None})
          .SetWorkgroupSize(128).SetDispatchGroupSize(tiles, routed_rows)
          .AddUniformVariables({hidden_size, cols, tiles, activation_alpha_, activation_beta_, swiglu_limit_})
          .CacheHint(fc1_experts_bias_optional != nullptr);
      ORT_RETURN_IF_ERROR(context.RunProgram(program));
    } else {
    status = is_nvfp4_ || use_int4_cooperative
                 ? ApplyBlockQuantizedExpertMatMul(context, hidden_state, fc1_experts_weights, fc1_scales,
                                          fc1_global_scale, fc1_experts_bias_optional, &fc1_outputs, routed_rows,
                                          static_cast<uint32_t>(N_fc1), static_cast<uint32_t>(K_fc1), 0,
                                          is_nvfp4_, is_nvfp4_ ? nvfp4_k_packed_ : true, &indirect_experts, true, k)
             : is_block_fp8_
                 ? ApplyBlockFp8ExpertMatMul(context, hidden_state, fc1_experts_weights, fc1_scales,
                                             fc1_experts_bias_optional, &fc1_outputs, routed_rows,
                                             static_cast<uint32_t>(N_fc1), static_cast<uint32_t>(K_fc1), 0,
                                             &indirect_experts, true, k)
                 : ApplyMatMulNBits(integer_input, fc1_experts_weights, fc1_scales, fc1_zero_points,
                                    fc1_experts_bias_optional, K_fc1, N_fc1, block_size_fc1, accuracy_level,
                                    fc1_expert_weight_bits_, context, &fc1_outputs, 0, projection_experts,
                                    /*override_M=*/num_tokens == 1 ? k : 0);
    ORT_RETURN_IF_ERROR(status);

    // Step 3: Apply the activation, optionally combining FC1 with a separate FC3 projection.
    std::optional<Tensor> fc3_outputs;
    if (fc3_experts_weights_optional) {
      fc3_outputs.emplace(context.CreateGPUTensor(dtype, fc1_activated_shape));
      ORT_RETURN_IF_ERROR(use_int4_cooperative
                              ? ApplyBlockQuantizedExpertMatMul(context, hidden_state, fc3_experts_weights_optional,
                                                                fc3_scales_optional, nullptr, fc3_experts_bias_optional,
                                                                &*fc3_outputs, routed_rows,
                                                                static_cast<uint32_t>(moe_params.inter_size),
                                                                static_cast<uint32_t>(K_fc1), 0, false, true,
                                                                &indirect_experts, true, k)
                          : is_block_fp8_
                              ? ApplyBlockFp8ExpertMatMul(context, hidden_state, fc3_experts_weights_optional,
                                                          fc3_scales_optional, fc3_experts_bias_optional,
                                                          &*fc3_outputs, routed_rows, static_cast<uint32_t>(moe_params.inter_size),
                                                          static_cast<uint32_t>(K_fc1), 0, &indirect_experts, true, k)
                              : ApplyMatMulNBits(integer_input, fc3_experts_weights_optional, fc3_scales_optional,
                                                 fc3_zero_points, fc3_experts_bias_optional,
                                                 K_fc1, moe_params.inter_size, block_size_fc3, accuracy_level,
                                                 fc3_expert_weight_bits_, context, &*fc3_outputs, 0, &indirect_experts,
                                                 /*override_M=*/num_tokens == 1 ? k : 0));
    }
    MoEActivationProgram activation{activation_type_, swiglu_fusion, fc3_outputs.has_value()};
    activation.AddInputs({{&fc1_outputs, ProgramTensorMetadataDependency::Type}});
    if (fc3_outputs) {
      activation.AddInputs({{&*fc3_outputs, ProgramTensorMetadataDependency::Type}});
    }
    activation
        .AddOutput({&fc1_activated, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(128)
        .SetDispatchGroupSize(((routed_rows * static_cast<uint32_t>(moe_params.inter_size)) + 127) / 128)
        .AddUniformVariables({routed_rows, static_cast<uint32_t>(moe_params.inter_size), activation_alpha_,
                              activation_beta_, swiglu_limit_})
        .CacheHint(static_cast<int>(activation_type_), swiglu_fusion, fc3_outputs.has_value());
    ORT_RETURN_IF_ERROR(context.RunProgram(activation));

    }

    TensorShape fc2_output_shape({static_cast<int64_t>(routed_rows), N_fc2});
    Tensor& fc2_outputs = temporary(4, dtype, fc2_output_shape);

    if (CanFuseFc2FinalMix(context,
          ParseEnvironmentVariableWithDefault<int>("ORT_WEBGPU_QMOE_FC2_FINAL_MIX", 0) == 1)) {
      QmoeFC2FinalMixProgram program{fc2_experts_bias_optional != nullptr, expert_bitcast == 1};
      program.AddInput({&fc1_activated, ProgramTensorMetadataDependency::TypeAndRank, 4})
          .AddInput({fc2_experts_weights, ProgramTensorMetadataDependency::TypeAndRank, 16})
          .AddInput({fc2_scales, ProgramTensorMetadataDependency::TypeAndRank});
      if (fc2_experts_bias_optional) {
        program.AddInput({fc2_experts_bias_optional, ProgramTensorMetadataDependency::None});
      }
      program.AddInput({&router_values, ProgramTensorMetadataDependency::Type})
          .AddInput({&indirect_experts, ProgramTensorMetadataDependency::Type})
          .AddOutput({output_tensor, ProgramTensorMetadataDependency::None})
          .SetWorkgroupSize(320).SetDispatchGroupSize(1280)
          .AddUniformVariables({hidden_size, k, num_experts})
          .CacheHint(fc2_experts_bias_optional != nullptr);
      ORT_RETURN_IF_ERROR(context.RunProgram(program));
    } else {
    status = is_nvfp4_ || use_int4_cooperative
                 ? ApplyBlockQuantizedExpertMatMul(context, &fc1_activated, fc2_experts_weights, fc2_scales,
                                          fc2_global_scale, fc2_experts_bias_optional, &fc2_outputs, routed_rows,
                                          static_cast<uint32_t>(N_fc2), static_cast<uint32_t>(K_fc2), 0,
                                          is_nvfp4_, is_nvfp4_ ? nvfp4_k_packed_ : true, &indirect_experts)
             : is_block_fp8_
                 ? ApplyBlockFp8ExpertMatMul(context, &fc1_activated, fc2_experts_weights, fc2_scales,
                                             fc2_experts_bias_optional, &fc2_outputs, routed_rows,
                                             static_cast<uint32_t>(N_fc2), static_cast<uint32_t>(K_fc2), 0,
                                             &indirect_experts)
                 : ApplyMatMulNBits(&fc1_activated, fc2_experts_weights, fc2_scales, fc2_zero_points,
                                    fc2_experts_bias_optional, K_fc2, N_fc2, block_size_fc2, accuracy_level,
                                    fc2_expert_weight_bits_, context, &fc2_outputs, 0, projection_experts,
                                    /*override_M=*/0);
    ORT_RETURN_IF_ERROR(status);


    std::optional<Tensor> restored_fc2;
    const Tensor* final_mix_input = &fc2_outputs;
    if (use_grouped_prefill) {
      restored_fc2.emplace(context.CreateGPUTensor(dtype, fc2_output_shape));
      QmoeGroupedCopyProgram scatter{true};
      scatter.AddInputs({{&fc2_outputs, ProgramTensorMetadataDependency::Type},
                         {&*group_rows, ProgramTensorMetadataDependency::Type}})
          .AddOutput({&*restored_fc2, ProgramTensorMetadataDependency::Type})
          .AddUniformVariables({{routed_rows}, {hidden_size}, {k}})
          .SetWorkgroupSize(64).SetDispatchGroupSize((hidden_size + 63) / 64, routed_rows);
      ORT_RETURN_IF_ERROR(context.RunProgram(scatter));
      final_mix_input = &*restored_fc2;
    }
    const uint32_t mix_wg_size = 256;
    FusedFinalMix1TokenProgram final_mix;
    final_mix
        .AddInputs({{final_mix_input, ProgramTensorMetadataDependency::Type}})
        .AddInputs({{&router_values, ProgramTensorMetadataDependency::Type}})
        .AddInputs({{&indirect_experts, ProgramTensorMetadataDependency::Type}})
        .AddOutput({output_tensor, ProgramTensorMetadataDependency::None})
        .SetWorkgroupSize(mix_wg_size)
        .SetDispatchGroupSize((hidden_size + mix_wg_size - 1) / mix_wg_size, num_tokens)
        .AddUniformVariables({hidden_size, k, num_experts});
    ORT_RETURN_IF_ERROR(context.RunProgram(final_mix));

    }

    return Status::OK();
  }

  // Multi-token path: accumulates into output_tensor, need to initialize to zero.
  const int total_output_size = (static_cast<int>(input_shape.Size()) + 3) / 4;
  ZeroTensorProgram zero;
  zero
      .AddOutput({output_tensor, ProgramTensorMetadataDependency::Type, ProgramOutput::Flatten, 4})
      .SetDispatchGroupSize((total_output_size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
      .AddUniformVariables({static_cast<uint32_t>(total_output_size)});
  ORT_RETURN_IF_ERROR(context.RunProgram(zero));

  // path for num_tokens > 1
  // process tokens in chunks of max_tokens to put some cap on memory usage
  for (int token_offset = 0; token_offset < moe_params.num_rows; token_offset += max_tokens) {
    //
    // Step 1: run the gate to get router indices and values
    //
    int num_tokens = static_cast<int>(moe_params.num_rows) - token_offset;
    if (num_tokens > max_tokens) {
      num_tokens = max_tokens;
    }
    TensorShape gate_value_shape({num_tokens, num_experts});   // use max_tokens ?
    TensorShape gate_hidden_shape({num_experts, num_tokens});  // use max_tokens ?
    TensorShape gate_count_shape({num_experts});

    // router_values: per expert float we multiply final results with
    Tensor router_values = context.CreateGPUTensor(dtype, gate_value_shape);
    // gate_counts: number of tokens assigned to each expert
    Tensor gate_counts = context.CreateGPUTensor(dtype_uint32, gate_count_shape);
    // gate_hidden: token_idx assigned to each expert
    //  token_idx is the global index into hidden_state
    Tensor gate_hidden = context.CreateGPUTensor(dtype_uint32, gate_hidden_shape);

    ZeroU32Program zero_counts;
    zero_counts.AddOutput({&gate_counts, ProgramTensorMetadataDependency::None})
        .SetDispatchGroupSize((num_experts + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE)
        .AddUniformVariables({num_experts});
    ORT_RETURN_IF_ERROR(context.RunProgram(zero_counts));

    GateProgram gate{k_, is_fp16, router_weights != nullptr, normalize_routing_weights_};
    gate
        .AddInputs({{router_logits, ProgramTensorMetadataDependency::Type}});
    if (router_weights) {
      gate.AddInputs({{router_weights, ProgramTensorMetadataDependency::Type}});
    }
    gate.AddOutput({&router_values, ProgramTensorMetadataDependency::None})
        .AddOutput({&gate_hidden, ProgramTensorMetadataDependency::None})
        .AddOutput({&gate_counts, ProgramTensorMetadataDependency::None, ProgramOutput::Atomic})
        .SetWorkgroupSize(num_experts)
        .SetDispatchGroupSize(static_cast<uint32_t>(num_tokens))
        .AddUniformVariables({static_cast<uint32_t>(num_tokens), num_experts, static_cast<uint32_t>(token_offset)})
        .CacheHint(k_, is_fp16 ? "fp16" : "fp32", router_weights != nullptr, normalize_routing_weights_);

    ORT_RETURN_IF_ERROR(context.RunProgram(gate));

    Tensor gate_counts_cpu = context.CreateCPUTensor(dtype_uint32, gate_count_shape);
    ORT_RETURN_IF_ERROR(Info().GetDataTransferManager().CopyTensor(gate_counts, gate_counts_cpu));

    for (uint32_t expert_idx = 0; expert_idx < num_experts; expert_idx++) {
      uint32_t used_by = *(gate_counts_cpu.Data<uint32_t>() + expert_idx);
      if (used_by <= 0) {
        continue;
      }

      //
      // Step 2: for each expert, gather the hidden_state rows assigned to it
      // FIXME: use vec4
      //
      TensorShape expert_hidden_shape({used_by, moe_params.hidden_size});
      // expert_hidden: hidden states assigned to this expert
      Tensor expert_hidden = context.CreateGPUTensor(dtype, expert_hidden_shape);
      TensorShape expert_tokens_shape({used_by});
      // expert_tokens: token_idx that match expert_hidden rows
      Tensor expert_tokens = context.CreateGPUTensor(dtype_uint32, expert_tokens_shape);
      HiddenStateGatherProgram gather;
      gather
          .AddInputs({{&gate_hidden, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{hidden_state, ProgramTensorMetadataDependency::Type, 1}})
          .AddOutput({&expert_hidden, ProgramTensorMetadataDependency::None, 1})
          .AddOutput({&expert_tokens, ProgramTensorMetadataDependency::None})
          .SetDispatchGroupSize(used_by)
          .AddUniformVariables({expert_idx,
                                num_experts,
                                static_cast<uint32_t>(num_tokens),
                                hidden_size});
      ORT_RETURN_IF_ERROR(context.RunProgram(gather));

      TensorShape fc1_output_shape({used_by, fc1_output_size});
      Tensor fc1_outputs = context.CreateGPUTensor(dtype, fc1_output_shape);
      TensorShape fc1_activated_shape({used_by, moe_params.inter_size});
      Tensor fc1_activated = context.CreateGPUTensor(dtype, fc1_activated_shape);
      TensorShape fc2_output_shape({used_by, N_fc2});
      Tensor fc2_outputs = context.CreateGPUTensor(dtype, fc2_output_shape);

      //
      // Step 3: matmul the hidden_state with fc1 (gate_up) of the selected experts
      //
      status = is_nvfp4_
                   ? ApplyBlockQuantizedExpertMatMul(context, &expert_hidden, fc1_experts_weights, fc1_scales,
                                            fc1_global_scale, fc1_experts_bias_optional, &fc1_outputs, used_by,
                                            static_cast<uint32_t>(N_fc1), static_cast<uint32_t>(K_fc1), expert_idx,
                                            true, nvfp4_k_packed_)
               : is_block_fp8_
                   ? ApplyBlockFp8ExpertMatMul(context, &expert_hidden, fc1_experts_weights, fc1_scales,
                                               fc1_experts_bias_optional, &fc1_outputs, used_by,
                                               static_cast<uint32_t>(N_fc1), static_cast<uint32_t>(K_fc1), expert_idx)
                   : ApplyMatMulNBits(&expert_hidden, fc1_experts_weights, fc1_scales, fc1_zero_points,
                                      fc1_experts_bias_optional, K_fc1, N_fc1, block_size_fc1, accuracy_level,
                                      fc1_expert_weight_bits_, context, &fc1_outputs, expert_idx);
      ORT_RETURN_IF_ERROR(status);

      //
      // Step 4: apply the activation and optional FC3 gate.
      //
      std::optional<Tensor> fc3_outputs;
      if (fc3_experts_weights_optional) {
        fc3_outputs.emplace(context.CreateGPUTensor(dtype, fc1_activated_shape));
        ORT_RETURN_IF_ERROR(is_block_fp8_
                                ? ApplyBlockFp8ExpertMatMul(context, &expert_hidden, fc3_experts_weights_optional,
                                                            fc3_scales_optional, fc3_experts_bias_optional,
                                                            &*fc3_outputs, used_by,
                                                            static_cast<uint32_t>(moe_params.inter_size),
                                                            static_cast<uint32_t>(K_fc1), expert_idx)
                                : ApplyMatMulNBits(&expert_hidden, fc3_experts_weights_optional, fc3_scales_optional,
                                                   fc3_zero_points, fc3_experts_bias_optional,
                                                   K_fc1, moe_params.inter_size, block_size_fc3, accuracy_level,
                                                   fc3_expert_weight_bits_, context, &*fc3_outputs, expert_idx));
      }
      MoEActivationProgram activation{activation_type_, swiglu_fusion, fc3_outputs.has_value()};
      activation.AddInputs({{&fc1_outputs, ProgramTensorMetadataDependency::Type}});
      if (fc3_outputs) {
        activation.AddInputs({{&*fc3_outputs, ProgramTensorMetadataDependency::Type}});
      }
      activation
          .AddOutput({&fc1_activated, ProgramTensorMetadataDependency::None})
          .SetWorkgroupSize(128)
          .SetDispatchGroupSize(((used_by * static_cast<uint32_t>(moe_params.inter_size)) + 127) / 128)
          .AddUniformVariables({used_by, static_cast<uint32_t>(moe_params.inter_size), activation_alpha_,
                                activation_beta_, swiglu_limit_})
          .CacheHint(static_cast<int>(activation_type_), swiglu_fusion, fc3_outputs.has_value());
      ORT_RETURN_IF_ERROR(context.RunProgram(activation));

      //
      // Step 5: multiply fc1_activated with fc2 (gate_down) of the selected experts
      //
      status = is_nvfp4_
                   ? ApplyBlockQuantizedExpertMatMul(context, &fc1_activated, fc2_experts_weights, fc2_scales,
                                            fc2_global_scale, fc2_experts_bias_optional, &fc2_outputs, used_by,
                                            static_cast<uint32_t>(N_fc2), static_cast<uint32_t>(K_fc2), expert_idx,
                                            true, nvfp4_k_packed_)
               : is_block_fp8_
                   ? ApplyBlockFp8ExpertMatMul(context, &fc1_activated, fc2_experts_weights, fc2_scales,
                                               fc2_experts_bias_optional, &fc2_outputs, used_by,
                                               static_cast<uint32_t>(N_fc2), static_cast<uint32_t>(K_fc2), expert_idx)
                   : ApplyMatMulNBits(&fc1_activated, fc2_experts_weights, fc2_scales, fc2_zero_points,
                                      fc2_experts_bias_optional, K_fc2, N_fc2, block_size_fc2, accuracy_level,
                                      fc2_expert_weight_bits_, context, &fc2_outputs, expert_idx);
      ORT_RETURN_IF_ERROR(status);

      //
      // Step 6: multiply fc2_outputs with router_values and accumulate
      //
      QMoEFinalMixProgram final_mix;
      final_mix
          .AddInputs({{&fc2_outputs, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{&router_values, ProgramTensorMetadataDependency::Type}})
          .AddInputs({{&expert_tokens, ProgramTensorMetadataDependency::Type}})
          .AddOutput({output_tensor, ProgramTensorMetadataDependency::None})
          .SetDispatchGroupSize(used_by)
          .AddUniformVariables({hidden_size,
                                num_experts,
                                expert_idx,
                                static_cast<uint32_t>(token_offset)})
          .CacheHint(router_weights != nullptr, normalize_routing_weights_);

      ORT_RETURN_IF_ERROR(context.RunProgram(final_mix));
    }
  }

  return Status::OK();
}

namespace {
const std::vector<MLDataType>& QMoET1Constraint() {
  static std::vector<MLDataType> types{
      DataTypeImpl::GetTensorType<uint8_t>(),
#if !defined(DISABLE_FLOAT8_TYPES)
      DataTypeImpl::GetTensorType<Float8E4M3FN>(),
#endif
  };
  return types;
}
const std::vector<MLDataType>& QMoET2Constraint() {
  static const std::vector<MLDataType> types = [] {
    auto result = WebGpuSupportedFloatTypes();
#if !defined(DISABLE_FLOAT8_TYPES)
    result.push_back(DataTypeImpl::GetTensorType<Float8E4M3FN>());
#endif
    return result;
  }();
  return types;
}
}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    QMoE,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", WebGpuSupportedFloatTypes())
        .TypeConstraint("T1", QMoET1Constraint())
        .TypeConstraint("T2", QMoET2Constraint()),
    QMoE);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
