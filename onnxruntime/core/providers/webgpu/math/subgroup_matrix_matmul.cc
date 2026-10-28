// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/subgroup_matrix_tiling.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>

#include "core/common/narrow.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/math/matmul.h"
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/math/subgroup_matrix_pad_b.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/vendor/intel/math/subgroup_matrix_tiling_selector.h"
namespace onnxruntime {
namespace webgpu {

// Computes Y = activation(A @ B + optional bias) using subgroupMatrixMultiplyAccumulate.
class SubgroupMatrixMatMulProgram final : public Program<SubgroupMatrixMatMulProgram> {
 public:
  SubgroupMatrixMatMulProgram(const Activation& activation, bool has_bias, SubgroupMatrixConfig config,
                              uint32_t sg_mat_count_m, uint32_t sg_mat_count_n, uint32_t split_k)
      : Program{"SubgroupMatrixMatMul"},
        activation_(activation),
        has_bias_(has_bias),
        config_(config),
        sg_mat_count_m_(sg_mat_count_m),
        sg_mat_count_n_(sg_mat_count_n),
        split_k_(split_k) {}
  Status GenerateShaderCode(ShaderHelper& sh) const override;
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"M", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"K", ProgramUniformVariableDataType::Uint32},
                                          {"num_n_tile", ProgramUniformVariableDataType::Uint32},
                                          {"N_b", ProgramUniformVariableDataType::Uint32},
                                          WEBGPU_PROGRAM_ACTIVATION_UNIFORM_VARIABLES);

 private:
  const Activation activation_;
  const bool has_bias_;
  const SubgroupMatrixConfig config_;
  const uint32_t sg_mat_count_m_;
  const uint32_t sg_mat_count_n_;
  const uint32_t split_k_;
};

namespace {

// Subgroup-matrix MatMul implementation. Loads both A and B directly from global
// memory and runs the subgroup-matrix kernel during Compute. The class is
// intended to support all subgroup-matrix configs. For now 8x16x16 and 16x16x16
// are implemented. The per-problem output tiling is supplied by a vendor-specific
// selector kept internal to this impl.
class SubgroupMatrixMatMulImplInternal final : public SubgroupMatrixMatMulImpl {
 public:
  SubgroupMatrixMatMulImplInternal(SubgroupMatrixConfig config, SubgroupMatrixTilingSelector tiling_selector)
      : config_(config),
        tiling_selector_(std::move(tiling_selector)) {}

  bool CanApply(const ComputeContext& context,
                const std::vector<const Tensor*>& inputs,
                bool is_channels_last,
                bool b_is_constant) const override {
    const auto* a = inputs[0];
    const auto* b = inputs[1];
    const auto& a_shape = a->Shape();
    const auto& b_shape = b->Shape();
    const bool has_bias = inputs.size() > 2;
    const size_t a_rank = a_shape.NumDimensions();
    const size_t b_rank = b_shape.NumDimensions();
    if ((!is_channels_last && has_bias) || a_rank < 2 || b_rank < 2 ||
        !a->IsDataType<MLFloat16>() || !b->IsDataType<MLFloat16>()) {
      return false;
    }

    const uint32_t K = narrow<uint32_t>(a_shape[a_rank - 1]);
    if (K == 0) {
      return false;
    }

    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t batch = 1;
    if (b_rank == 2) {
      if (narrow<uint32_t>(b_shape[0]) != K) {
        return false;
      }
      M = narrow<uint32_t>(a_shape.Size() / static_cast<int64_t>(K));
      N = narrow<uint32_t>(b_shape[1]);
    } else {
      if (a_rank != b_rank || narrow<uint32_t>(b_shape[b_rank - 2]) != K) {
        return false;
      }
      M = narrow<uint32_t>(a_shape[a_rank - 2]);
      N = narrow<uint32_t>(b_shape[b_rank - 1]);
      for (size_t i = 0; i + 2 < a_rank; ++i) {
        if (a_shape[i] != b_shape[i]) {
          return false;
        }
      }
      batch = narrow<uint32_t>(a_shape.SizeToDimension(a_rank - 2));
    }
    if (M == 0 || N == 0) {
      return false;
    }

    const std::optional<SubgroupMatrixTiling> tiling = tiling_selector_(context, M, N, K, batch);
    if (!tiling) {
      return false;
    }

    const auto& config = config_;
    const uint32_t n_align = config.Is(16, 16, 16) ? 8 : 2;
    const bool needs_padded_b = N % n_align != 0;
    if (needs_padded_b && (!b_is_constant || N > std::numeric_limits<uint32_t>::max() - n_align)) {
      return false;
    }
    const uint32_t n_b = needs_padded_b ? (N / n_align + 1) * n_align : N;
    return config.K != 0 && K % config.K == 0 &&
           M >= tiling->tile_m && n_b >= tiling->tile_n;
  }

  Status Compute(ComputeContext& context,
                 const std::vector<const Tensor*>& inputs,
                 Tensor* output,
                 const Activation& activation,
                 bool is_channels_last,
                 bool b_is_constant) override {
    ORT_RETURN_IF_NOT(inputs.size() >= 2,
                      "MatMul algorithm subgroup_matrix requires two inputs.");
    const auto* a = inputs[0];
    const auto* b = inputs[1];
    const auto& a_shape = a->Shape();
    const auto& b_shape = b->Shape();
    const bool has_bias = inputs.size() > 2;
    const size_t a_rank = a_shape.NumDimensions();
    const size_t b_rank = b_shape.NumDimensions();
    ORT_RETURN_IF_NOT((is_channels_last || !has_bias) && a_rank >= 2 && b_rank >= 2 &&
                          a->IsDataType<MLFloat16>() && b->IsDataType<MLFloat16>(),
                      "MatMul algorithm subgroup_matrix does not support these inputs.");
    const uint32_t K = narrow<uint32_t>(a_shape[a_rank - 1]);
    ORT_RETURN_IF_NOT(K != 0,
                      "MatMul algorithm subgroup_matrix requires a nonzero contraction dimension.");

    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t batch = 1;
    if (b_rank == 2) {
      ORT_ENFORCE(narrow<uint32_t>(b_shape[0]) == K,
                  "MatMul contraction dim mismatch: A K=", K, " vs B rows=", b_shape[0]);
      M = narrow<uint32_t>(a_shape.Size() / static_cast<int64_t>(K));
      N = narrow<uint32_t>(b_shape[1]);
    } else {
      ORT_ENFORCE(narrow<uint32_t>(b_shape[b_rank - 2]) == K,
                  "MatMul contraction dim mismatch: A K=", K,
                  " vs B rows=", b_shape[b_rank - 2]);
      M = narrow<uint32_t>(a_shape[a_rank - 2]);
      N = narrow<uint32_t>(b_shape[b_rank - 1]);
      for (size_t i = 0; i + 2 < a_rank; ++i) {
        ORT_ENFORCE(a_shape[i] == b_shape[i]);
      }
      batch = narrow<uint32_t>(a_shape.SizeToDimension(a_rank - 2));
    }
    ORT_RETURN_IF_NOT(M != 0 && N != 0,
                      "MatMul algorithm subgroup_matrix requires nonzero output dimensions.");

    const std::optional<SubgroupMatrixTiling> tiling = tiling_selector_(context, M, N, K, batch);
    ORT_RETURN_IF_NOT(tiling.has_value(),
                      "MatMul algorithm subgroup_matrix has no tiling for these inputs.");

    const auto& config = config_;
    // Vulkan cooperative matrix loads need 16 byte aligned rows so the
    // 16x16x16 config pads B to a multiple of 8 columns.
    const uint32_t n_align = config.Is(16, 16, 16) ? 8 : 2;
    const bool needs_padded_b = N % n_align != 0;
    // Require whole subgroup-matrix K blocks. An unaligned B must be constant
    // because its padded copy is cached by this implementation.
    ORT_RETURN_IF_NOT(config.K != 0 && K % config.K == 0 &&
                          (!needs_padded_b || b_is_constant),
                      "MatMul algorithm subgroup_matrix requires aligned K and a constant unaligned-N weight.");

    // N_b is just N rounded up to a multiple of n_align. Compute it before padding so
    // the tile-fit check below can bail out without a wasted pad dispatch.
    uint32_t N_b = N;
    if (needs_padded_b) {
      ORT_RETURN_IF_NOT(N <= std::numeric_limits<uint32_t>::max() - n_align,
                        "Cannot pad B because its padded width exceeds uint32_t range.");
      N_b = (N / n_align + 1) * n_align;
    }

    // The kernel keeps its operand loads in bounds by shifting a trailing partial
    // tile back, which is only possible when the tile fits within M and N.
    ORT_ENFORCE(M >= tiling->tile_m && N_b >= tiling->tile_n);

    // The optimized path will run: now materialize the aligned B for unaligned N.
    const Tensor* b_used = b;
    if (needs_padded_b) {
      ORT_RETURN_IF_ERROR(EnsurePaddedB(context, *b, b_shape, N, N_b));
      b_used = padded_b_.get();
    }

    const Tensor* bias = has_bias ? inputs[2] : nullptr;

    const uint32_t tile_m = tiling->tile_m;
    const uint32_t tile_n = tiling->tile_n;
    const uint32_t split_k = tiling->split_k;
    const uint32_t sg_mat_count_m = tile_m / config.M;
    const uint32_t sg_mat_count_n = tile_n / config.N;
    ORT_ENFORCE(tile_m % config.M == 0 && tile_n % config.N == 0,
                "Tiling must be a multiple of the subgroup-matrix shape: ",
                tile_m, "x", tile_n, " vs ", config.M, "x", config.N);
    const uint32_t dispatch_x = (N + tile_n - 1) / tile_n;
    const uint32_t dispatch_y = (M + tile_m - 1) / tile_m;

    SubgroupMatrixMatMulProgram program{activation, has_bias, config,
                                        sg_mat_count_m, sg_mat_count_n, split_k};
    program.SetWorkgroupSize(config.subgroupSize * split_k);
    if (context.HasFeature(wgpu::FeatureName::SubgroupSizeControl)) {
      program.SetSubgroupSize(config.subgroupSize);
    }
    program.SetDispatchGroupSize(dispatch_x, dispatch_y, batch);
    program.CacheHint(activation.CacheKey(), has_bias,
                      static_cast<uint32_t>(config.componentType),
                      static_cast<uint32_t>(config.resultComponentType),
                      config.M, config.N, config.K, config.subgroupSize,
                      sg_mat_count_m, sg_mat_count_n, split_k)
        .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, 1},
                    {b_used, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .AddOutput({output, ProgramTensorMetadataDependency::Rank, 1})
        .AddUniformVariables({{M}, {N}, {K}, {dispatch_x}, {N_b}});
    // Activation uniforms must remain last because definitions and values are matched by index.
    AppendActivationUniformsData(activation, program);
    if (has_bias) {
      program.AddInput({bias, ProgramTensorMetadataDependency::None});
    }
    ORT_RETURN_IF_ERROR(context.RunProgram(program));

    return Status::OK();
  }

 private:
  // Lazily builds an aligned copy of a constant weight B with unaligned N, caching
  // it so the per-run pad cost is paid once. Only valid when B is a constant
  // initializer (checked by the caller) - a runtime B changes per run and must not
  // be cached.
  Status EnsurePaddedB(ComputeContext& context,
                       const Tensor& b,
                       const TensorShape& b_shape,
                       uint32_t N,
                       uint32_t n_b) const {
    std::call_once(pad_once_, [&]() {
      std::unique_ptr<Tensor> padded;
      if (PadSubgroupMatrixWeightColumns(context, b, b_shape, N, n_b, padded).IsOK()) {
        padded_b_ = std::move(padded);
      }
    });
    // padded_b_ persists the outcome across calls: call_once runs the body only on
    // the first call, so a failed pad stays failed (and null) on later calls.
    return padded_b_ ? Status::OK()
                     : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to pad B for subgroup-matrix MatMul.");
  }

  const SubgroupMatrixConfig config_;
  SubgroupMatrixTilingSelector tiling_selector_;

  // Cached aligned B for unaligned N; built once by EnsurePaddedB.
  mutable std::once_flag pad_once_;
  mutable std::unique_ptr<Tensor> padded_b_;
};

Status GenerateSubgroupMatrixShaderCode(ShaderHelper& shader, const SubgroupMatrixConfig& config,
                                        uint32_t sg_mat_count_m, uint32_t sg_mat_count_n,
                                        uint32_t split_k) {
  return WGSL_TEMPLATE_APPLY(shader, "math/subgroup_matrix_matmul.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(sg_mat_count_m, sg_mat_count_m),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_count_n, sg_mat_count_n),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_k, config.K),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_m, config.M),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_n, config.N),
                             WGSL_TEMPLATE_PARAMETER(split_k, split_k));
}

// Default tiling used on any vendor without a specialized policy: a 64x64 output
// tile split two ways along K when M and N are at least 64 and otherwise a 32x32
// output tile split four ways along K.
SubgroupMatrixTilingSelector MakeDefaultTilingSelector() {
  return [](const ComputeContext&, uint32_t M, uint32_t N,
            uint32_t /*K*/, uint32_t /*batch*/) -> std::optional<SubgroupMatrixTiling> {
    if (M >= 64 && N >= 64) {
      return SubgroupMatrixTiling{64, 64, 2};
    }
    return SubgroupMatrixTiling{32, 32, 4};
  };
}

}  // namespace

Status SubgroupMatrixMatMulProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform);
  shader.AddInput("input_b", ShaderUsage::UseUniform);
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput(
      "output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  auto& additional_implementation = shader.AdditionalImplementation();
  additional_implementation
      << GetActivationDeclaration(activation_, "output_value_t", "output_element_t")
      << "fn write_output(output_offset: u32, bias_offset: u32, value_in: output_value_t) {\n"
         "  var value = value_in;\n";
  if (has_bias_) {
    additional_implementation << "  value += output_value_t(bias[bias_offset]);\n";
  }
  additional_implementation
      << "  " << GetActivationSnippet(activation_, "output_value_t", "output_element_t") << "\n"
      << "  " << output.SetByOffset("output_offset", "value") << "\n"
      << "}\n";

  if (config_.Is(8, 16, 16) || config_.Is(16, 16, 16)) {
    return GenerateSubgroupMatrixShaderCode(shader, config_, sg_mat_count_m_, sg_mat_count_n_, split_k_);
  }
  return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED,
                "Unsupported subgroup matrix config dimensions.");
}

std::unique_ptr<SubgroupMatrixMatMulImpl> CreateSubgroupMatrixMatMulImpl(const ComputeContextBase& context) {
  // Only run on devices that report an F16 subgroup-matrix config this kernel
  // implements (8x16x16 or 16x16x16) and can provide its required subgroup size.
  constexpr auto kF16 = wgpu::SubgroupMatrixComponentType::F16;
  const auto config = SelectSubgroupMatrixConfig(context, {{kF16, kF16, 8, 16, 16, 32, false},
                                                           {kF16, kF16, 16, 16, 16, 32, false}});
  if (!config) {
    return nullptr;
  }
  // Intel GPUs use a tuned/heuristic tiling policy; every other vendor falls back
  // to a default tiling.
  const bool is_intel = context.AdapterInfo().vendor == std::string_view{"intel"};
  SubgroupMatrixTilingSelector tiling_selector =
      is_intel ? intel::CreateSubgroupMatrixTilingSelector(context) : MakeDefaultTilingSelector();
  if (!tiling_selector) {
    return nullptr;
  }
  return std::make_unique<SubgroupMatrixMatMulImplInternal>(*config, std::move(tiling_selector));
}

}  // namespace webgpu
}  // namespace onnxruntime
