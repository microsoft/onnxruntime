// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/subgroup_matrix_tiling.h"

#include <algorithm>
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
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/vendor/intel/math/subgroup_matrix_tiling_selector.h"
#include "core/providers/webgpu/webgpu_utils.h"
namespace onnxruntime {
namespace webgpu {

// Computes Y = activation(A @ B + optional bias) using subgroupMatrixMultiplyAccumulate.
class SubgroupMatrixMatMulProgram final : public Program<SubgroupMatrixMatMulProgram> {
 public:
  SubgroupMatrixMatMulProgram(const Activation& activation, bool has_bias, SubgroupMatrixConfig config,
                              uint32_t sg_mat_count_m, uint32_t sg_mat_count_n, uint32_t split_k,
                              uint32_t sg_grid_m, uint32_t sg_grid_n, uint32_t tile_k)
      : Program{"SubgroupMatrixMatMul"},
        activation_(activation),
        has_bias_(has_bias),
        config_(config),
        sg_mat_count_m_(sg_mat_count_m),
        sg_mat_count_n_(sg_mat_count_n),
        split_k_(split_k),
        sg_grid_m_(sg_grid_m),
        sg_grid_n_(sg_grid_n),
        tile_k_(tile_k) {}
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
  const uint32_t sg_grid_m_;
  const uint32_t sg_grid_n_;
  const uint32_t tile_k_;
};

namespace {

// Copies a row-major f16 weight B [K, N] into a column-padded [K, N_b] buffer
// (N_b >= N), zero-filling columns [N, N_b). Gives B an aligned row stride so the
// subgroup-matrix f16 load's row-start alignment holds for any N.
class SubgroupMatrixMatMulPadBProgram final : public Program<SubgroupMatrixMatMulPadBProgram> {
 public:
  SubgroupMatrixMatMulPadBProgram() : Program{"SubgroupMatrixMatMulPadB"} {}
  Status GenerateShaderCode(ShaderHelper& shader) const override {
    const auto& input_b = shader.AddInput("input_b", ShaderUsage::UseValueTypeAlias);
    const auto& output = shader.AddOutput("output", ShaderUsage::UseValueTypeAlias);
    return WGSL_TEMPLATE_APPLY(shader, "math/subgroup_matrix_matmul_pad_b.wgsl.template",
                               WGSL_TEMPLATE_VARIABLE(input_b, input_b),
                               WGSL_TEMPLATE_VARIABLE(output, output));
  }
  WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES({"output_size", ProgramUniformVariableDataType::Uint32},
                                          {"N", ProgramUniformVariableDataType::Uint32},
                                          {"N_b", ProgramUniformVariableDataType::Uint32});
};

// A staged tiling reads A and B as vec4 so N must be a multiple of 4 and the template
// unrolls at most 2x2 subgroup matrices per subgroup.
bool IsStagedTilingSupported(const wgpu::Limits& limits, const SubgroupMatrixConfig& config,
                             const SubgroupMatrixTiling& tiling, uint32_t N) {
  const uint32_t cell_m = config.M * tiling.sg_grid_m;
  const uint32_t cell_n = config.N * tiling.sg_grid_n;
  if (N % 4 != 0 || tiling.split_k != 1 || cell_m == 0 || cell_n == 0 || tiling.tile_k == 0 ||
      tiling.tile_k % config.K != 0 || tiling.tile_m % cell_m != 0 || tiling.tile_n % cell_n != 0 ||
      tiling.tile_m / cell_m > 2 || tiling.tile_n / cell_n > 2) {
    return false;
  }
  const uint64_t threads = uint64_t{config.subgroupSize} * tiling.sg_grid_m * tiling.sg_grid_n;
  // Matches the scratch size in the template where each staged row is padded by 8 elements.
  const uint64_t staged_elems = uint64_t{tiling.tile_m} * (tiling.tile_k + 8) + uint64_t{tiling.tile_k} * (tiling.tile_n + 8);
  const uint64_t scratch_bytes = sizeof(uint16_t) * std::max(staged_elems, uint64_t{tiling.tile_m} * tiling.tile_n);
  return threads <= limits.maxComputeInvocationsPerWorkgroup && threads <= limits.maxComputeWorkgroupSizeX &&
         scratch_bytes <= limits.maxComputeWorkgroupStorageSize;
}

// Subgroup-matrix MatMul implementation. Loads both A and B directly from global
// memory (or stages them in shared memory when the tiling sets tile_k) and runs the
// subgroup-matrix kernel during Compute. The class is
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
    if (tiling->tile_k > 0) {
      return config.K != 0 && K % config.K == 0 && IsStagedTilingSupported(context.DeviceLimits(), config, *tiling, N);
    }
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
    // Staged tilings read B through shared memory so they never need the padded copy.
    const bool staged = tiling->tile_k > 0;
    ORT_RETURN_IF_NOT(staged ? IsStagedTilingSupported(context.DeviceLimits(), config, *tiling, N)
                             : tiling->sg_grid_m == 1 && tiling->sg_grid_n == 1,
                      "MatMul algorithm subgroup_matrix got a tiling this device cannot run.");
    // Vulkan cooperative matrix loads need 16 byte aligned rows so the
    // 16x16x16 config pads B to a multiple of 8 columns.
    const uint32_t n_align = config.Is(16, 16, 16) ? 8 : 2;
    const bool needs_padded_b = !staged && N % n_align != 0;
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
    ORT_ENFORCE(staged || (M >= tiling->tile_m && N_b >= tiling->tile_n));

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
    const uint32_t sg_grid_m = tiling->sg_grid_m;
    const uint32_t sg_grid_n = tiling->sg_grid_n;
    const uint32_t tile_k = tiling->tile_k;
    const uint32_t sg_mat_count_m = tile_m / (config.M * sg_grid_m);
    const uint32_t sg_mat_count_n = tile_n / (config.N * sg_grid_n);
    ORT_ENFORCE(tile_m % (config.M * sg_grid_m) == 0 && tile_n % (config.N * sg_grid_n) == 0,
                "Tiling must be a multiple of the subgroup-matrix shape: ",
                tile_m, "x", tile_n, " vs ", config.M, "x", config.N);
    const uint32_t dispatch_x = (N + tile_n - 1) / tile_n;
    const uint32_t dispatch_y = (M + tile_m - 1) / tile_m;

    SubgroupMatrixMatMulProgram program{activation, has_bias, config,
                                        sg_mat_count_m, sg_mat_count_n, split_k,
                                        sg_grid_m, sg_grid_n, tile_k};
    program.SetWorkgroupSize(config.subgroupSize * split_k * sg_grid_m * sg_grid_n);
    if (context.HasFeature(wgpu::FeatureName::SubgroupSizeControl)) {
      program.SetSubgroupSize(config.subgroupSize);
    }
    program.SetDispatchGroupSize(dispatch_x, dispatch_y, batch);
    program.CacheHint(activation.CacheKey(), has_bias,
                      static_cast<uint32_t>(config.componentType),
                      static_cast<uint32_t>(config.resultComponentType),
                      config.M, config.N, config.K, config.subgroupSize,
                      sg_mat_count_m, sg_mat_count_n, split_k, sg_grid_m, sg_grid_n, tile_k)
        .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, staged ? 4 : 1},
                    {b_used, ProgramTensorMetadataDependency::TypeAndRank, staged ? 4 : 1}})
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
  // Lazily builds an aligned copy of a constant weight B [..., K, N] with unaligned N
  // by widening its last dim to an aligned N_b (zero-filling the extra columns) and
  // caches it, so the per-run pad cost is paid once. Works for a 2D weight [K, N]
  // and a batched weight [batch, K, N] alike: the pad pass treats B as a flat
  // [rows, N] -> [rows, N_b] copy over rows = numel / N (= K, or batch*K), which is
  // exactly the even-stride layout the kernel indexes via N_b. Runs the GPU pad pass
  // on first use under call_once; the cached tensor is held for the kernel's
  // lifetime. Only valid when B is a constant initializer (checked by the caller) -
  // a runtime B changes per run and must not be cached.
  Status EnsurePaddedB(ComputeContext& context,
                       const Tensor& b,
                       const TensorShape& b_shape,
                       uint32_t N,
                       uint32_t n_b) const {
    TensorShapeVector padded_dims{b_shape.GetDims().begin(), b_shape.GetDims().end()};
    padded_dims.back() = static_cast<int64_t>(n_b);
    const TensorShape padded_shape{padded_dims};
    const int64_t output_size_i64 = padded_shape.Size();
    ORT_RETURN_IF_NOT(output_size_i64 <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                      "Cannot pad B because the padded tensor has ", output_size_i64,
                      " elements, exceeding uint32_t shader indexing range.");
    const uint32_t output_size = narrow<uint32_t>(output_size_i64);

    std::call_once(pad_once_, [&]() {
      auto padded = std::make_unique<Tensor>(context.CreateGPUTensor(b.DataType(), padded_shape));
      Status s = Status::OK();
      // A zero-element padded tensor (e.g. a zero-batch or empty constant B) needs no
      // pad pass - dispatching 0 workgroups is pointless and some drivers reject it.
      // Just cache the empty tensor; the main kernel dispatches nothing for it.
      if (output_size != 0) {
        SubgroupMatrixMatMulPadBProgram program;
        program.SetWorkgroupSize(WORKGROUP_SIZE)
            .SetDispatchGroupSize(CeilDiv<uint32_t>(output_size, WORKGROUP_SIZE))
            .AddInput({&b, ProgramTensorMetadataDependency::TypeAndRank, b_shape, 1})
            .AddOutput({padded.get(), ProgramTensorMetadataDependency::TypeAndRank, padded->Shape(), 1})
            .AddUniformVariables({{output_size}, {N}, {n_b}});
        s = context.RunProgram(program);
      }
      if (s.IsOK()) {
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
                                        uint32_t split_k, uint32_t sg_grid_m, uint32_t sg_grid_n,
                                        uint32_t tile_k) {
  return WGSL_TEMPLATE_APPLY(shader, "math/subgroup_matrix_matmul.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(sg_grid_m, sg_grid_m),
                             WGSL_TEMPLATE_PARAMETER(sg_grid_n, sg_grid_n),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_count_m, sg_mat_count_m),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_count_n, sg_mat_count_n),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_k, config.K),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_m, config.M),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_n, config.N),
                             WGSL_TEMPLATE_PARAMETER(split_k, split_k),
                             WGSL_TEMPLATE_PARAMETER(tile_k, tile_k));
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

// A 4x4 grid of subgroups stages K slices 64 deep and each subgroup accumulates 2x2 subgroup
// matrices when the problem has at least 128 rows and 256 tiles of 64x64 and 1x1 otherwise.
// Staging starts at 64 rows and below that only for outputs at least 8192 wide because the
// default split K tiling fills the GPU better on small problems.
SubgroupMatrixTilingSelector MakeStagedTilingSelector(const SubgroupMatrixConfig& config) {
  return [config, fallback = MakeDefaultTilingSelector()](
             const ComputeContext& context, uint32_t M, uint32_t N,
             uint32_t K, uint32_t batch) -> std::optional<SubgroupMatrixTiling> {
    if ((M >= 64 || N >= 8192) && K % config.K == 0) {
      const auto& limits = context.DeviceLimits();
      const bool large = M >= 128 && uint64_t{(M + 63) / 64} * ((N + 63) / 64) >= 256;
      const SubgroupMatrixTiling big{config.M * 8, config.N * 8, 1, 4, 4, 64};
      const SubgroupMatrixTiling small{config.M * 4, config.N * 4, 1, 4, 4, 64};
      if (large && IsStagedTilingSupported(limits, config, big, N)) {
        return big;
      }
      if (IsStagedTilingSupported(limits, config, small, N)) {
        return small;
      }
    }
    return fallback(context, M, N, K, batch);
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
    return GenerateSubgroupMatrixShaderCode(shader, config_, sg_mat_count_m_, sg_mat_count_n_, split_k_,
                                            sg_grid_m_, sg_grid_n_, tile_k_);
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
  // Intel and NVIDIA GPUs use a tuned/heuristic tiling policy; every other vendor falls back
  // to a default tiling.
  const auto vendor = context.AdapterInfo().vendor;
  SubgroupMatrixTilingSelector tiling_selector =
      vendor == std::string_view{"intel"}    ? intel::CreateSubgroupMatrixTilingSelector(context)
      : vendor == std::string_view{"nvidia"} ? MakeStagedTilingSelector(*config)
                                             : MakeDefaultTilingSelector();
  if (!tiling_selector) {
    return nullptr;
  }
  return std::make_unique<SubgroupMatrixMatMulImplInternal>(*config, std::move(tiling_selector));
}

}  // namespace webgpu
}  // namespace onnxruntime
