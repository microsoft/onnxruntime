// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/math/subgroup_matrix_matmul.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
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
  SubgroupMatrixMatMulProgram(const Activation& activation, bool has_bias, int32_t config_index,
                              uint32_t sg_mat_count_m, uint32_t sg_mat_count_n, uint32_t split_k)
      : Program{"SubgroupMatrixMatMul"},
        activation_(activation),
        has_bias_(has_bias),
        config_index_(config_index),
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
  const int32_t config_index_;
  const uint32_t sg_mat_count_m_;
  const uint32_t sg_mat_count_n_;
  const uint32_t split_k_;
};

namespace {

// Lanes per subgroup assumed by the subgroup-matrix kernel. The workgroup runs
// split_k subgroups, so its size is kSubgroupMatrixSubgroupSize * split_k.
constexpr uint32_t kSubgroupMatrixSubgroupSize = 32;

bool TryGetPositiveUint32(int64_t dim, uint32_t& value) {
  if (dim <= 0 || dim > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    return false;
  }
  value = static_cast<uint32_t>(dim);
  return true;
}

// Copies a row-major f16 weight B [K, N] into a column-padded [K, N_b] buffer
// (N_b >= N), zero-filling columns [N, N_b). Gives B an even row stride so the
// subgroup-matrix f16 load's 4-byte row-start alignment holds for odd N.
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

// Subgroup-matrix MatMul implementation. Loads both A and B directly from global
// memory and runs the subgroup-matrix kernel during Compute. The class is
// intended to support all subgroup-matrix configs; for now only 8x16x16 is
// implemented. The per-problem output tiling is supplied by a vendor-specific
// selector kept internal to this impl.
class SubgroupMatrixMatMulImpl final : public MatMulOptImpl {
 public:
  SubgroupMatrixMatMulImpl(int32_t config_index, SubgroupMatrixTilingSelector tiling_selector)
      : config_index_(config_index),
        tiling_selector_(std::move(tiling_selector)) {}

  Status Compute(ComputeContext& context,
                 const std::vector<const Tensor*>& inputs,
                 const TensorShape& a_shape,
                 const TensorShape& b_shape,
                 const TensorShape& output_shape,
                 Tensor* output,
                 const Activation& activation,
                 bool is_channels_last,
                 bool b_is_constant,
                 bool has_persistent_cache,
                 /*out*/ bool& handled) override {
    handled = false;

    const auto* a = inputs[0];
    const auto* b = inputs[1];
    const bool has_bias = inputs.size() > 2;
    const bool inputs_are_fp16 = a->IsDataType<MLFloat16>() && b->IsDataType<MLFloat16>();
    const auto problem = AnalyzeSubgroupMatrixMatMulRoute(
        a_shape, b_shape, inputs_are_fp16, is_channels_last, has_bias,
        activation.activation_kind_ != ActivationKind::None);
    if (!problem) {
      return Status::OK();
    }
    const uint32_t M = problem->M;
    const uint32_t N = problem->N;
    const uint32_t K = problem->K;
    const uint32_t batch = problem->batch;

    const std::optional<SubgroupMatrixTiling> tiling = tiling_selector_(context, M, N, K, batch);
    if (!tiling) {
      return Status::OK();
    }

    const auto& config = supported_subgroup_matrix_configs[config_index_];
    // Require whole subgroup-matrix K blocks. Odd N additionally needs a padded
    // constant B owned by a persistent per-kernel cache. Keep this after tiling
    // selection so a declined problem never allocates or dispatches padding.
    if (!CanDispatchSubgroupMatrixMatMul(
            *problem, config.K, b_is_constant, has_persistent_cache)) {
      return Status::OK();
    }

    // The optimized path will run: now materialize the even-strided B for odd N.
    const Tensor* b_used = b;
    TensorShape b_used_shape = b_shape;
    uint32_t N_b = N;
    if (N % 2 != 0) {
      ORT_RETURN_IF_ERROR(EnsurePaddedB(context, *b, b_shape, N));
      b_used = padded_b_.get();
      b_used_shape = b_used->Shape();
      N_b = padded_b_stride_;
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

    SubgroupMatrixMatMulProgram program{activation, has_bias, config_index_,
                                        sg_mat_count_m, sg_mat_count_n, split_k};
    program.SetWorkgroupSize(kSubgroupMatrixSubgroupSize * split_k);
    program.SetSubgroupSize(kSubgroupMatrixSubgroupSize);
    program.SetDispatchGroupSize(dispatch_x, dispatch_y, batch);
    program.CacheHint(activation.CacheKey(), has_bias, config_index_,
                      sg_mat_count_m, sg_mat_count_n, split_k)
        .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, a_shape, 1},
                    {b_used, ProgramTensorMetadataDependency::TypeAndRank, b_used_shape, 1}})
        .AddOutput({output, ProgramTensorMetadataDependency::Rank, output_shape, 1})
        .AddUniformVariables({{M}, {N}, {K}, {dispatch_x}, {N_b}});
    // Activation uniforms must remain last because definitions and values are matched by index.
    AppendActivationUniformsData(activation, program);
    if (has_bias) {
      program.AddInput({bias, ProgramTensorMetadataDependency::None});
    }
    ORT_RETURN_IF_ERROR(context.RunProgram(program));

    handled = true;
    return Status::OK();
  }

 private:
  // Lazily builds an even-strided copy of a constant weight B [..., K, N] with odd N
  // by widening its last dim to N_b = N + 1 (zero-filling the extra column) and
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
                       uint32_t N) const {
    ORT_RETURN_IF_NOT(N < std::numeric_limits<uint32_t>::max(),
                      "Cannot pad odd-N B because N+1 exceeds uint32_t range.");
    const uint32_t n_b = N + 1;
    TensorShapeVector padded_dims{b_shape.GetDims().begin(), b_shape.GetDims().end()};
    padded_dims.back() = static_cast<int64_t>(n_b);
    const TensorShape padded_shape{padded_dims};
    const int64_t output_size_i64 = padded_shape.Size();
    ORT_RETURN_IF_NOT(output_size_i64 <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
                      "Cannot pad odd-N B because the padded tensor has ", output_size_i64,
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
        padded_b_stride_ = n_b;
      }
    });
    // padded_b_ persists the outcome across calls: call_once runs the body only on
    // the first call, so a failed pad stays failed (and null) on later calls.
    return padded_b_ ? Status::OK()
                     : ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to pad odd-N B for subgroup-matrix MatMul.");
  }

  const int32_t config_index_;
  SubgroupMatrixTilingSelector tiling_selector_;

  // Cached even-strided B for odd N; built once by EnsurePaddedB.
  mutable std::once_flag pad_once_;
  mutable std::unique_ptr<Tensor> padded_b_;
  mutable uint32_t padded_b_stride_ = 0;
};

Status GenerateShaderCode8x16x16(ShaderHelper& shader,
                                 uint32_t sg_mat_count_m, uint32_t sg_mat_count_n,
                                 uint32_t split_k) {
  return WGSL_TEMPLATE_APPLY(shader, "math/subgroup_matrix_matmul_8x16x16.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(sg_mat_count_m, sg_mat_count_m),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_count_n, sg_mat_count_n),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_k, 16),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_m, 8),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_n, 16),
                             WGSL_TEMPLATE_PARAMETER(split_k, split_k));
}

// Default tiling used on any vendor without a specialized policy: a fixed 32x32
// output tile with no split-K.
SubgroupMatrixTilingSelector MakeDefaultTilingSelector() {
  return [](const ComputeContext&, uint32_t /*M*/, uint32_t /*N*/,
            uint32_t /*K*/, uint32_t /*batch*/) -> std::optional<SubgroupMatrixTiling> {
    return SubgroupMatrixTiling{32, 32, 1};
  };
}

}  // namespace

std::optional<SubgroupMatrixMatMulProblem> AnalyzeSubgroupMatrixMatMulProblem(
    const TensorShape& a_shape, const TensorShape& b_shape,
    bool is_channels_last, bool has_bias) {
  if ((!is_channels_last && has_bias) ||
      a_shape.NumDimensions() < 2 || b_shape.NumDimensions() < 2) {
    return std::nullopt;
  }

  const size_t a_rank = a_shape.NumDimensions();
  const size_t b_rank = b_shape.NumDimensions();
  uint32_t K = 0;
  uint32_t b_k = 0;
  uint32_t N = 0;
  if (!TryGetPositiveUint32(a_shape[a_rank - 1], K) ||
      !TryGetPositiveUint32(b_shape[b_rank - 2], b_k) ||
      !TryGetPositiveUint32(b_shape[b_rank - 1], N) || K != b_k) {
    return std::nullopt;
  }

  uint32_t M = 1;
  uint32_t batch = 1;
  if (b_rank == 2) {
    for (size_t i = 0; i + 1 < a_rank; ++i) {
      uint32_t dim = 0;
      if (!TryGetPositiveUint32(a_shape[i], dim) ||
          dim > std::numeric_limits<uint32_t>::max() / M) {
        return std::nullopt;
      }
      M *= dim;
    }
  } else {
    if (a_rank != b_rank || !TryGetPositiveUint32(a_shape[a_rank - 2], M)) {
      return std::nullopt;
    }
    for (size_t i = 0; i + 2 < a_rank; ++i) {
      if (a_shape[i] != b_shape[i]) {
        return std::nullopt;
      }
      uint32_t dim = 0;
      if (!TryGetPositiveUint32(a_shape[i], dim) ||
          dim > std::numeric_limits<uint32_t>::max() / batch) {
        return std::nullopt;
      }
      batch *= dim;
    }
  }

  return SubgroupMatrixMatMulProblem{M, N, K, batch};
}

std::optional<SubgroupMatrixMatMulProblem> AnalyzeSubgroupMatrixMatMulRoute(
    const TensorShape& a_shape, const TensorShape& b_shape,
    bool inputs_are_fp16, bool is_channels_last,
    bool has_bias, bool /*has_activation*/) {
  if (!inputs_are_fp16) {
    return std::nullopt;
  }

  return AnalyzeSubgroupMatrixMatMulProblem(a_shape, b_shape, is_channels_last, has_bias);
}

bool CanUseSubgroupMatrixRightOperand(uint32_t N,
                                      bool b_is_constant,
                                      bool has_persistent_cache) {
  return N % 2 == 0 || (b_is_constant && has_persistent_cache);
}

bool CanDispatchSubgroupMatrixMatMul(const SubgroupMatrixMatMulProblem& problem,
                                     uint32_t config_k,
                                     bool b_is_constant,
                                     bool has_persistent_cache) {
  return config_k != 0 && problem.K % config_k == 0 &&
         CanUseSubgroupMatrixRightOperand(problem.N, b_is_constant, has_persistent_cache);
}

std::string BuildSubgroupMatrixMatMulOutputWriter(bool has_bias,
                                                  std::string_view activation_snippet,
                                                  std::string_view output_store_snippet) {
  std::string writer =
      "fn write_output(output_offset: u32, bias_offset: u32, value_in: output_value_t) {\n"
      "  var value = value_in;\n";
  if (has_bias) {
    writer += "  value += output_value_t(bias[bias_offset]);\n";
  }
  writer += "  ";
  writer += activation_snippet;
  writer +=
      "\n"
      "  ";
  writer += output_store_snippet;
  writer +=
      "\n"
      "}\n";
  return writer;
}

Status SubgroupMatrixMatMulProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform);
  shader.AddInput("input_b", ShaderUsage::UseUniform);
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput(
      "output", ShaderUsage::UseUniform | ShaderUsage::UseValueTypeAlias | ShaderUsage::UseElementTypeAlias);
  shader.AdditionalImplementation()
      << GetActivationDeclaration(activation_, "output_value_t", "output_element_t")
      << BuildSubgroupMatrixMatMulOutputWriter(
             has_bias_, GetActivationSnippet(activation_, "output_value_t", "output_element_t"),
             output.SetByOffset("output_offset", "value"));

  const auto& config = supported_subgroup_matrix_configs[config_index_];
  if (config.Is(8, 16, 16)) {
    return GenerateShaderCode8x16x16(shader, sg_mat_count_m_, sg_mat_count_n_, split_k_);
  }
  return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED,
                "Unsupported subgroup matrix config dimensions.");
}

std::unique_ptr<MatMulOptImpl> CreateSubgroupMatrixMatMulImpl(const ComputeContextBase& context) {
  // Only run on devices that report the fixed 8x16x16 F16 subgroup-matrix config
  // this kernel is implemented for. That config's adapters expose a 16-32 subgroup
  // size range, so the kernel's fixed 32 lanes per subgroup must be pinned with
  // subgroup-size control.
  int32_t config_index = 0;
  if (!IsSubgroupMatrixConfigSupported(context, /*is_fp16=*/true, config_index) ||
      !supported_subgroup_matrix_configs[config_index].Is(8, 16, 16) ||
      !context.HasFeature(wgpu::FeatureName::SubgroupSizeControl)) {
    return nullptr;
  }
  // Intel GPUs use a tuned/heuristic tiling policy; every other vendor falls back
  // to a fixed default tiling.
  const bool is_intel = context.AdapterInfo().vendor == std::string_view{"intel"};
  SubgroupMatrixTilingSelector tiling_selector =
      is_intel ? intel::CreateSubgroupMatrixTilingSelector(context) : MakeDefaultTilingSelector();
  if (!tiling_selector) {
    return nullptr;
  }
  return std::make_unique<SubgroupMatrixMatMulImpl>(config_index, std::move(tiling_selector));
}

}  // namespace webgpu
}  // namespace onnxruntime
