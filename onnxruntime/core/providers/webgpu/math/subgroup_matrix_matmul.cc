// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(__wasm__)

#include "core/providers/webgpu/math/subgroup_matrix_matmul.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>

#include "core/common/narrow.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/vendor/intel/math/subgroup_matrix_tiling_selector.h"
namespace onnxruntime {
namespace webgpu {

namespace {

// Lanes per subgroup assumed by the subgroup-matrix kernel. The workgroup runs
// split_k subgroups, so its size is kSubgroupMatrixSubgroupSize * split_k.
// TODO: use subgroup-size-control to enforce the subgroup size is 32.
constexpr uint32_t kSubgroupMatrixSubgroupSize = 32;

// Subgroup-matrix MatMul implementation. Loads both A and B directly from global
// memory and runs the subgroup-matrix kernel during Compute. The class is
// intended to support all subgroup-matrix configs; for now only 8x16x16 is
// implemented. The per-problem output tiling is supplied by a vendor-specific
// selector kept internal to this impl.
class SubgroupMatrixMatMulImpl final : public MatMul::MatMulOptImpl {
 public:
  SubgroupMatrixMatMulImpl(const MatMul& parent, int32_t config_index,
                           SubgroupMatrixTilingSelector tiling_selector)
      : MatMul::MatMulOptImpl(parent),
        config_index_(config_index),
        tiling_selector_(std::move(tiling_selector)) {}

  Status Compute(ComputeContext& context, /*out*/ bool& handled) override {
    handled = false;

    const auto* a = context.Input(0);
    const auto* b = context.Input(1);
    const auto& a_shape = a->Shape();
    const auto& b_shape = b->Shape();
    if (a_shape.NumDimensions() < 2 || !a->IsDataType<MLFloat16>() || !b->IsDataType<MLFloat16>()) {
      return Status::OK();
    }

    const uint32_t K = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 1]);
    // The tiling selector is responsible for any subgroup-matrix alignment
    // requirements (e.g. K % sg_mat_k == 0) and declines otherwise.
    if (K == 0) {
      return Status::OK();
    }

    // Two shapes are handled:
    //  * Shared 2D weight B [K, N]: all leading A dims fold into M and the whole
    //    problem runs as one z-slice (batch == 1). Optimal for weight matmuls -
    //    B is read once and M-parallelism is maximal.
    //  * Batched B [..., K, N] (true bmm): A is [..., M, K] with batch dims
    //    identical to B (no broadcasting). Each (A, B) pair is one z-dispatched
    //    slice; the kernel derives the per-slice A/B/output strides from M, N, K.
    // Anything else (e.g. broadcasting A across B's batch) falls back.
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t batch = 1;
    if (b_shape.NumDimensions() == 2) {
      ORT_ENFORCE(narrow<uint32_t>(b_shape[0]) == K,
                  "MatMul contraction dim mismatch: A K=", K, " vs B rows=", b_shape[0]);
      N = narrow<uint32_t>(b_shape[1]);
      M = narrow<uint32_t>(a_shape.Size() / static_cast<int64_t>(K));
    } else {
      // Batched B (true bmm): A is [..., M, K], B is [..., K, N]. The kernel pairs
      // slice i of A with slice i of B and copies A's shape to the output, so it
      // can only be correct when A and B have identical batch dims - it does not
      // implement numpy broadcasting. Require equal rank and equal leading dims;
      // anything broadcastable-but-not-identical (e.g. A=[2,1,M,K], B=[1,2,K,N])
      // falls back to the generic MatMul path.
      const size_t rank = a_shape.NumDimensions();
      if (b_shape.NumDimensions() != rank) {
        return Status::OK();
      }
      ORT_ENFORCE(narrow<uint32_t>(b_shape[rank - 2]) == K,
                  "MatMul contraction dim mismatch: A K=", K, " vs B rows=", b_shape[rank - 2]);
      M = narrow<uint32_t>(a_shape[rank - 2]);
      N = narrow<uint32_t>(b_shape[rank - 1]);
      for (size_t i = 0; i + 2 < rank; ++i) {
        if (a_shape[i] != b_shape[i]) {
          return Status::OK();
        }
      }
      batch = narrow<uint32_t>(a_shape.SizeToDimension(rank - 2));
    }

    // An empty M or N yields an empty output (ONNX permits zero-length dims). Let
    // the generic MatMul path allocate the empty result rather than dispatch a
    // degenerate (zero-workgroup) kernel.
    if (M == 0 || N == 0) {
      return Status::OK();
    }

    // The B right-operand is loaded with subgroupMatrixLoad using a row stride of
    // N (uniforms.N). Intel's f16 subgroup-matrix load reads columns in 32-bit
    // (2xf16) pairs and requires each K-row to start 4-byte aligned, i.e. an even
    // element stride. An odd N offsets every other K-row by 2 bytes and corrupts
    // the odd output columns. Fall back to the generic MatMul path for odd N.
    if (N % 2 != 0) {
      return Status::OK();
    }

    const std::optional<SubgroupMatrixTiling> tiling = tiling_selector_(context, M, N, K, batch);
    if (!tiling) {
      return Status::OK();
    }

    TensorShapeVector output_dims{a_shape.GetDims().begin(), a_shape.GetDims().end()};
    output_dims.back() = static_cast<int64_t>(N);
    TensorShape output_shape{output_dims};
    auto* output = context.Output(0, output_shape);

    const bool has_bias = context.InputCount() > 2;
    const Tensor* bias = has_bias ? context.Input(2) : nullptr;

    const auto& config = supported_subgroup_matrix_configs[config_index_];
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

    SubgroupMatrixMatMulProgram program{has_bias, config_index_, sg_mat_count_m, sg_mat_count_n, split_k};
    program.SetWorkgroupSize(kSubgroupMatrixSubgroupSize * split_k);
    program.SetDispatchGroupSize(dispatch_x, dispatch_y, batch);
    program.CacheHint(has_bias, config_index_, sg_mat_count_m, sg_mat_count_n, split_k)
        .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, 1},
                    {b, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .AddOutput({output, ProgramTensorMetadataDependency::Rank, output->Shape(), 1})
        .AddUniformVariables({{M}, {N}, {K}, {dispatch_x}});
    if (has_bias) {
      program.AddInput({bias, ProgramTensorMetadataDependency::None});
    }
    ORT_RETURN_IF_ERROR(context.RunProgram(program));

    handled = true;
    return Status::OK();
  }

 private:
  const int32_t config_index_;
  SubgroupMatrixTilingSelector tiling_selector_;
};

Status GenerateShaderCode8x16x16(ShaderHelper& shader, const ShaderVariableHelper& output,
                                 bool has_bias, uint32_t sg_mat_count_m, uint32_t sg_mat_count_n,
                                 uint32_t split_k) {
  return WGSL_TEMPLATE_APPLY(shader, "math/subgroup_matrix_matmul_8x16x16.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_bias, has_bias),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_count_m, sg_mat_count_m),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_count_n, sg_mat_count_n),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_k, 16),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_m, 8),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_n, 16),
                             WGSL_TEMPLATE_PARAMETER(split_k, split_k),
                             WGSL_TEMPLATE_VARIABLE(output, output));
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

Status SubgroupMatrixMatMulProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform);
  shader.AddInput("input_b", ShaderUsage::UseUniform);
  if (has_bias_) {
    shader.AddInput("bias", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  const auto& config = supported_subgroup_matrix_configs[config_index_];
  if (config.Is(8, 16, 16)) {
    return GenerateShaderCode8x16x16(shader, output, has_bias_, sg_mat_count_m_, sg_mat_count_n_, split_k_);
  }
  return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED,
                "Unsupported subgroup matrix config dimensions.");
}

std::unique_ptr<MatMul::MatMulOptImpl> CreateSubgroupMatrixMatMulImpl(
    const MatMul& parent, const ComputeContextBase& context) {
  // Only run on devices that report the fixed 8x16x16 F16 subgroup-matrix config
  // this kernel is implemented for.
  int32_t config_index = 0;
  if (!IsSubgroupMatrixConfigSupported(context, /*is_fp16=*/true, config_index) ||
      !supported_subgroup_matrix_configs[config_index].Is(8, 16, 16)) {
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
  return std::make_unique<SubgroupMatrixMatMulImpl>(parent, config_index, std::move(tiling_selector));
}

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
