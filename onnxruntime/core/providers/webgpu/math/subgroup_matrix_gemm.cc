// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(__wasm__)

#include "core/providers/webgpu/math/subgroup_matrix_gemm.h"

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

// Subgroup-matrix Gemm implementation. Loads A and B directly from global memory
// (transposed operands via column-major loads) and runs the cooperative
// subgroup-matrix kernel during Compute. Y = alpha * op(A) @ op(B) + beta * C.
// f16 only; anything the kernel cannot serve (odd load stride, non-f16, K not a
// multiple of the tile) falls back to the generic Gemm path. The per-problem
// output tiling is supplied by a vendor-specific selector kept internal to this
// impl (shared with the subgroup-matrix MatMul kernel).
class SubgroupMatrixGemmImpl final : public Gemm::GemmOptImpl {
 public:
  SubgroupMatrixGemmImpl(const Gemm& parent, int32_t config_index,
                         SubgroupMatrixTilingSelector tiling_selector)
      : Gemm::GemmOptImpl(parent),
        config_index_(config_index),
        tiling_selector_(std::move(tiling_selector)) {}

  Status Compute(ComputeContext& context, /*out*/ bool& handled) override {
    handled = false;

    const auto* a = context.Input<Tensor>(0);
    const auto* b = context.Input<Tensor>(1);
    if (a == nullptr || b == nullptr) {
      return Status::OK();
    }
    const auto& a_shape = a->Shape();
    const auto& b_shape = b->Shape();
    if (a_shape.NumDimensions() != 2 || b_shape.NumDimensions() != 2 ||
        !a->IsDataType<MLFloat16>() || !b->IsDataType<MLFloat16>()) {
      return Status::OK();
    }

    const bool trans_a = parent_.TransA();
    const bool trans_b = parent_.TransB();
    const uint32_t M = narrow<uint32_t>(trans_a ? a_shape[1] : a_shape[0]);
    const uint32_t K = narrow<uint32_t>(trans_a ? a_shape[0] : a_shape[1]);
    const uint32_t N = narrow<uint32_t>(trans_b ? b_shape[0] : b_shape[1]);
    if (M == 0 || N == 0 || K == 0) {
      return Status::OK();
    }

    // Even-stride constraint: Intel's f16 subgroupMatrixLoad reads the contiguous
    // (minor) dimension in 32-bit (2xf16) pairs, so the load row stride must be
    // even. The B row-major load (trans_b == false) strides by N; the A
    // column-major load (trans_a == true) strides by M. The other two cases
    // stride by K, which is already a multiple of the tile size. Fall back when
    // the relevant stride is odd.
    if ((!trans_b && (N % 2 != 0)) || (trans_a && (M % 2 != 0))) {
      return Status::OK();
    }

    const auto* c = context.Input<Tensor>(2);
    const float alpha = parent_.Alpha();
    const float beta = parent_.Beta();
    const bool has_c = c != nullptr && beta != 0.0f;

    // Encode C's broadcast to [M, N] as element strides (0 for a broadcast dim).
    // C is unidirectionally broadcastable to (M, N): its (optional) trailing two
    // dims must each be the matching output extent or 1; any leading dims must be 1.
    uint32_t c_stride_m = 0;
    uint32_t c_stride_n = 0;
    if (has_c) {
      if (!c->IsDataType<MLFloat16>()) {
        return Status::OK();
      }
      const auto& c_shape = c->Shape();
      const size_t c_rank = c_shape.NumDimensions();
      const int64_t rn = c_rank >= 1 ? c_shape[c_rank - 1] : 1;
      const int64_t rm = c_rank >= 2 ? c_shape[c_rank - 2] : 1;
      for (size_t i = 0; i + 2 < c_rank; ++i) {
        if (c_shape[i] != 1) {
          return Status::OK();
        }
      }
      if (!((rn == static_cast<int64_t>(N) || rn == 1) &&
            (rm == static_cast<int64_t>(M) || rm == 1))) {
        return Status::OK();
      }
      c_stride_n = (rn == static_cast<int64_t>(N)) ? 1u : 0u;
      c_stride_m = (rm == static_cast<int64_t>(M)) ? narrow<uint32_t>(rn) : 0u;
    }

    // Tile shape and split-K come from the vendor selector, which also enforces
    // the subgroup-matrix K alignment (K % sg_mat_k == 0) and declines otherwise.
    // Gemm is 2D (no batch), so batch is fixed at 1; the cooperative-matrix kernel
    // shape matches MatMul's, so the same selection and pretuned table apply.
    const std::optional<SubgroupMatrixTiling> tiling = tiling_selector_(context, M, N, K, /*batch=*/1);
    if (!tiling) {
      return Status::OK();
    }

    TensorShape output_shape{{static_cast<int64_t>(M), static_cast<int64_t>(N)}};
    auto* output = context.Output(0, output_shape);
    if (output->Shape().Size() == 0) {
      handled = true;
      return Status::OK();
    }

    const auto& config = supported_subgroup_matrix_configs[config_index_];
    const uint32_t tile_m = tiling->tile_m;
    const uint32_t tile_n = tiling->tile_n;
    const uint32_t split_k = tiling->split_k;
    ORT_ENFORCE(tile_m % config.M == 0 && tile_n % config.N == 0,
                "Tiling must be a multiple of the subgroup-matrix shape: ",
                tile_m, "x", tile_n, " vs ", config.M, "x", config.N);
    const uint32_t sg_mat_count_m = tile_m / config.M;
    const uint32_t sg_mat_count_n = tile_n / config.N;
    const uint32_t dispatch_x = (N + tile_n - 1) / tile_n;
    const uint32_t dispatch_y = (M + tile_m - 1) / tile_m;

    SubgroupMatrixGemmProgram program{has_c, trans_a, trans_b, config_index_, sg_mat_count_m, sg_mat_count_n, split_k};
    program.SetWorkgroupSize(kSubgroupMatrixSubgroupSize * split_k);
    program.SetDispatchGroupSize(dispatch_x, dispatch_y, 1);
    program.CacheHint(has_c, trans_a, trans_b, config_index_, sg_mat_count_m, sg_mat_count_n, split_k)
        .AddInputs({{a, ProgramTensorMetadataDependency::TypeAndRank, 1},
                    {b, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .AddOutput({output, ProgramTensorMetadataDependency::Rank, output->Shape(), 1})
        .AddUniformVariables({{M}, {N}, {K}, {alpha}, {beta}, {c_stride_m}, {c_stride_n}});
    if (has_c) {
      program.AddInput({c, ProgramTensorMetadataDependency::None});
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
                                 bool has_c, bool trans_a, bool trans_b,
                                 uint32_t sg_mat_count_m, uint32_t sg_mat_count_n, uint32_t split_k) {
  return WGSL_TEMPLATE_APPLY(shader, "math/subgroup_matrix_gemm_8x16x16.wgsl.template",
                             WGSL_TEMPLATE_PARAMETER(has_c, has_c),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_count_m, sg_mat_count_m),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_count_n, sg_mat_count_n),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_k, 16),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_m, 8),
                             WGSL_TEMPLATE_PARAMETER(sg_mat_n, 16),
                             WGSL_TEMPLATE_PARAMETER(split_k, split_k),
                             WGSL_TEMPLATE_PARAMETER(trans_a, trans_a),
                             WGSL_TEMPLATE_PARAMETER(trans_b, trans_b),
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

Status SubgroupMatrixGemmProgram::GenerateShaderCode(ShaderHelper& shader) const {
  shader.AddInput("input_a", ShaderUsage::UseUniform);
  shader.AddInput("input_b", ShaderUsage::UseUniform);
  if (has_c_) {
    shader.AddInput("input_c", ShaderUsage::UseUniform);
  }
  const auto& output = shader.AddOutput("output", ShaderUsage::UseUniform | ShaderUsage::UseElementTypeAlias);

  const auto& config = supported_subgroup_matrix_configs[config_index_];
  if (config.Is(8, 16, 16)) {
    return GenerateShaderCode8x16x16(shader, output, has_c_, trans_a_, trans_b_,
                                     sg_mat_count_m_, sg_mat_count_n_, split_k_);
  }
  return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::NOT_IMPLEMENTED,
                "Unsupported subgroup matrix config dimensions.");
}

std::unique_ptr<Gemm::GemmOptImpl> CreateSubgroupMatrixGemmImpl(
    const Gemm& parent, const ComputeContextBase& context) {
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
  return std::make_unique<SubgroupMatrixGemmImpl>(parent, config_index, std::move(tiling_selector));
}

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
