// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(__wasm__)

#include "core/providers/webgpu/nn/subgroup_matrix_conv1x1.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "core/common/narrow.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/math/subgroup_matrix_config.h"
#include "core/providers/webgpu/math/subgroup_matrix_matmul.h"
#include "core/providers/webgpu/program.h"
#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/vendor/intel/math/subgroup_matrix_tiling_selector.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {

namespace {

// Lanes per subgroup assumed by the subgroup-matrix kernel. The workgroup runs
// split_k subgroups, so its size is kSubgroupMatrixSubgroupSize * split_k.
constexpr uint32_t kSubgroupMatrixSubgroupSize = 32;

// Default tiling used on any vendor without a specialized policy: a fixed 32x32
// output tile with no split-K.
SubgroupMatrixTilingSelector MakeDefaultTilingSelector() {
  return [](const ComputeContext&, uint32_t /*M*/, uint32_t /*N*/,
            uint32_t /*K*/, uint32_t /*batch*/) -> std::optional<SubgroupMatrixTiling> {
    return SubgroupMatrixTiling{32, 32, 1};
  };
}

// Subgroup-matrix implementation of the 1x1 / same-size Conv matmul. Reuses the
// shared SubgroupMatrixMatMulProgram kernel and vendor tiling policy; the only
// Conv-specific piece is folding the reshaped N,H,W,C operands into a matmul and
// writing into the caller's pre-allocated Conv output. Self-contained so
// subgroup_matrix_matmul.cc stays untouched.
class SubgroupMatrixConv1x1Impl final : public Conv1x1OptImpl {
 public:
  SubgroupMatrixConv1x1Impl(int32_t config_index, SubgroupMatrixTilingSelector tiling_selector)
      : config_index_(config_index), tiling_selector_(std::move(tiling_selector)) {}

  Status Compute(ComputeContext& context,
                 std::vector<const Tensor*>& inputs, Tensor* output,
                 const TensorShape& input_a_reshape,
                 const TensorShape& input_b_reshape,
                 bool w_is_constant,
                 /*out*/ bool& handled) override {
    handled = false;

    const auto* a = inputs[0];
    const auto* b = inputs[1];
    const Tensor* bias = inputs.size() > 2 ? inputs[2] : nullptr;
    TensorShape a_shape = input_a_reshape.NumDimensions() > 0 ? input_a_reshape : a->Shape();
    TensorShape b_shape = input_b_reshape.NumDimensions() > 0 ? input_b_reshape : b->Shape();
    if (a_shape.NumDimensions() < 2 || !a->IsDataType<MLFloat16>() || !b->IsDataType<MLFloat16>()) {
      return Status::OK();
    }

    // When B is a shared 2D weight (its batch folds to 1), fold A's leading dims
    // into M so the problem runs as a single 2D-weight matmul (e.g. the 1x1 Conv
    // [batch,H*W,C] @ [1,C,N] -> [batch*H*W, C] @ [C, N]).
    const int64_t batch_a = a_shape.SizeToDimension(a_shape.NumDimensions() - 2);
    const int64_t batch_b = b_shape.SizeToDimension(b_shape.NumDimensions() - 2);
    if (batch_a != 1 && batch_b == 1) {
      const int64_t k = a_shape[a_shape.NumDimensions() - 1];
      const int64_t n = b_shape[b_shape.NumDimensions() - 1];
      const int64_t batch_and_m = a_shape.SizeToDimension(a_shape.NumDimensions() - 1);
      a_shape = TensorShape({batch_and_m, k});
      b_shape = TensorShape({k, n});
    }

    const uint32_t K = narrow<uint32_t>(a_shape[a_shape.NumDimensions() - 1]);
    // The tiling selector is responsible for any subgroup-matrix alignment
    // requirements (e.g. K % sg_mat_k == 0) and declines otherwise.
    if (K == 0) {
      return Status::OK();
    }

    // Two shapes are handled:
    //  * Shared 2D weight B [K, N]: all leading A dims fold into M and the whole
    //    problem runs as one z-slice (batch == 1).
    //  * Batched B [..., K, N] (true bmm): A is [..., M, K] with batch dims
    //    identical to B (no broadcasting). Each (A, B) pair is one z-slice.
    uint32_t M = 0;
    uint32_t N = 0;
    uint32_t batch = 1;
    if (b_shape.NumDimensions() == 2) {
      ORT_ENFORCE(narrow<uint32_t>(b_shape[0]) == K,
                  "Conv 1x1 matmul contraction dim mismatch: A K=", K, " vs B rows=", b_shape[0]);
      N = narrow<uint32_t>(b_shape[1]);
      M = narrow<uint32_t>(a_shape.Size() / static_cast<int64_t>(K));
    } else {
      const size_t rank = a_shape.NumDimensions();
      if (b_shape.NumDimensions() != rank) {
        return Status::OK();
      }
      ORT_ENFORCE(narrow<uint32_t>(b_shape[rank - 2]) == K,
                  "Conv 1x1 matmul contraction dim mismatch: A K=", K, " vs B rows=", b_shape[rank - 2]);
      M = narrow<uint32_t>(a_shape[rank - 2]);
      N = narrow<uint32_t>(b_shape[rank - 1]);
      for (size_t i = 0; i + 2 < rank; ++i) {
        if (a_shape[i] != b_shape[i]) {
          return Status::OK();
        }
      }
      batch = narrow<uint32_t>(a_shape.SizeToDimension(rank - 2));
    }

    // An empty M or N yields an empty output; let the generic MatMul path allocate
    // it rather than dispatch a degenerate (zero-workgroup) kernel.
    if (M == 0 || N == 0) {
      return Status::OK();
    }

    // The B right-operand is loaded with a row stride of N_b. Intel's f16
    // subgroup-matrix load reads columns in 32-bit (2xf16) pairs and requires each
    // K-row to start 4-byte aligned, i.e. an even element stride. For a constant
    // weight we can pad B to an even stride (N_b = N + 1); a non-constant odd-N B
    // falls back here.
    if (N % 2 != 0 && !w_is_constant) {
      return Status::OK();
    }

    const std::optional<SubgroupMatrixTiling> tiling = tiling_selector_(context, M, N, K, batch);
    if (!tiling) {
      return Status::OK();
    }

    // The optimized path will run: now materialize the even-strided B for odd N.
    const Tensor* b_used = b;
    uint32_t N_b = N;
    if (N % 2 != 0) {
      ORT_RETURN_IF_ERROR(pad_cache_.EnsurePaddedB(context, *b, N, b_used, N_b));
    }

    const bool has_bias = bias != nullptr;
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
                    {b_used, ProgramTensorMetadataDependency::TypeAndRank, 1}})
        .AddOutput({output, ProgramTensorMetadataDependency::Rank, output->Shape(), 1})
        .AddUniformVariables({{M}, {N}, {K}, {dispatch_x}, {N_b}});
    if (has_bias) {
      program.AddInput({bias, ProgramTensorMetadataDependency::None});
    }
    ORT_RETURN_IF_ERROR(context.RunProgram(program));

    handled = true;
    return Status::OK();
  }

 private:
  const int32_t config_index_;
  const SubgroupMatrixTilingSelector tiling_selector_;

  // Odd-N even-strided B cache, shared with the subgroup-matrix MatMul path.
  mutable SubgroupMatrixPadBCache pad_cache_;
};

}  // namespace

std::unique_ptr<Conv1x1OptImpl> CreateSubgroupMatrixConv1x1Impl(const ComputeContextBase& context) {
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
  return std::make_unique<SubgroupMatrixConv1x1Impl>(config_index, std::move(tiling_selector));
}

}  // namespace webgpu
}  // namespace onnxruntime

#endif  // !defined(__wasm__)
