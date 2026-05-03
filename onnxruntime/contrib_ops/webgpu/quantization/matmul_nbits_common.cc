// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/quantization/matmul_nbits_common.h"

#include <sstream>

#include "core/common/common.h"
#include "contrib_ops/webgpu/quantization/dp4a_matmul_nbits.h"
#include "contrib_ops/webgpu/quantization/subgroup_matrix_matmul_nbits.h"
#include "core/providers/cpu/math/matmul_helper.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_utils.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

std::string GenerateZeroPointReadingCode(uint32_t nbits, bool has_zero_points,
                                         const std::string& output_type) {
  ORT_ENFORCE(nbits == 8 || nbits == 4, "Only 4/8 bits are supported for webgpu matmulnbits");
  std::stringstream ss;

  if (has_zero_points) {
    ss << "const elements_in_uint32 = " << (32 / nbits) << "u;\n"
       << "const bits = " << nbits << "u;\n";
    ss << R"(
fn mm_read_zero(row : u32, col : u32, r_dim: u32, c_dim: u32) -> )"
       << output_type << R"( {
  if (row < r_dim && col < c_dim) {
    let offset = row * c_dim + col;

    // u32 holds elements_in_uint32 packed nbits.
    let array_index = offset / elements_in_uint32;
    let component_index = offset % elements_in_uint32;
    let packed_value = zero_points[array_index];

    // Extract the nbits component
    let shift_amount = component_index * bits;
)";
    ss << "    let masked_value = (packed_value >> shift_amount) & " << (nbits == 4 ? "0xFu" : "0xFF") << ";\n";
    ss << R"(
    return )"
       << output_type << R"((masked_value);
  }
  return )"
       << output_type << R"((0);
}
)";
  } else {
    ss << "const default_zero_point = " << (nbits == 4 ? 8 : 128) << ";\n";
    ss << R"(
fn mm_read_zero(row : u32, col : u32, r_dim: u32, c_dim: u32) -> )"
       << output_type << R"( {
  return )"
       << output_type << R"((default_zero_point);
}
)";
  }

  return ss.str();
}

bool HasDP4ADeviceSupport(int context_id) {
  auto& ctx = onnxruntime::webgpu::WebGpuContextFactory::GetContext(context_id);
  return ctx.DeviceHasFeature(wgpu::FeatureName::Subgroups) &&
         ctx.AdapterInfo().vendor != std::string_view{"apple"};
}

bool WouldApplySubgroupMatrixMatMulNBitsInCurrentDispatch(uint32_t M,
                                                          [[maybe_unused]] uint32_t N,
                                                          [[maybe_unused]] uint32_t K,
                                                          [[maybe_unused]] uint32_t batch_count,
                                                          [[maybe_unused]] uint32_t block_size,
                                                          [[maybe_unused]] int64_t accuracy_level,
                                                          [[maybe_unused]] int64_t nbits,
                                                          [[maybe_unused]] onnxruntime::webgpu::ComputeContext& context,
                                                          [[maybe_unused]] Tensor* y,
                                                          [[maybe_unused]] bool has_weight_idx_indirect,
                                                          [[maybe_unused]] int32_t* subgroup_matrix_config_index,
                                                          [[maybe_unused]] uint32_t override_M) {
  [[maybe_unused]] const uint32_t dispatch_M = override_M > 0 ? override_M : M;

#if !defined(__wasm__)
  int32_t local_subgroup_matrix_config_index = -1;
  if (dispatch_M != M) {
    return false;
  }

  return (M >= kMinMForTileOptimization && !has_weight_idx_indirect) &&
         CanApplySubgroupMatrixMatMulNBits(context,
                                           accuracy_level,
                                           block_size,
                                           batch_count,
                                           N,
                                           K,
                                           static_cast<uint32_t>(nbits),
                                           y->DataType() == DataTypeImpl::GetType<MLFloat16>(),
                                           subgroup_matrix_config_index != nullptr ? *subgroup_matrix_config_index : local_subgroup_matrix_config_index);
#else
  return false;
#endif
}

bool WouldApplySubgroupMatrixMatMulNBitsInCurrentDispatch(const Tensor* a,
                                                          int64_t K_op,
                                                          int64_t N_op,
                                                          int64_t block_size_op,
                                                          int64_t accuracy_level,
                                                          int64_t nbits,
                                                          onnxruntime::webgpu::ComputeContext& context,
                                                          Tensor* y,
                                                          bool has_weight_idx_indirect,
                                                          int32_t* subgroup_matrix_config_index,
                                                          uint32_t override_M) {
  TensorShape b_shape({N_op, K_op});
  MatMulComputeHelper helper;
  if (!helper.Compute(a->Shape(), b_shape, false, true).IsOK()) {
    return false;
  }

  return WouldApplySubgroupMatrixMatMulNBitsInCurrentDispatch(
      onnxruntime::narrow<uint32_t>(helper.M()),
      onnxruntime::narrow<uint32_t>(helper.N()),
      onnxruntime::narrow<uint32_t>(helper.K()),
      onnxruntime::narrow<uint32_t>(helper.OutputOffsets().size()),
      onnxruntime::narrow<uint32_t>(block_size_op),
      accuracy_level,
      nbits,
      context,
      y,
      has_weight_idx_indirect,
      subgroup_matrix_config_index,
      override_M);
}

bool WouldApplyDP4AMatMulNBitsInCurrentDispatch(uint32_t M,
                                                uint32_t N,
                                                uint32_t K,
                                                uint32_t block_size,
                                                int64_t accuracy_level,
                                                onnxruntime::webgpu::ComputeContext& context,
                                                Tensor* y,
                                                bool has_weight_idx_indirect) {
  const uint32_t components_a = GetMaxComponents(K);

  return ((M >= kMinMForTileOptimization && !has_weight_idx_indirect) ||
          y->DataType() == DataTypeImpl::GetType<float>() ||
          context.AdapterInfo().vendor == std::string_view{"qualcomm"}) &&
         CanApplyDP4AMatrixMatMulNBits(context, accuracy_level, block_size, N, K, components_a);
}

bool WouldApplyDP4AMatMulNBitsInCurrentDispatch(const Tensor* a,
                                                int64_t K_op,
                                                int64_t N_op,
                                                int64_t block_size_op,
                                                int64_t accuracy_level,
                                                onnxruntime::webgpu::ComputeContext& context,
                                                Tensor* y,
                                                bool has_weight_idx_indirect) {
  TensorShape b_shape({N_op, K_op});
  MatMulComputeHelper helper;
  if (!helper.Compute(a->Shape(), b_shape, false, true).IsOK()) {
    return false;
  }

  return WouldApplyDP4AMatMulNBitsInCurrentDispatch(
      onnxruntime::narrow<uint32_t>(helper.M()),
      onnxruntime::narrow<uint32_t>(helper.N()),
      onnxruntime::narrow<uint32_t>(helper.K()),
      onnxruntime::narrow<uint32_t>(block_size_op),
      accuracy_level,
      context,
      y,
      has_weight_idx_indirect);
}

bool WouldApplyWideTileMatMulNBitsInCurrentDispatch(uint32_t M,
                                                    uint32_t K,
                                                    uint32_t block_size,
                                                    int64_t nbits,
                                                    bool has_weight_idx_indirect) {
  if (has_weight_idx_indirect) {
    return false;
  }

  const uint32_t components_a = GetMaxComponents(K);
  const uint32_t block_size_per_col = block_size;
  const uint32_t blob_size = (block_size_per_col / 8) * static_cast<uint32_t>(nbits);
  const uint32_t blob_size_in_words = blob_size / 4;
  const uint32_t components_b = GetMaxComponents(blob_size_in_words);

  return block_size == 32 &&
         components_a == 4 &&
         components_b == 4 &&
         nbits != 2 &&
         M >= kMinMForTileOptimization;
}

bool WouldApplyWideTileMatMulNBitsInCurrentDispatch(const Tensor* a,
                                                    int64_t K_op,
                                                    int64_t N_op,
                                                    int64_t block_size_op,
                                                    int64_t nbits,
                                                    bool has_weight_idx_indirect) {
  if (has_weight_idx_indirect) {
    return false;
  }

  TensorShape b_shape({N_op, K_op});
  MatMulComputeHelper helper;
  if (!helper.Compute(a->Shape(), b_shape, false, true).IsOK()) {
    return false;
  }

  return WouldApplyWideTileMatMulNBitsInCurrentDispatch(
      onnxruntime::narrow<uint32_t>(helper.M()),
      onnxruntime::narrow<uint32_t>(helper.K()),
      onnxruntime::narrow<uint32_t>(block_size_op),
      nbits,
      has_weight_idx_indirect);
}

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
