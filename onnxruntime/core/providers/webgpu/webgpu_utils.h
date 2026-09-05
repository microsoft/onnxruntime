// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string_view>
#include <utility>
#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/webgpu/webgpu_external_header.h"
#include "core/providers/webgpu/nn/fuse_utils.h"

namespace onnxruntime {
namespace webgpu {

class ShaderVariableHelper;

template <typename T>
inline T CeilDiv(T numerator, T denominator) {
  return (numerator + denominator - 1) / denominator;
}

/**
 * Returns the maximum number of components `N` to be used as `vecN` for the given size.
 */
inline int GetMaxComponents(int64_t size) {
  if (size % 4 == 0) {
    return 4;
  } else if (size % 2 == 0) {
    return 2;
  }
  return 1;
}

inline bool IsNvidiaAdapter(const wgpu::AdapterInfo& adapter_info) {
  return adapter_info.vendor == std::string_view{"nvidia"};
}

/**
 * The device capabilities the packed-tile workgroup rule keys off.
 *
 * Read from wgpu::AdapterInfo and wgpu::Limits by GetPackedTileCaps so the rule itself takes
 * plain scalars: wgpu::AdapterInfo has const members and cannot be synthesized by a test.
 */
struct PackedTileCaps {
  // The adapter's reported subgroup size, or 0 when it reports none.
  //
  // This is wgpu::AdapterInfo::subgroupMinSize, not subgroupMaxSize: on D3D12 the max is
  // derived from WaveLaneCountMax, which Dawn documents as unreliable and "not intended to be
  // used" (D3D12Info.cpp). The min comes from WaveLaneCountMin, which Dawn does rely on, and is
  // what contrib_ops/webgpu/quantization/matmul_nbits.cc reads. NVIDIA reports 32 for both, so
  // the rule below derives the same value either way on the hardware it is gated to.
  uint32_t subgroup_size = 0;
  // wgpu::Limits::maxComputeWorkgroupSizeY.
  uint32_t max_workgroup_size_y = 0;
  // wgpu::Limits::maxComputeInvocationsPerWorkgroup.
  uint32_t max_invocations_per_workgroup = 0;
  // The warp-scheduler count the rule targets is an NVIDIA hardware fact rather than a WebGPU
  // capability, so callers keep the rule vendor-gated on this.
  bool is_nvidia = false;
};

PackedTileCaps GetPackedTileCaps(const wgpu::AdapterInfo& adapter_info, const wgpu::Limits& limits);

/**
 * Derives the y dimension of a packed MatMul/Conv2d workgroup from WebGPU capabilities,
 * holding the A tile constant.
 *
 * The workgroup is sized to hold a whole number of subgroups -- `subgroups_per_workgroup` of
 * them, at the size the adapter reports -- then clamped to the workgroup limits that
 * ShaderHelper::Init enforces.
 *
 * Why this helps: `mm_Asub` and `mm_Bsub` are sized from tile_a_outer, tile_inner and
 * tile_b_outer (see MakeMatMulPackedVec4Source), so pinning workgroup_size_y *
 * elements_per_thread_y to `tile_a_outer` leaves shared memory per workgroup unchanged while
 * raising the invocations per workgroup. Where shared memory is what caps how many workgroups
 * are co-resident on a core -- the usual case for a tiled GEMM -- more invocations per
 * workgroup means proportionally more resident threads. The dispatch grid is unchanged too,
 * because the tile still covers the same rows of A.
 *
 * WebGPU deliberately exposes no core count, achieved occupancy, register-file size or shared
 * memory per core, so `subgroups_per_workgroup` stays a caller-supplied, empirically tuned
 * value. The subgroup size and the workgroup limits it is combined with are both queried.
 *
 * Preconditions, which callers are expected to static_assert since every argument other than
 * `caps` is a compile-time constant at the call site: `workgroup_size_x`,
 * `subgroups_per_workgroup` and `default_workgroup_size_y` are non-zero, and
 * `default_workgroup_size_y` divides `tile_a_outer`.
 *
 * @return {workgroup_size_y, elements_per_thread_y}, whose product is always `tile_a_outer`.
 *         Falls back to the default y dimension when the adapter reports no subgroup size, or
 *         when the derived size would exceed a device limit or break the tile invariant. A
 *         fallback on a vendor-gated adapter is logged, since it silently disables the tuning.
 */
std::pair<uint32_t, int64_t> SelectSubgroupAlignedTileConfigY(const PackedTileCaps& caps,
                                                              uint32_t workgroup_size_x,
                                                              uint32_t subgroups_per_workgroup,
                                                              uint32_t tile_a_outer,
                                                              uint32_t default_workgroup_size_y);

/**
 * Returns a string representing a WGSL expression that sums the components of a value T.
 *
 * T can be a scalar S, vec2<S> or vec4<S>.
 */
inline std::string SumVector(std::string x, int components) {
  switch (components) {
    case 1:
      return x;
    case 2:
      return "(" + x + ".x + " + x + ".y" + ")";
    case 4:
      return "(" + x + ".x + " + x + ".y + " + x + ".z + " + x + ".w" + ")";
    default:
      ORT_THROW("Unsupported number of components: ", components);
  }
}

inline std::string MakeScalarOrVectorType(int components, std::string_view data_type) {
  switch (components) {
    case 1:
      return std::string{data_type};
    case 2:
      return MakeStringWithClassicLocale("vec2<", data_type, ">");
    case 3:
      return MakeStringWithClassicLocale("vec3<", data_type, ">");
    case 4:
      return MakeStringWithClassicLocale("vec4<", data_type, ">");
    default:
      ORT_THROW("Unsupported number of components: ", components);
  }
}

TensorShape ReduceShapeByComponents(const TensorShape& shape, int64_t components);

/**
 * Create a reshaped tensor from an existing tensor.
 *
 * The specified new shape must have the same number of elements as the original tensor.
 *
 * The new tensor is a "view" of the original tensor. It uses the same data of the original tensor.
 * The new tensor does not take or share ownership of the underlying data. The original tensor must outlive the new tensor.
 */
inline Tensor CreateTensorView(const Tensor& tensor, const TensorShape& new_shape) {
  ORT_ENFORCE(tensor.Shape().Size() == new_shape.Size(), "Cannot reshape tensor ", tensor.Shape().ToString(), " to ", new_shape.ToString());
  return {tensor.DataType(), new_shape, const_cast<void*>(tensor.DataRaw()), tensor.Location()};
}

/**
 * Create a reinterpreted tensor from an existing tensor with a new data type and shape.
 *
 * The new data type and shape must match the original tensor's storage size.
 *
 * The new tensor is a "view" of the original tensor. It uses the same data of the original tensor.
 * The new tensor does not take or share ownership of the underlying data. The original tensor must outlive the new tensor.
 */
inline Tensor CreateTensorView(const Tensor& tensor, MLDataType new_data_type, const TensorShape& new_shape) {
  auto byte_size = Tensor::CalculateTensorStorageSize(tensor.DataType(), tensor.Shape());
  auto new_byte_size = Tensor::CalculateTensorStorageSize(new_data_type, new_shape);
  ORT_ENFORCE(byte_size == new_byte_size,
              "Cannot reshape tensor ", tensor.Shape().ToString(), " to ", new_shape.ToString(),
              " with data type ", DataTypeImpl::ToString(new_data_type), ". The byte size of the original tensor is ",
              byte_size, " and the byte size of the new tensor is ", new_byte_size);
  return {new_data_type, new_shape, const_cast<void*>(tensor.DataRaw()), tensor.Location()};
}

/**
 * Configuration for Split-K optimization (Conv|MatMul).
 */
class SplitKConfig {
 public:
  explicit SplitKConfig(const wgpu::AdapterInfo& adapter_info);

  bool UseSplitK(
      bool is_vec4, ActivationKind activation_kind, uint64_t batch_size,
      uint32_t dim_a_outer, uint32_t dim_b_outer, uint32_t dim_inner, bool is_channels_last = true) const;

  uint32_t GetSplitDimInner() const;

 private:
  bool enable_split_k_ = false;
  uint32_t split_dim_inner_ = 0;
  uint32_t min_dim_inner_with_split_k_ = 0;
  uint32_t max_batch_size_ = 0;

  uint32_t GetMaxDimInnerWithSplitK() const;

  struct ConfigAtRange {
    ConfigAtRange(uint32_t max_dim_inner, double rate);
    uint32_t max_dim_inner_with_rate = 0;
    double max_dim_a_outer_x_dim_b_outer_x_batch_size_divides_dim_inner = 0.0;
  };
  std::vector<ConfigAtRange> configs_per_dim_inner_range_;
};

/**
 * Generates WGSL (WebGPU Shading Language) code for performing an atomic add operation
 * on a non-integer value (e.g., floating-point) in a shader.
 *
 * Since WGSL natively supports atomic operations only on integer types, this function
 * generates code that emulates atomic addition for non-integer types using a compare-and-swap loop.
 *
 * @param output        A reference to the ShaderVariableHelper representing the atomic variable
 *                      to be updated. This encapsulates the variable's name and access logic.
 * @param offset        The offset or index within the atomic variable where the operation is applied.
 * @param output_type   The WGSL type of the value being added (e.g., "f32").
 * @param add_value     The expression or variable representing the value to add.
 * @return              A string containing the generated WGSL code for the atomic add operation.
 */
std::string GenerateAtomicAddNonIntegerCode(const ShaderVariableHelper& output, const std::string& offset, const std::string& output_type, const std::string& add_value);

}  // namespace webgpu
}  // namespace onnxruntime
