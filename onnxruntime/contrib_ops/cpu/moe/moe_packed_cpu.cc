// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef ORT_MINIMAL_BUILD

#include "contrib_ops/cpu/moe/moe_packed_cpu.h"

#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

#include "contrib_ops/cpu/quantization/gguf_vec_dot_cpu.h"
#include "core/common/safeint.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor_shape.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

namespace {
constexpr int64_t kMoEPackedMaxDimension = std::numeric_limits<int>::max() - 2047;

bool PackedTensorBytesFit(int64_t rows, int64_t columns, size_t element_bytes) {
  size_t row_bytes = 0;
  size_t bytes = 0;
  return rows >= 0 && columns >= 0 &&
         IAllocator::CalcMemSizeForArray(static_cast<size_t>(columns), element_bytes, &row_bytes) &&
         IAllocator::CalcMemSizeForArray(static_cast<size_t>(rows), row_bytes, &bytes) &&
         bytes <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
}
}  // namespace

template <typename T>
MoEPacked<T>::MoEPacked(const OpKernelInfo& info) : OpKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("k", &k_).IsOK() && k_ > 0 && k_ <= 16,
              "MoEPacked requires k in 1..16");
  normalize_routing_weights_ = info.GetAttrOrDefault<int64_t>("normalize_routing_weights", 1);
  activation_type_ = info.GetAttrOrDefault<std::string>("activation_type", "swiglu");
  ORT_ENFORCE(activation_type_ == "swiglu", "MoEPacked only supports activation_type='swiglu'");
  ORT_ENFORCE(info.GetAttr<int64_t>("gate_type", &gate_type_).IsOK(), "MoEPacked requires gate_type");
  ORT_ENFORCE(info.GetAttr<int64_t>("up_type", &up_type_).IsOK(), "MoEPacked requires up_type");
  ORT_ENFORCE(info.GetAttr<int64_t>("down_type", &down_type_).IsOK(), "MoEPacked requires down_type");
  ORT_ENFORCE(gguf_quant::IsGgmlTypeSupported(gate_type_), "MoEPacked: unsupported gate_type=", gate_type_);
  ORT_ENFORCE(gguf_quant::IsGgmlTypeSupported(up_type_), "MoEPacked: unsupported up_type=", up_type_);
  ORT_ENFORCE(gguf_quant::IsGgmlTypeSupported(down_type_), "MoEPacked: unsupported down_type=", down_type_);
}

template <typename T>
Status MoEPacked<T>::Compute(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* gate_w = context->Input<Tensor>(2);
  const Tensor* up_w = context->Input<Tensor>(3);
  const Tensor* down_w = context->Input<Tensor>(4);
  const Tensor* gate_b = context->Input<Tensor>(5);
  const Tensor* up_b = context->Input<Tensor>(6);
  const Tensor* down_b = context->Input<Tensor>(7);

  const auto& in_shape = input->Shape();
  const auto& rp_shape = router_probs->Shape();
  const auto& gw_shape = gate_w->Shape();
  const auto& uw_shape = up_w->Shape();
  const auto& dw_shape = down_w->Shape();

  ORT_RETURN_IF_NOT(in_shape.NumDimensions() >= 2, "MoEPacked input must have rank >= 2");
  ORT_RETURN_IF_NOT(rp_shape.NumDimensions() == 2, "MoEPacked router_probs must be 2D [num_rows, num_experts]");
  ORT_RETURN_IF_NOT(gw_shape.NumDimensions() == 3 && uw_shape.NumDimensions() == 3 && dw_shape.NumDimensions() == 3,
                    "MoEPacked expert weights must be 3D [num_experts, out_dim, row_bytes]");

  const size_t last_axis = in_shape.NumDimensions() - 1;
  const int64_t hidden_size = in_shape[last_axis];
  const int64_t num_rows = in_shape.SizeToDimension(last_axis);
  const int64_t num_experts = rp_shape[1];
  const int64_t inter_size = gw_shape[1];

  ORT_RETURN_IF_NOT(hidden_size > 0 && hidden_size <= kMoEPackedMaxDimension &&
                        inter_size > 0 && inter_size <= kMoEPackedMaxDimension,
                    "MoEPacked requires hidden_size and inter_size in 1..", kMoEPackedMaxDimension);
  ORT_RETURN_IF_NOT(num_rows >= 0, "MoEPacked input tensor has an invalid number of rows");
  ORT_RETURN_IF_NOT(rp_shape[0] == num_rows, "MoEPacked router_probs row count mismatch");
  ORT_RETURN_IF_NOT(num_experts > 0 && num_experts <= 512, "MoEPacked supports up to 512 experts");
  ORT_RETURN_IF_NOT(k_ <= num_experts, "MoEPacked requires k <= num_experts");
  ORT_RETURN_IF_NOT(gw_shape[0] == num_experts && uw_shape[0] == num_experts && dw_shape[0] == num_experts,
                    "MoEPacked expert count mismatch");
  ORT_RETURN_IF_NOT(uw_shape[1] == inter_size && dw_shape[1] == hidden_size,
                    "MoEPacked expert inner/outer dimension mismatch");
  ORT_RETURN_IF_NOT((hidden_size % 32) == 0 && (inter_size % 32) == 0,
                    "MoEPacked requires hidden_size and inter_size to be multiples of 32");

  const int64_t exp_gate_rb = gguf_quant::GgmlRowBytes(static_cast<int>(gate_type_), hidden_size);
  const int64_t exp_up_rb = gguf_quant::GgmlRowBytes(static_cast<int>(up_type_), hidden_size);
  const int64_t exp_down_rb = gguf_quant::GgmlRowBytes(static_cast<int>(down_type_), inter_size);
  ORT_RETURN_IF_NOT(exp_gate_rb > 0 && exp_up_rb > 0 && exp_down_rb > 0 &&
                        gw_shape[2] == exp_gate_rb && uw_shape[2] == exp_up_rb && dw_shape[2] == exp_down_rb,
                    "MoEPacked expert weight row_bytes mismatch");
  ORT_RETURN_IF_NOT(gate_b == nullptr || gate_b->Shape() == TensorShape({num_experts, inter_size}),
                    "MoEPacked gate_experts_bias must have shape [num_experts, inter_size]");
  ORT_RETURN_IF_NOT(up_b == nullptr || up_b->Shape() == TensorShape({num_experts, inter_size}),
                    "MoEPacked up_experts_bias must have shape [num_experts, inter_size]");
  ORT_RETURN_IF_NOT(down_b == nullptr || down_b->Shape() == TensorShape({num_experts, hidden_size}),
                    "MoEPacked down_experts_bias must have shape [num_experts, hidden_size]");
  ORT_RETURN_IF_NOT(num_rows <= std::numeric_limits<int>::max() / k_,
                    "MoEPacked total routes exceed int32 range");
  const int64_t total_routes_i64 = num_rows * k_;
  ORT_RETURN_IF_NOT(PackedTensorBytesFit(num_rows, hidden_size, sizeof(T)) &&
                        PackedTensorBytesFit(num_rows, num_experts, sizeof(T)) &&
                        PackedTensorBytesFit(num_rows, hidden_size, sizeof(float)) &&
                        PackedTensorBytesFit(total_routes_i64, inter_size, sizeof(float)) &&
                        PackedTensorBytesFit(total_routes_i64, hidden_size, sizeof(T)) &&
                        PackedTensorBytesFit(num_experts, SafeInt<int64_t>(inter_size) * exp_gate_rb, sizeof(uint8_t)) &&
                        PackedTensorBytesFit(num_experts, SafeInt<int64_t>(inter_size) * exp_up_rb, sizeof(uint8_t)) &&
                        PackedTensorBytesFit(num_experts, SafeInt<int64_t>(hidden_size) * exp_down_rb, sizeof(uint8_t)),
                    "MoEPacked tensor byte size exceeds the supported address range");

  Tensor* output = context->Output(0, in_shape);
  if (num_rows == 0) return Status::OK();

  const int rows_i = static_cast<int>(num_rows);
  const int experts_i = static_cast<int>(num_experts);
  const int hid_i = static_cast<int>(hidden_size);
  const int inter_i = static_cast<int>(inter_size);
  const int k_i = static_cast<int>(k_);
  const int total_routes = rows_i * k_i;

  std::vector<float> topk_scales(static_cast<size_t>(total_routes));
  std::vector<int> topk_indices(static_cast<size_t>(total_routes));
  gguf_quant::ComputeMoEPackedSoftmaxTopKCpu<T>(
      router_probs->Data<T>(), rows_i, experts_i, k_i, normalize_routing_weights_ != 0,
      topk_scales.data(), topk_indices.data());

  std::vector<int> route_ids(static_cast<size_t>(total_routes));
  std::iota(route_ids.begin(), route_ids.end(), 0);
  std::vector<T> fc2_out(SafeInt<size_t>(total_routes) * hid_i);

  auto* tp = context->GetOperatorThreadPool();
  auto parallel_for = [tp](std::ptrdiff_t count, const auto& fn) {
    concurrency::ThreadPool::TryBatchParallelFor(tp, count, fn, 0);
  };

  gguf_quant::ExecuteMoEPackedColdRoutesCpu<T>(
      parallel_for,
      input->Data<T>(),
      gate_w->Data<uint8_t>(),
      up_w->Data<uint8_t>(),
      down_w->Data<uint8_t>(),
      gate_b != nullptr ? gate_b->Data<T>() : nullptr,
      up_b != nullptr ? up_b->Data<T>() : nullptr,
      down_b != nullptr ? down_b->Data<T>() : nullptr,
      hid_i,
      inter_i,
      k_i,
      static_cast<int>(gate_type_),
      static_cast<int>(up_type_),
      static_cast<int>(down_type_),
      exp_gate_rb,
      exp_up_rb,
      exp_down_rb,
      topk_indices.data(),
      route_ids.data(),
      total_routes,
      fc2_out.data());

  T* out_ptr = output->MutableData<T>();
  auto combine_fn = [&](std::ptrdiff_t row_idx) {
    const int row = static_cast<int>(row_idx);
    const float* row_scales = topk_scales.data() + static_cast<size_t>(row) * k_i;
    const T* row_fc2 = fc2_out.data() + static_cast<size_t>(row) * k_i * hid_i;
    T* dst_row = out_ptr + static_cast<size_t>(row) * hid_i;
    for (int col = 0; col < hid_i; ++col) {
      float sum = 0.0f;
      for (int r = 0; r < k_i; ++r) {
        sum += row_scales[r] * gguf_quant::ActTypeToFloatCpu<T>(row_fc2[static_cast<size_t>(r) * hid_i + col]);
      }
      dst_row[col] = gguf_quant::FloatToActTypeCpu<T>(sum);
    }
  };
  parallel_for(static_cast<std::ptrdiff_t>(rows_i), combine_fn);

  return Status::OK();
}

#define REGISTER_MOE_PACKED_CPU(T)                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      MoEPacked,                                                         \
      kMSDomain, 1,                                                      \
      T,                                                                 \
      kCpuExecutionProvider,                                             \
      (*KernelDefBuilder::Create())                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()), \
      MoEPacked<T>);

REGISTER_MOE_PACKED_CPU(float)
REGISTER_MOE_PACKED_CPU(MLFloat16)
REGISTER_MOE_PACKED_CPU(BFloat16)

#undef REGISTER_MOE_PACKED_CPU

}  // namespace contrib
}  // namespace onnxruntime

#endif  // ORT_MINIMAL_BUILD
