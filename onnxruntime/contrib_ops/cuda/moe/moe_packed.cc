// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/moe_packed.h"

#include <algorithm>
#include <type_traits>

#include "contrib_ops/cpu/quantization/gguf_quant_tables.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_MOE_PACKED_CUDA(T)                                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                         \
      MoEPacked,                                                         \
      kMSDomain, 1,                                                      \
      T,                                                                 \
      kCudaExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())         \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<uint8_t>()), \
      MoEPacked<T>);

REGISTER_MOE_PACKED_CUDA(MLFloat16)
REGISTER_MOE_PACKED_CUDA(BFloat16)

#undef REGISTER_MOE_PACKED_CUDA

template <typename T>
MoEPacked<T>::MoEPacked(const OpKernelInfo& info) : CudaKernel(info) {
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
Status MoEPacked<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* gate_w = context->Input<Tensor>(2);
  const Tensor* up_w = context->Input<Tensor>(3);
  const Tensor* down_w = context->Input<Tensor>(4);

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

  Tensor* output = context->Output(0, in_shape);
  if (num_rows == 0) return Status::OK();
  constexpr bool is_bf16 = std::is_same_v<T, BFloat16>;
  ORT_RETURN_IF_NOT(!is_bf16 || GetDeviceProp().major >= 8,
                    "MoEPacked BF16 requires compute capability 8.0 or higher");

  // Process in chunks of at most 2048 rows so workspace stays compact during large prefills.
  constexpr int64_t kMaxChunkRows = 2048;
  const int64_t max_chunk = std::min<int64_t>(num_rows, kMaxChunkRows);

  MoEPackedParams max_params{};
  max_params.num_rows = static_cast<int>(max_chunk);
  max_params.num_experts = static_cast<int>(num_experts);
  max_params.hidden_size = static_cast<int>(hidden_size);
  max_params.inter_size = static_cast<int>(inter_size);
  max_params.k = static_cast<int>(k_);
  max_params.normalize_routing_weights = (normalize_routing_weights_ != 0);
  max_params.gate_type = static_cast<int>(gate_type_);
  max_params.up_type = static_cast<int>(up_type_);
  max_params.down_type = static_cast<int>(down_type_);
  max_params.gate_row_bytes = exp_gate_rb;
  max_params.up_row_bytes = exp_up_rb;
  max_params.down_row_bytes = exp_down_rb;

  const size_t ws_bytes = GetMoEPackedWorkspaceBytes(max_params, sizeof(T));
  auto workspace = GetScratchBuffer<uint8_t>(ws_bytes, GetComputeStream(context));

  const uint8_t* in_bytes = static_cast<const uint8_t*>(input->DataRaw());
  const uint8_t* rp_bytes = static_cast<const uint8_t*>(router_probs->DataRaw());
  uint8_t* out_bytes = static_cast<uint8_t*>(output->MutableDataRaw());

  for (int64_t r0 = 0; r0 < num_rows; r0 += kMaxChunkRows) {
    const int64_t chunk_rows = std::min<int64_t>(kMaxChunkRows, num_rows - r0);
    MoEPackedParams chunk_params = max_params;
    chunk_params.num_rows = static_cast<int>(chunk_rows);
    ORT_RETURN_IF_ERROR(LaunchMoEPackedCuda(
        Stream(context),
        is_bf16,
        in_bytes + static_cast<size_t>(r0 * hidden_size) * sizeof(T),
        rp_bytes + static_cast<size_t>(r0 * num_experts) * sizeof(T),
        gate_w->Data<uint8_t>(),
        up_w->Data<uint8_t>(),
        down_w->Data<uint8_t>(),
        chunk_params,
        workspace.get(),
        out_bytes + static_cast<size_t>(r0 * hidden_size) * sizeof(T)));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
