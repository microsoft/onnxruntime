// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <type_traits>

#include "contrib_ops/cpu/quantization/gguf_quant_tables.h"
#include "core/common/common.h"
#include "core/common/float16.h"
#include "core/common/inlined_containers.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensor_shape.h"
#include "core/platform/threadpool.h"

namespace onnxruntime {
namespace contrib {

template <typename T2, typename Tind>
class GatherPacked final : public OpKernel {
 public:
  explicit GatherPacked(const OpKernelInfo& info) : OpKernel(info) {
    ORT_ENFORCE(info.GetAttr<int64_t>("K", &k_).IsOK() && k_ > 0, "GatherPacked requires K > 0");
    ORT_ENFORCE(info.GetAttr<int64_t>("packed_type", &ggml_type_).IsOK(), "GatherPacked requires packed_type");
    ORT_ENFORCE(gguf_quant::IsGgmlTypeSupported(ggml_type_), "GatherPacked: unsupported packed_type=", ggml_type_);
    row_bytes_ = gguf_quant::GgmlRowBytes(static_cast<int>(ggml_type_), k_);
    ORT_ENFORCE(row_bytes_ > 0, "GatherPacked: unsupported packed_type=", ggml_type_, " or invalid K=", k_);
  }

  Status Compute(OpKernelContext* context) const override {
    const Tensor* data = context->Input<Tensor>(0);
    const Tensor* indices = context->Input<Tensor>(1);
    const auto& data_shape = data->Shape();
    ORT_RETURN_IF_NOT(data_shape.NumDimensions() == 2, "GatherPacked data must be 2D [vocab_size, row_bytes]");
    const int64_t vocab_size = data_shape[0];
    ORT_RETURN_IF_NOT(data_shape[1] == row_bytes_,
                      "GatherPacked data.shape[1]=", data_shape[1], " does not match expected row_bytes=", row_bytes_);

    const auto& indices_shape = indices->Shape();
    TensorShapeVector out_dims = indices_shape.AsShapeVector();
    out_dims.push_back(k_);
    Tensor* output = context->Output(0, TensorShape(out_dims));

    const int64_t num_rows = indices_shape.Size();
    if (num_rows == 0) return Status::OK();
    ORT_RETURN_IF_NOT(vocab_size > 0, "GatherPacked requires a nonempty vocabulary for nonempty indices");

    const uint8_t* data_ptr = data->Data<uint8_t>();
    const Tind* idx_ptr = indices->Data<Tind>();
    T2* out_ptr = output->MutableData<T2>();

    auto* tp = context->GetOperatorThreadPool();
    const int type_id = static_cast<int>(ggml_type_);
    const size_t k = SafeInt<size_t>(k_);
    const size_t rb = SafeInt<size_t>(row_bytes_);
    const std::ptrdiff_t rows = SafeInt<std::ptrdiff_t>(num_rows);
    constexpr size_t kChunkSize = 256;
    int64_t block_elems = 0, block_bytes = 0;
    ORT_ENFORCE(gguf_quant::GgmlTypeBlockInfo(type_id, block_elems, block_bytes) &&
                    kChunkSize % block_elems == 0,
                "GatherPacked CPU conversion chunks must contain whole packed blocks");

    concurrency::ThreadPool::TryBatchParallelFor(
        tp, rows,
        [&](std::ptrdiff_t r) {
          const int64_t raw_id = static_cast<int64_t>(idx_ptr[r]);
          const int64_t tok = (raw_id >= 0 && raw_id < vocab_size) ? raw_id : 0;
          const uint8_t* src_row = data_ptr + static_cast<size_t>(tok) * rb;
          T2* dst_row = out_ptr + static_cast<size_t>(r) * k;
          if constexpr (std::is_same_v<T2, float>) {
            ORT_ENFORCE(gguf_quant::DequantizeGgufRowCpu(src_row, type_id, k_, dst_row));
          } else {
            std::array<float, kChunkSize> tmp;
            for (size_t offset = 0; offset < k;) {
              const size_t count = std::min(kChunkSize, k - offset);
              const uint8_t* src_chunk = src_row + (offset / block_elems) * block_bytes;
              ORT_ENFORCE(gguf_quant::DequantizeGgufRowCpu(src_chunk, type_id, count, tmp.data()));
              for (size_t j = 0; j < count; ++j) {
                dst_row[offset + j] = T2(tmp[j]);
              }
              offset += count;
            }
          }
        },
        0);

    return Status::OK();
  }

 private:
  int64_t k_{0};
  int64_t ggml_type_{-1};
  int64_t row_bytes_{0};
};

#define REGISTER_GATHER_PACKED_CPU(T2, Tind)                            \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                    \
      GatherPacked,                                                     \
      kMSDomain, 1,                                                     \
      T2, Tind,                                                         \
      kCpuExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>())      \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<Tind>()), \
      GatherPacked<T2, Tind>);

REGISTER_GATHER_PACKED_CPU(float, int32_t)
REGISTER_GATHER_PACKED_CPU(float, int64_t)
REGISTER_GATHER_PACKED_CPU(MLFloat16, int32_t)
REGISTER_GATHER_PACKED_CPU(MLFloat16, int64_t)
REGISTER_GATHER_PACKED_CPU(BFloat16, int32_t)
REGISTER_GATHER_PACKED_CPU(BFloat16, int64_t)

#undef REGISTER_GATHER_PACKED_CPU

}  // namespace contrib
}  // namespace onnxruntime
