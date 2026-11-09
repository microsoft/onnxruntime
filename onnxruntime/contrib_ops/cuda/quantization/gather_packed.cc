// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/gather_packed.h"

#include <type_traits>

#include "contrib_ops/cpu/quantization/gguf_quant_tables.h"
#include "core/framework/allocator.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {
bool PackedTensorBytesFit(int64_t rows, int64_t columns, size_t element_bytes) {
  size_t row_bytes = 0;
  size_t bytes = 0;
  return rows >= 0 && columns >= 0 &&
         IAllocator::CalcMemSizeForArray(static_cast<size_t>(columns), element_bytes, &row_bytes) &&
         IAllocator::CalcMemSizeForArray(static_cast<size_t>(rows), row_bytes, &bytes) &&
         bytes <= static_cast<size_t>(std::numeric_limits<ptrdiff_t>::max());
}
}  // namespace

#define REGISTER_GATHER_PACKED_CUDA(T2, Tind)                           \
  ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(                                    \
      GatherPacked,                                                     \
      kMSDomain, 1,                                                     \
      T2, Tind,                                                         \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>())      \
          .TypeConstraint("Tind", DataTypeImpl::GetTensorType<Tind>()), \
      GatherPacked<T2, Tind>);

REGISTER_GATHER_PACKED_CUDA(float, int32_t)
REGISTER_GATHER_PACKED_CUDA(float, int64_t)
REGISTER_GATHER_PACKED_CUDA(MLFloat16, int32_t)
REGISTER_GATHER_PACKED_CUDA(MLFloat16, int64_t)
REGISTER_GATHER_PACKED_CUDA(BFloat16, int32_t)
REGISTER_GATHER_PACKED_CUDA(BFloat16, int64_t)

#undef REGISTER_GATHER_PACKED_CUDA

template <typename T2, typename Tind>
GatherPacked<T2, Tind>::GatherPacked(const OpKernelInfo& info) : CudaKernel(info) {
  ORT_ENFORCE(info.GetAttr<int64_t>("K", &k_).IsOK() && k_ > 0, "GatherPacked requires K > 0");
  ORT_ENFORCE(k_ <= kGatherPackedMaxK, "GatherPacked requires K <= ", kGatherPackedMaxK);
  ORT_ENFORCE(info.GetAttr<int64_t>("packed_type", &ggml_type_).IsOK(), "GatherPacked requires packed_type");
  ORT_ENFORCE(gguf_quant::IsGgmlTypeSupported(ggml_type_), "GatherPacked: unsupported packed_type=", ggml_type_);
  row_bytes_ = gguf_quant::GgmlRowBytes(static_cast<int>(ggml_type_), k_);
  ORT_ENFORCE(row_bytes_ > 0, "GatherPacked: unsupported packed_type=", ggml_type_, " or invalid K=", k_);
}

template <typename T2, typename Tind>
Status GatherPacked<T2, Tind>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* data = context->Input<Tensor>(0);
  const Tensor* indices = context->Input<Tensor>(1);
  const auto& data_shape = data->Shape();
  ORT_RETURN_IF_NOT(data_shape.NumDimensions() == 2, "GatherPacked data must be 2D [vocab_size, row_bytes]");
  const int64_t vocab_size = data_shape[0];
  ORT_RETURN_IF_NOT(data_shape[1] == row_bytes_,
                    "GatherPacked data.shape[1]=", data_shape[1], " does not match expected row_bytes=", row_bytes_);

  const auto& indices_shape = indices->Shape();
  const int64_t num_rows = indices_shape.Size();
  ORT_RETURN_IF_NOT(num_rows >= 0 && num_rows <= kGatherPackedMaxRows &&
                        num_rows <= GetDeviceProp().maxGridSize[0],
                    "GatherPacked index count exceeds the CUDA grid limit");
  ORT_RETURN_IF_NOT(PackedTensorBytesFit(num_rows, k_, sizeof(T2)) &&
                        PackedTensorBytesFit(vocab_size, row_bytes_, 1),
                    "GatherPacked tensor byte size exceeds the supported address range");
  ORT_RETURN_IF_NOT(num_rows == 0 || vocab_size > 0,
                    "GatherPacked requires a nonempty vocabulary for nonempty indices");

  TensorShapeVector out_dims = indices_shape.AsShapeVector();
  out_dims.push_back(k_);
  Tensor* output = context->Output(0, TensorShape(out_dims));

  if (num_rows == 0) return Status::OK();

  constexpr int out_dtype =
      std::is_same_v<T2, BFloat16> ? 16 : (std::is_same_v<T2, MLFloat16> ? 10 : 1);
  constexpr bool is_int64 = std::is_same_v<Tind, int64_t>;

  return LaunchGatherPackedCuda(
      Stream(context),
      data->Data<uint8_t>(),
      indices->DataRaw(),
      is_int64,
      output->MutableDataRaw(),
      out_dtype,
      num_rows,
      vocab_size,
      k_,
      row_bytes_,
      static_cast<int>(ggml_type_));
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
