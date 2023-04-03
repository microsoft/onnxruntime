// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/pool_attributes.h"
#include "core/common/safeint.h"
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED

/**
 * @brief Pooling operator for type FP16, layout NHWC.
 * Only max pool and average pool supported.
 * 
 * Single threadded operation for now.
 * 
 * TODO!! implemente thread partition similar with
 * fp16 conv operator
*/
class NhwcPoolFp16 : public OpKernel {
 public:
  explicit NhwcPoolFp16(const OpKernelInfo& info, const std::string& pooltype, bool is_max_pool)
      : OpKernel(info), pool_attrs_(info, pooltype, info.node().SinceVersion()), is_max_pool_(is_max_pool) {}

  Status Compute(OpKernelContext* context) const override;

 protected:
  PoolAttributes pool_attrs_;
  bool is_max_pool_;  // either max pool or average pool
};

Status NhwcPoolFp16::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const TensorShape& input_shape = X->Shape();

  const size_t input_rank = input_shape.NumDimensions();
  ORT_RETURN_IF_NOT(input_rank >= 3, "Input dimension cannot be less than 3.");

  const int64_t N = input_shape[0];
  const int64_t C = input_shape[input_rank - 1];

  ORT_ENFORCE(input_shape.Size() > 0 || N == 0, "Invalid input shape. Only N can be zero. Got:", input_shape);

  const size_t spatial_dims = input_rank - 2;

  // Compute the output size and effective padding for this pooling operation.
  TensorShapeVector output_dims({N});
  TensorShapeVector pads = pool_attrs_.pads;
  int64_t kernel_size = 1;
  int64_t input_image_size = 1;
  int64_t output_image_size = 1;
  for (size_t dim = 0; dim < spatial_dims; ++dim) {
    int64_t kernel = pool_attrs_.kernel_shape[dim];
    int64_t input_dim = input_shape[dim + 1];

    kernel_size *= kernel;
    input_image_size *= input_dim;

    int64_t output_dim = 0;
    pool_attrs_.ComputeSizePadDilations(input_dim,
                                        pool_attrs_.strides[dim],
                                        kernel,
                                        &pads.at(dim),
                                        &pads.at(spatial_dims + dim),
                                        pool_attrs_.dilations[dim],
                                        &output_dim);
    output_dims.push_back(output_dim);

    output_image_size *= output_dim;
  }
  output_dims.push_back(C);

  Tensor* Y = context->Output(0, output_dims);

  constexpr int64_t output_batch_count = 512;

  // Allocate indirection buffer pointers and prepare a padding vector for the
  // im2col transform.
  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  int64_t col_buffer_batch_count = std::min(output_image_size, output_batch_count);
  auto* col_data = alloc->Alloc(SafeInt<size_t>(sizeof(const MLFloat16*)) * kernel_size * col_buffer_batch_count);
  BufferUniquePtr col_buffer(col_data, BufferDeleter(std::move(alloc)));

  const bool need_padding = !is_max_pool_ && pool_attrs_.count_include_pad;
  std::vector<MLFloat16> padding_data;
  if (need_padding) {
    padding_data.resize(static_cast<size_t>(C), MLFloat16());
  }

  const auto* Xdata = X->Data<MLFloat16>();
  auto* Ydata = Y->MutableData<MLFloat16>();

  for (int64_t image_id = 0; image_id < N; ++image_id) {
    for (int64_t output_start = 0; output_start < output_image_size;) {
      int64_t output_count = std::min(output_image_size - output_start, output_batch_count);
      math::Im2col<MLFloat16, StorageOrder::NHWC>()(
          Xdata,
          C,
          input_shape.GetDims().data() + 1,
          output_dims.data() + 1,
          pool_attrs_.kernel_shape.data(),
          pool_attrs_.strides.data(),
          pool_attrs_.dilations.data(),
          pads.data(),
          static_cast<ptrdiff_t>(spatial_dims),
          output_start,
          output_count,
          static_cast<MLFloat16 const**>(col_buffer.get()),
          need_padding ? padding_data.data() : nullptr);
      if (is_max_pool_) {
        MlasNhwcMaxPool(
            static_cast<MLFloat16 const**>(col_buffer.get()),
            Ydata,
            static_cast<size_t>(C),
            static_cast<size_t>(output_count),
            static_cast<size_t>(kernel_size));
      } else {
        MlasNhwcAvgPool(
            static_cast<MLFloat16 const**>(col_buffer.get()),
            Ydata,
            static_cast<size_t>(C),
            static_cast<size_t>(output_count),
            static_cast<size_t>(kernel_size));
      }

      Ydata += output_count * C;
      output_start += output_count;
    }

    Xdata += input_image_size * C;
  }

  return Status::OK();
}

class NhwcMaxPoolFp16 : public NhwcPoolFp16 {
 public:
  explicit NhwcMaxPoolFp16(const OpKernelInfo& info)
      : NhwcPoolFp16(info, "MaxPool", true /*maxpool*/) {}
};

class NhwcAvgPoolFp16 : public NhwcPoolFp16 {
 public:
  explicit NhwcAvgPoolFp16(const OpKernelInfo& info)
      : NhwcPoolFp16(info, "AveragePool", false /*not maxpool*/) {}
};

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MaxPool,
    kMSInternalNHWCDomain,
    11,
    MLFloat16,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    NhwcMaxPoolFp16);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    AveragePool,
    kMSInternalNHWCDomain,
    11,
    MLFloat16,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    NhwcAvgPoolFp16);


#endif

}  // namespace contrib
}  // namespace onnxruntime
