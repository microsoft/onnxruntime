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

template <typename T8Bits>
class NhwcMaxPool : public OpKernel {
 public:
  explicit NhwcMaxPool(const OpKernelInfo& info) : OpKernel(info),
                                                   pool_attrs_(info, "MaxPool", info.node().SinceVersion()) {
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  PoolAttributes pool_attrs_;
};

template <typename T8Bits>
Status NhwcMaxPool<T8Bits>::Compute(OpKernelContext* context) const {
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
  auto* col_data = alloc->Alloc(SafeInt<size_t>(sizeof(const T8Bits*)) * kernel_size * col_buffer_batch_count);
  BufferUniquePtr col_buffer(col_data, BufferDeleter(std::move(alloc)));
  std::vector<T8Bits> padding_data(static_cast<size_t>(C), std::numeric_limits<T8Bits>::lowest());

  const auto* Xdata = X->Data<T8Bits>();
  auto* Ydata = Y->MutableData<T8Bits>();

  for (int64_t image_id = 0; image_id < N; ++image_id) {
    for (int64_t output_start = 0; output_start < output_image_size;) {
      int64_t output_count = std::min(output_image_size - output_start, output_batch_count);
      math::Im2col<T8Bits, StorageOrder::NHWC>()(
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
          static_cast<T8Bits const**>(col_buffer.get()),
          padding_data.data());
      MlasMaximumPool(
          static_cast<T8Bits const**>(col_buffer.get()),
          Ydata,
          static_cast<size_t>(C),
          static_cast<size_t>(output_count),
          static_cast<size_t>(kernel_size));

      Ydata += output_count * C;
      output_start += output_count;
    }

    Xdata += input_image_size * C;
  }

  return Status::OK();
}

#define REGISTER_NHWCMAXPOOL_TYPED_KERNEL(T)                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      NhwcMaxPool,                                                \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCpuExecutionProvider,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      NhwcMaxPool<T>);

REGISTER_NHWCMAXPOOL_TYPED_KERNEL(int8_t);
REGISTER_NHWCMAXPOOL_TYPED_KERNEL(uint8_t);

}  // namespace contrib
}  // namespace onnxruntime
