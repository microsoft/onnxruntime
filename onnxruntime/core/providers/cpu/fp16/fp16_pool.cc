// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/mlas/inc/mlas.h"

#ifdef MLAS_F16VEC_INTRINSICS_SUPPORTED

#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/providers/cpu/nn/pool_attributes.h"
#include "core/platform/threadpool.h"
#include "core/common/safeint.h"

namespace onnxruntime {

/**
 * @brief Pooling operator for type FP16,
 * Only max pool and average pool supported.
 *
 * Single threadded operation for now.
 *
 * TODO!! implemente thread partition similar with
 * fp16 conv operator
 */
class PoolFp16 : public OpKernel {
 public:
  explicit PoolFp16(const OpKernelInfo& info)
      : OpKernel(info),
        pool_attrs_(info, info.GetKernelDef().OpName(), info.node().SinceVersion()),
        is_max_pool_(info.GetKernelDef().OpName() == "MaxPool"),
        channels_last_(info.GetKernelDef().Domain() == kMSInternalNHWCDomain) {}

  Status Compute(OpKernelContext* context) const override;

 protected:
  PoolAttributes pool_attrs_;
  bool is_max_pool_;  // either max pool or average pool
  bool channels_last_;
};

Status PoolFp16::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const TensorShape& input_shape = X->Shape();

  const size_t input_rank = input_shape.NumDimensions();
  ORT_RETURN_IF_NOT(input_rank >= 3, "Input dimension cannot be less than 3.");

  const int64_t N = input_shape[0];
  const int64_t C = channels_last_ ? input_shape[input_rank - 1] : input_shape[1];

  ORT_ENFORCE(input_shape.Size() > 0 || N == 0, "Invalid input shape. Only N can be zero. Got:", input_shape);

  const size_t spatial_dims = input_rank - 2;
  const size_t spatial_dim_start = channels_last_ ? 1 : 2;

  // Compute the output size and effective padding for this pooling operation.
  TensorShapeVector output_dims({N});
  if (!channels_last_) {
    output_dims.push_back(C);
  }
  TensorShapeVector pads = pool_attrs_.pads;
  TensorShapeVector kernel_shape = pool_attrs_.kernel_shape;
  TensorShapeVector strides = pool_attrs_.strides;
  TensorShapeVector dilations = pool_attrs_.dilations;
  if (pool_attrs_.global_pooling) {
    const auto& input_dims = input_shape.GetDims();
    if (channels_last_) {
      kernel_shape.assign(input_dims.begin() + 1, input_dims.end() - 1);
    } else {
      kernel_shape.assign(input_dims.begin() + 2, input_dims.end());
    }
    pads.resize(kernel_shape.size() * 2, 0);
    strides.resize(kernel_shape.size(), 1);
    dilations.resize(kernel_shape.size(), 1);
  }
  if (kernel_shape.size() != spatial_dims) {
    std::ostringstream ss;
    ss << "Invalid kernel shape. Input shape ";
    ss << (channels_last_ ? "(NHWC):[" : "(NCHW):[");
    for (int64_t i = 0; i < input_shape.Size(); i++) {
      ss << input_shape[i] << ", ";
    }
    ss << "] Kernel shape:[";
    for (size_t i = 0; i < kernel_shape.size(); i++) {
      ss << kernel_shape[i] << ", ";
    }
    ss << "]";

    ORT_THROW(ss.str());
  }

  int64_t kernel_size = 1;
  int64_t input_image_size = 1;
  int64_t output_image_size = 1;
  for (size_t dim = 0; dim < spatial_dims; ++dim) {
    int64_t kernel = kernel_shape[dim];
    int64_t input_dim = input_shape[dim + spatial_dim_start];

    kernel_size *= kernel;
    input_image_size *= input_dim;

    int64_t output_dim = 0;
    pool_attrs_.ComputeSizePadDilations(input_dim,
                                        strides[dim],
                                        kernel,
                                        &pads.at(dim),
                                        &pads.at(spatial_dims + dim),
                                        dilations[dim],
                                        &output_dim);
    output_dims.push_back(output_dim);

    output_image_size *= output_dim;
  }
  if (channels_last_) {
    output_dims.push_back(C);
  }

  const bool need_padding = !is_max_pool_ && pool_attrs_.count_include_pad;
  std::vector<MLFloat16> padding_data;
  if (need_padding) {
    padding_data.resize(static_cast<size_t>(C), MLFloat16());
  }

  const auto* Xdata = X->Data<MLFloat16>();
  auto* Y = context->Output(0, output_dims);
  auto* Ydata = Y->MutableData<MLFloat16>();

  // Allocate temporary buffers for transposing to channels last format.
  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  BufferUniquePtr transpose_input_buffer;
  BufferUniquePtr transpose_output_buffer;
  if (!channels_last_) {
    auto* transpose_input = alloc->Alloc(SafeInt<size_t>(sizeof(MLFloat16)) * C * input_image_size + MLAS_SYMM_QGEMM_BUF_OVERRUN);
    transpose_input_buffer = BufferUniquePtr(transpose_input, BufferDeleter(alloc));
    auto* transpose_output = alloc->Alloc(SafeInt<size_t>(sizeof(MLFloat16)) * C * output_image_size);
    transpose_output_buffer = BufferUniquePtr(transpose_output, BufferDeleter(alloc));
  }

  // Allocate indirection buffer pointers and prepare a padding vector for the
  // im2col transform.
  auto* col_data = alloc->Alloc(SafeInt<size_t>(sizeof(const MLFloat16*)) * kernel_size * output_image_size);
  BufferUniquePtr col_buffer(col_data, BufferDeleter(std::move(alloc)));

  const int64_t output_stride = std::max((int64_t)2, (int64_t)8192 / (kernel_size * C));
  const int64_t task_count = (output_image_size + output_stride - 1) / output_stride;
  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  for (int64_t image_id = 0; image_id < N; ++image_id) {
    const auto* input_data = Xdata;
    auto* output_data = Ydata;

    if (!channels_last_) {
      // Transpose the input from channels first (CHW) to channels last (HWC).
      MlasTranspose(
          Xdata,
          static_cast<MLFloat16*>(transpose_input_buffer.get()),
          static_cast<size_t>(C),
          static_cast<size_t>(input_image_size));
      input_data = static_cast<MLFloat16*>(transpose_input_buffer.get());
      output_data = static_cast<MLFloat16*>(transpose_output_buffer.get());
    }

    auto worker = [&](ptrdiff_t batch) {
      int64_t output_start = (int64_t)batch * (int64_t)output_stride;
      int64_t output_count = std::min((int64_t)output_stride, output_image_size - output_start);
      auto* outputptr = output_data + output_stride * C * batch;
      auto indirection_buffer = static_cast<MLFloat16 const**>(col_buffer.get()) + output_start * kernel_size;

      math::Im2col<MLFloat16, StorageOrder::NHWC>()(
          input_data,
          C,
          input_shape.GetDims().data() + spatial_dim_start,
          output_dims.data() + spatial_dim_start,
          kernel_shape.data(),
          strides.data(),
          dilations.data(),
          pads.data(),
          static_cast<ptrdiff_t>(spatial_dims),
          output_start,
          output_count,
          indirection_buffer,
          need_padding ? padding_data.data() : nullptr);

      if (is_max_pool_) {
        MlasNhwcMaxPool(
            indirection_buffer,
            outputptr,
            static_cast<size_t>(C),
            static_cast<size_t>(output_count),
            static_cast<size_t>(kernel_size));
      } else {
        MlasNhwcAvgPool(
            indirection_buffer,
            outputptr,
            static_cast<size_t>(C),
            static_cast<size_t>(output_count),
            static_cast<size_t>(kernel_size));
      }
    };
    concurrency::ThreadPool::TrySimpleParallelFor(thread_pool, onnxruntime::narrow<ptrdiff_t>(task_count), worker);

    if (!channels_last_) {
      // Transpose the output from channels last (NHWC) to channels first (NCHW).
      MlasTranspose(
          output_data,
          Ydata,
          static_cast<size_t>(output_image_size),
          static_cast<size_t>(C));
    }
    Xdata += input_image_size * C;
    Ydata += output_image_size * C;
  }

  return Status::OK();
}

//
// Operator definitions
//
ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MaxPool, 8, 11,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    MaxPool,
    12, 21,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    MaxPool,
    22,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    AveragePool, 11, 18,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    AveragePool,
    19, 21,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    AveragePool,
    22,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    GlobalAveragePool,
    1, 21,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_CPU_OPERATOR_TYPED_KERNEL(
    GlobalAveragePool,
    22,
    MLFloat16,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

#ifndef DISABLE_CONTRIB_OPS
namespace contrib {

ONNX_OPERATOR_TYPED_KERNEL_EX(
    MaxPool,
    kMSInternalNHWCDomain,
    12,
    MLFloat16,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    AveragePool,
    kMSInternalNHWCDomain,
    11,
    MLFloat16,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    GlobalAveragePool,
    kMSInternalNHWCDomain,
    1,
    MLFloat16,
    kCpuExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<MLFloat16>()),
    PoolFp16);

}  // namespace contrib
#endif  // DISABLE_CONTRIB_OPS

}  // namespace onnxruntime
#endif  // MLAS_F16VEC_INTRINSICS_SUPPORTED
