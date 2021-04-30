// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "nchwc_ops.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace contrib {

Status ReorderInput::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto& X_shape = X->Shape().GetDims();
  const auto X_rank = X_shape.size();
  ORT_ENFORCE(X_rank == 4);

  const int64_t batch_count = X_shape[0];
  const int64_t channels = X_shape[channels_last_ ? X_rank - 1 : 1];
  const auto* X_spatial_dims = X_shape.data() + (channels_last_ ? 1 : 2);

  // The current implementation of MlasReorderInputNchw does not work for channels that
  // are not a multiple of 4.
  ORT_ENFORCE((channels % 4) == 0);

  const int64_t nchwc_block_size = static_cast<int64_t>(MlasNchwcGetBlockSize());
  const int64_t nchwc_channels = (channels + nchwc_block_size - 1) & ~(nchwc_block_size - 1);

  std::vector<int64_t> Y_shape(X_rank);
  Y_shape[0] = batch_count;
  Y_shape[1] = nchwc_channels;
  int64_t spatial_size = 1;
  for (size_t i = 0; i < X_rank - 2; i++) {
    const int64_t spatial_dim = X_spatial_dims[i];
    spatial_size *= spatial_dim;
    Y_shape[2 + i] = spatial_dim;
  }

  auto* Y = context->Output(0, Y_shape);

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  // Compute the total amount of work depending on NCHW or NHWC format and estimate
  // a number of workers to use.
  ptrdiff_t total_work;
  ptrdiff_t worker_count;

  if (channels_last_) {
    total_work = static_cast<ptrdiff_t>(batch_count * spatial_size);
    // Partition the work with the goal of reordering the following number of
    // elements, so that operations involving a smaller number of channels will
    // process more rows per worker.
    constexpr ptrdiff_t worker_goal = 48 * 1024;
    ptrdiff_t work_per_worker = worker_goal / nchwc_channels;
    if (work_per_worker == 0) {
      work_per_worker = 1;
    }
    worker_count = total_work / work_per_worker;
    if (worker_count == 0) {
      worker_count = 1;
    }
  } else {
    // Each iteration produces one spatial_size chunk of NCHWc blocks.
    total_work = static_cast<ptrdiff_t>(batch_count * (nchwc_channels / nchwc_block_size));
    worker_count = total_work;
  }

  const auto* x_data = X->template Data<float>();
  auto* y_data = Y->template MutableData<float>();

  auto reorder_worker = [&](ptrdiff_t batch) {
    auto work = concurrency::ThreadPool::PartitionWork(batch, worker_count, total_work);

    if (channels_last_) {
      int64_t work_index = static_cast<int64_t>(work.start);
      int64_t work_remaining = static_cast<int64_t>(work.end - work.start);

      while (work_remaining > 0) {
        const int64_t batch_index = work_index / spatial_size;
        const int64_t spatial_index = work_index % spatial_size;
        const int64_t rows_this_iteration = std::min(work_remaining, spatial_size - spatial_index);

        MlasReorderInputNhwc(
            x_data + ((batch_index * spatial_size) + spatial_index) * channels,
            y_data + (batch_index * spatial_size * nchwc_channels) + (spatial_index * nchwc_block_size),
            static_cast<size_t>(channels),
            static_cast<size_t>(rows_this_iteration),
            static_cast<size_t>(spatial_size));

        work_index += rows_this_iteration;
        work_remaining -= rows_this_iteration;
      }
    } else {
      int64_t work_index = static_cast<int64_t>(work.start) * nchwc_block_size;
      int64_t work_remaining = static_cast<int64_t>(work.end - work.start) * nchwc_block_size;

      while (work_remaining > 0) {
        const int64_t batch_index = work_index / nchwc_channels;
        const int64_t channel_index = work_index % nchwc_channels;
        const int64_t channels_this_iteration = std::min(work_remaining, channels - channel_index);

        MlasReorderInputNchw(
            x_data + ((batch_index * channels) + channel_index) * spatial_size,
            y_data + ((batch_index * nchwc_channels) + channel_index) * spatial_size,
            static_cast<size_t>(channels_this_iteration),
            static_cast<size_t>(spatial_size));

        const int64_t nchwc_channels_this_iteration = std::min(work_remaining, nchwc_channels - channel_index);
        work_index += nchwc_channels_this_iteration;
        work_remaining -= nchwc_channels_this_iteration;
      }
    }
  };

  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  // Handle the work in a single batch if only a single thread is available.
  if (concurrency::ThreadPool::DegreeOfParallelism(thread_pool) == 1) {
    worker_count = 1;
  }

  concurrency::ThreadPool::TrySimpleParallelFor(thread_pool, worker_count, reorder_worker);

  return Status::OK();
}

Status ReorderOutput::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto& X_shape = X->Shape().GetDims();
  const auto X_rank = X_shape.size();
  ORT_ENFORCE(X_rank == 4);
  ORT_ENFORCE(channels_ <= X_shape[1]);

  // Build the output shape in NCHW or NHWC order.
  std::vector<int64_t> Y_shape(X_rank);
  Y_shape[0] = X_shape[0];
  Y_shape[channels_last_ ? X_rank - 1 : 1] = channels_;
  auto* Y_spatial_dims = Y_shape.data() + (channels_last_ ? 1 : 2);
  for (size_t i = 0; i < X_rank - 2; i++) {
    Y_spatial_dims[i] = X_shape[2 + i];
  }
  auto* Y = context->Output(0, Y_shape);

  const auto* x_data = X->template Data<float>();
  auto* y_data = Y->template MutableData<float>();
  if (channels_last_) {
    MlasReorderOutputNhwc(Y_shape.data(), x_data, y_data);
  } else {
    MlasReorderOutputNchw(Y_shape.data(), x_data, y_data);
  }

  return Status::OK();
}

Status NchwcConv::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);
  const auto* B = context->Input<Tensor>(2);
  const auto* Sum = context->Input<Tensor>(3);

  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));

  const auto& X_shape = X->Shape();
  const auto& W_shape = W->Shape();
  ORT_ENFORCE(X_shape.NumDimensions() == 4);

  const size_t nchwc_block_size = MlasNchwcGetBlockSize();
  ORT_ENFORCE((static_cast<size_t>(X_shape[1]) < nchwc_block_size) || ((X_shape[1] % nchwc_block_size) == 0));

  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W_shape, kernel_shape));
  if (kernel_shape.size() != 2) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Unsupported convolution size.");
  }

  std::vector<int64_t> pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  std::vector<int64_t> dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  std::vector<int64_t> Y_dims;
  Y_dims.insert(Y_dims.begin(), {X_shape[0], W_shape[0]});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
  auto* Y = context->Output(0, Y_dims);
  auto* y_data = Y->template MutableData<float>();

  // Check for the optional Conv/Sum fusion.
  if (Sum != nullptr) {
    const auto& sum_shape = Sum->Shape();
    ORT_RETURN_IF_NOT(Y->Shape() == sum_shape, "output and sum shape must match");
    // If the output was not allocated inplace with the sum tensor, then copy here.
    const auto* sum_data = Sum->template Data<float>();
    if (y_data != sum_data) {
      memcpy(y_data, sum_data, sum_shape.Size() * sizeof(float));
    }
  }

  MlasNchwcConv(X_shape.GetDims().data(),
                kernel_shape.data(),
                dilations.data(),
                pads.data(),
                strides.data(),
                Y_dims.data(),
                static_cast<size_t>(conv_attrs_.group),
                X->template Data<float>(),
                W->template Data<float>(),
                B != nullptr ? B->template Data<float>() : nullptr,
                y_data,
                &activation_,
                Sum == nullptr,
                context->GetOperatorThreadPool());

  return Status::OK();
}

Status NchwcPoolBase::NchwcPool(OpKernelContext* context, MLAS_POOLING_KIND kind) const {
  const auto* X = context->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  ORT_ENFORCE(X_shape.NumDimensions() == 4);
  ORT_ENFORCE((X_shape[1] % MlasNchwcGetBlockSize()) == 0);

  std::vector<int64_t> pads = pool_attrs_.pads;
  std::vector<int64_t> output_dims = pool_attrs_.SetOutputSize(X_shape, X_shape[1], &pads);
  auto* Y = context->Output(0, output_dims);

  MlasNchwcPool(kind,
                X_shape.GetDims().data(),
                pool_attrs_.global_pooling ? nullptr : pool_attrs_.kernel_shape.data(),
                pool_attrs_.global_pooling ? nullptr : pool_attrs_.dilations.data(),
                pool_attrs_.global_pooling ? nullptr : pads.data(),
                pool_attrs_.global_pooling ? nullptr : pool_attrs_.strides.data(),
                output_dims.data(),
                X->template Data<float>(),
                Y->template MutableData<float>(),
                context->GetOperatorThreadPool());

  return Status::OK();
}

Status NchwcMaxPool::Compute(OpKernelContext* context) const {
  return NchwcPoolBase::NchwcPool(context, MlasMaximumPooling);
}

Status NchwcAveragePool::Compute(OpKernelContext* context) const {
  return NchwcPoolBase::NchwcPool(context, pool_attrs_.count_include_pad ? MlasAveragePoolingIncludePad
                                                                         : MlasAveragePoolingExcludePad);
}

Status NchwcUpsample::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  ORT_ENFORCE(X_shape.NumDimensions() == 4);
  ORT_ENFORCE((X_shape[1] % MlasNchwcGetBlockSize()) == 0);

  TensorShape Y_shape{X_shape[0], X_shape[1], X_shape[2] * scales_[2], X_shape[3] * scales_[3]};
  auto* Y = context->Output(0, Y_shape);

  MlasNchwcUpsample(X_shape.GetDims().data(),
                    scales_.data() + 2,
                    X->template Data<float>(),
                    Y->template MutableData<float>());

  return Status::OK();
}

#define ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(name, ver, type, builder, ...) \
  ONNX_OPERATOR_TYPED_KERNEL_EX(name, kMSNchwcDomain, ver, type, kCpuExecutionProvider, builder, __VA_ARGS__)

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    ReorderInput,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReorderInput);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    ReorderOutput,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ReorderOutput);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    Conv,
    1,
    float,
    KernelDefBuilder()
        .MayInplace(3, 0)
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcConv);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    MaxPool,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcMaxPool);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    GlobalMaxPool,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcMaxPool);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    AveragePool,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcAveragePool);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    GlobalAveragePool,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcAveragePool);

ONNX_CPU_OPERATOR_TYPED_NCHWC_KERNEL(
    Upsample,
    1,
    float,
    KernelDefBuilder()
        .TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    NchwcUpsample);

}  // namespace contrib
}  // namespace onnxruntime
