// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "nchwc_ops.h"
#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
using ConvPadVector = ConvAttributes::ConvPadVector;
namespace contrib {

Status ReorderInput::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto X_shape = X->Shape().GetDims();
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

  TensorShapeVector Y_shape(X_rank);
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
    ptrdiff_t work_per_worker = std::max<ptrdiff_t>(worker_goal / narrow<ptrdiff_t>(nchwc_channels), 1);
    worker_count = std::max<ptrdiff_t>(total_work / work_per_worker, 1);
  } else {
    // Each iteration produces one spatial_size chunk of NCHWc blocks.
    total_work = static_cast<ptrdiff_t>(batch_count * (nchwc_channels / nchwc_block_size));
    worker_count = total_work;
  }

  const auto* x_data = X->Data<float>();
  auto* y_data = Y->MutableData<float>();

  auto reorder_worker = [&](ptrdiff_t batch) {
    auto work = concurrency::ThreadPool::PartitionWork(batch, worker_count, total_work);

    if (channels_last_) {
      int64_t work_index = static_cast<int64_t>(work.start);
      int64_t work_remaining = static_cast<int64_t>(work.end) - work.start;

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
      int64_t work_remaining = (static_cast<int64_t>(work.end) - work.start) * nchwc_block_size;

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
  TensorShapeVector Y_shape(X_rank);
  Y_shape[0] = X_shape[0];
  Y_shape[channels_last_ ? X_rank - 1 : 1] = channels_;
  auto* Y_spatial_dims = Y_shape.data() + (channels_last_ ? 1 : 2);
  for (size_t i = 0; i < X_rank - 2; i++) {
    Y_spatial_dims[i] = X_shape[2 + i];
  }
  auto* Y = context->Output(0, Y_shape);

  const auto* x_data = X->Data<float>();
  auto* y_data = Y->MutableData<float>();
  if (channels_last_) {
    MlasReorderOutputNhwc(Y_shape.data(), x_data, y_data);
  } else {
    MlasReorderOutputNchw(Y_shape.data(), x_data, y_data, context->GetOperatorThreadPool());
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

  TensorShapeVector kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W_shape, kernel_shape));
  if (kernel_shape.size() != 2) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Unsupported convolution size.");
  }

  ConvPadVector pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  TensorShapeVector dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  TensorShapeVector strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  TensorShapeVector Y_dims;
  Y_dims.insert(Y_dims.begin(), {X_shape[0], W_shape[0]});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
  auto* Y = context->Output(0, Y_dims);
  auto y_data = Y->MutableDataAsSpan<float>();

  // Check for the optional Conv/Sum fusion.
  if (Sum != nullptr) {
    const auto& sum_shape = Sum->Shape();
    ORT_RETURN_IF_NOT(Y->Shape() == sum_shape, "output and sum shape must match");
    // If the output was not allocated inplace with the sum tensor, then copy here.
    auto sum_data = Sum->DataAsSpan<float>();
    if (y_data.data() != sum_data.data()) {
      gsl::copy(sum_data, y_data);
    }
  }

  MlasNchwcConv(
      X_shape.GetDims().data(),
      kernel_shape.data(),
      dilations.data(),
      pads.data(),
      strides.data(),
      Y_dims.data(),
      static_cast<size_t>(conv_attrs_.group),
      X->Data<float>(),
      W->Data<float>(),
      B != nullptr ? B->Data<float>() : nullptr,
      y_data.data(),
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

  TensorShapeVector pads = pool_attrs_.pads;
  TensorShapeVector output_dims = pool_attrs_.SetOutputSize(X_shape, X_shape[1], &pads);
  auto* Y = context->Output(0, output_dims);

  MlasNchwcPool(
      kind,
      X_shape.GetDims().data(),
      pool_attrs_.global_pooling ? nullptr : pool_attrs_.kernel_shape.data(),
      pool_attrs_.global_pooling ? nullptr : pool_attrs_.dilations.data(),
      pool_attrs_.global_pooling ? nullptr : pads.data(),
      pool_attrs_.global_pooling ? nullptr : pool_attrs_.strides.data(),
      output_dims.data(),
      X->Data<float>(),
      Y->MutableData<float>(),
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

std::vector<float> NchwcUpsample::ComputeInterpolation(int64_t input_length,
                                                       int64_t output_length,
                                                       int64_t scale) const {
  std::vector<float> interpolation;
  interpolation.resize(narrow<size_t>(output_length));

  if (scale == 1) {
    // Identity map for unscaled.
    for (int64_t o = 0; o < output_length; o++) {
      interpolation[narrow<size_t>(o)] = static_cast<float>(o);
    }
  } else if (transformation_mode_ == TransformationMode::ALIGN_CORNERS) {
    for (int64_t o = 0; o < output_length; o++) {
      interpolation[narrow<size_t>(o)] =
          static_cast<float>(o) * static_cast<float>(input_length - 1) / static_cast<float>(output_length - 1);
    }
  } else if (transformation_mode_ == TransformationMode::HALF_PIXEL) {
    for (int64_t o = 0; o < output_length; o++) {
      interpolation[narrow<size_t>(o)] =
          std::max(0.0f, (static_cast<float>(o) + 0.5f) / static_cast<float>(scale) - 0.5f);
    }
  } else {
    // Default to TransformationMode::ASYMMETRIC.
    for (int64_t o = 0; o < output_length; o++) {
      interpolation[narrow<size_t>(o)] = static_cast<float>(o) / static_cast<float>(scale);
    }
  }

  return interpolation;
}

Status NchwcUpsample::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto X_shape = X->Shape().GetDims();
  ORT_ENFORCE(X_shape.size() == 4);
  ORT_ENFORCE((X_shape[1] % MlasNchwcGetBlockSize()) == 0);

  const int64_t batch_count = X_shape[0];
  const int64_t nchwc_channels = X_shape[1];

  const int64_t input_h = X_shape[2];
  const int64_t input_w = X_shape[3];

  const int64_t output_h = input_h * scales_[2];
  const int64_t output_w = input_w * scales_[3];

  auto* Y = context->Output(0, {batch_count, nchwc_channels, output_h, output_w});

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const auto* x_data = X->Data<float>();
  auto* y_data = Y->MutableData<float>();

  if (nearest_mode_) {
    MlasNchwcUpsampleNearest(
        X_shape.data(),
        scales_.data() + 2,
        x_data,
        y_data);
  } else {
    // Compute the interpolation value per output height and width.
    const auto interpolation_h = ComputeInterpolation(input_h, output_h, scales_[2]);
    const auto interpolation_w = ComputeInterpolation(input_w, output_w, scales_[3]);

    const int64_t nchwc_block_size = static_cast<int64_t>(MlasNchwcGetBlockSize());
    const ptrdiff_t total_work = ((SafeInt<ptrdiff_t>(batch_count) * nchwc_channels) / nchwc_block_size) * output_h;
    // Partition the work with the goal of generating the following number of
    // elements, so that operations involving a smaller number of columns will
    // process more rows per worker.
    constexpr ptrdiff_t worker_goal = 16 * 1024;
    ptrdiff_t work_per_worker = std::max<ptrdiff_t>(worker_goal / (SafeInt<ptrdiff_t>(output_w) * nchwc_block_size), 1);
    ptrdiff_t worker_count = std::max<ptrdiff_t>(total_work / work_per_worker, 1);

    auto upsample_worker = [&](ptrdiff_t batch) {
      auto work = concurrency::ThreadPool::PartitionWork(batch, worker_count, total_work);
      int64_t work_index = static_cast<int64_t>(work.start);
      int64_t work_remaining = static_cast<int64_t>(work.end) - work.start;

      while (work_remaining > 0) {
        // Limit the current loop iteration to the same source image.
        const int64_t channel_index = work_index / output_h;
        int64_t row_index = work_index % output_h;
        int64_t rows_this_iteration = std::min(work_remaining, output_h - row_index);

        work_index += rows_this_iteration;
        work_remaining -= rows_this_iteration;

        const auto* x_channel_base = x_data + (channel_index * input_h * input_w * nchwc_block_size);
        auto* y_row = y_data + (((channel_index * output_h) + row_index) * output_w * nchwc_block_size);

        // Loop upsampling each row of the output.
        do {
          MlasNchwcUpsampleLinear(
              static_cast<size_t>(input_h),
              static_cast<size_t>(input_w),
              static_cast<size_t>(output_w),
              interpolation_h[narrow<size_t>(row_index)],
              interpolation_w.data(),
              x_channel_base,
              y_row);
          y_row += output_w * nchwc_block_size;
          row_index++;
        } while (--rows_this_iteration);
      }
    };

    concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

    // Handle the work in a single batch if only a single thread is available.
    if (concurrency::ThreadPool::DegreeOfParallelism(thread_pool) == 1) {
      worker_count = 1;
    }

    concurrency::ThreadPool::TrySimpleParallelFor(thread_pool, worker_count, upsample_worker);
  }

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
