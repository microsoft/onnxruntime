/**
 * Copyright (c) 2016-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
/* Modifications Copyright (c) Microsoft. */

#include "core/providers/cpu/nn/conv.h"

#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
using ConvPadVector = ConvAttributes::ConvPadVector;

template <typename T>
Status Conv<T>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);
  const Tensor* B = context->Input<Tensor>(2);  // optional. nullptr if not provided
  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape()[1];
  const int64_t M = W->Shape()[0];
  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));

  TensorShapeVector kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape));

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

  TensorShapeVector Y_dims({N, M});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
  Tensor* Y = context->Output(0, Y_dims);
  TensorShape output_shape = Y->Shape().Slice(2);

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();
  const int64_t X_offset = C / conv_attrs_.group * input_image_size;
  const int64_t Y_offset = Y->Shape().Size() / Y->Shape()[0] / conv_attrs_.group;
  const int64_t W_offset = W->Shape().Size() / conv_attrs_.group;
  const int64_t kernel_dim = C / conv_attrs_.group * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;

  const size_t kernel_rank = kernel_shape.size();

  BufferUniquePtr col_buffer;

  // Pointwise convolutions can use the original input tensor in place,
  // otherwise a temporary buffer is required for the im2col transform.
  if (kernel_size != 1 || !conv_attrs_.HasStridesOneAndNoPadding()) {
    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

    auto* col_data = alloc->Alloc(sizeof(T) * SafeInt<size_t>(col_buffer_size));
    col_buffer = BufferUniquePtr(col_data, BufferDeleter(std::move(alloc)));
  }

  T* col_buffer_data = static_cast<T*>(col_buffer.get());

  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  const T* Xdata = X->Data<T>();
  T* Ydata = Y->MutableData<T>();

  for (int image_id = 0; image_id < N; ++image_id) {
    for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
      if (col_buffer_data != nullptr) {
        if (kernel_rank == 2) {
          math::Im2col<T, StorageOrder::NCHW>()(
              Xdata + group_id * X_offset,
              C / conv_attrs_.group,
              input_shape[0],
              input_shape[1],
              kernel_shape[0],
              kernel_shape[1],
              dilations[0],
              dilations[1],
              pads[0],
              pads[1],
              pads[2],
              pads[3],
              strides[0],
              strides[1],
              col_buffer_data);
        } else {
          math::Im2col<T, StorageOrder::NCHW>()(
              Xdata + group_id * X_offset,
              input_shape.GetDims().data(),
              output_shape.GetDims().data(),
              kernel_dim,
              kernel_shape.data(),
              strides.data(),
              dilations.data(),
              pads.data(),
              static_cast<int>(kernel_shape.size()),
              col_buffer_data);
        }
      }

      math::Gemm<T>(
          CblasNoTrans,
          CblasNoTrans,
          M / conv_attrs_.group,
          output_image_size,
          kernel_dim,
          1,
          W->Data<T>() + group_id * W_offset,
          col_buffer_data == nullptr ? Xdata + group_id * X_offset : col_buffer_data,
          0,
          Ydata + group_id * Y_offset,
          thread_pool);
    }

    if (B != nullptr) {
      auto Ymatrix = EigenMatrixMap<T>(Ydata, output_image_size, M);
      auto Bvec = ConstEigenVectorMap<T>(B->Data<T>(), M);
      Ymatrix.rowwise() += Bvec.transpose();
    }

    Xdata += X_offset * conv_attrs_.group;
    Ydata += Y_offset * conv_attrs_.group;
  }

  return Status::OK();
}

Status Conv<float>::Compute(OpKernelContext* context) const {
  size_t num_inputs = OpKernel::Node().InputDefs().size();
  const Tensor* X = context->Input<Tensor>(0);
  const Tensor* W = context->Input<Tensor>(1);
  const Tensor* B = num_inputs >= 3 ? context->Input<Tensor>(2) : nullptr;
  const Tensor* Sum = num_inputs >= 4 ? context->Input<Tensor>(3) : nullptr;
  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape()[1];
  const int64_t M = W->Shape()[0];
  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X, W));

  // kernel_shape is an optional attribute and has to be inferred from W if not provided
  TensorShapeVector kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape));

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

  TensorShapeVector Y_dims({N, M});
  TensorShape input_shape = X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  TensorShape output_shape = Y->Shape().Slice(2);

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  auto Xdata = X->DataAsSpan<float>();
  const auto* Bdata = B != nullptr ? B->Data<float>() : nullptr;
  auto Ydata = Y->MutableDataAsSpan<float>();
  // Check for the optional Conv/Sum fusion.
  float Beta = 0.0f;
  if (Sum != nullptr) {
    const auto& sum_shape = Sum->Shape();
    ORT_RETURN_IF_NOT(Y->Shape() == sum_shape, "output and sum shape must match");
    // If the output was not allocated inplace with the sum tensor, then copy here.
    auto sum_data = Sum->DataAsSpan<float>();
    if (Ydata.data() != sum_data.data()) {
      gsl::copy(sum_data, Ydata);
    }
    Beta = 1.0f;
  }
  const size_t kernel_rank = kernel_shape.size();
  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  if (kernel_rank >= 1 && kernel_rank <= 3) {
    MLAS_CONV_PARAMETERS Parameters;
    size_t WorkingBufferSize;
    MlasConvPrepare(&Parameters,
                    kernel_rank,
                    narrow<size_t>(N),
                    narrow<size_t>(conv_attrs_.group),
                    narrow<size_t>(C / conv_attrs_.group),
                    input_shape.GetDims().data(),
                    kernel_shape.data(),
                    dilations.data(),
                    pads.data(),
                    strides.data(),
                    output_shape.GetDims().data(),
                    narrow<size_t>(M / conv_attrs_.group),
                    &activation_,
                    &WorkingBufferSize,
                    Beta,
                    thread_pool);

    auto* working_data = WorkingBufferSize > 0 ? alloc->Alloc(sizeof(float) * SafeInt<size_t>(WorkingBufferSize))
                                               : nullptr;
    BufferUniquePtr working_buffer(working_data, BufferDeleter(std::move(alloc)));

    MlasConv(&Parameters,
             Xdata.data(),
             W->Data<float>(),
             Bdata,
             static_cast<float*>(working_buffer.get()),
             Ydata.data(),
             thread_pool);
  } else {
    const int64_t input_image_size = input_shape.Size();
    const int64_t output_image_size = output_shape.Size();
    const int64_t kernel_size = TensorShape(kernel_shape).Size();
    const SafeInt<int64_t> X_offset = SafeInt<int64_t>(C) / conv_attrs_.group * input_image_size;
    const SafeInt<int64_t> Y_offset = SafeInt<int64_t>(Y->Shape().Size()) / Y->Shape()[0] / conv_attrs_.group;
    const SafeInt<int64_t> W_offset = SafeInt<int64_t>(W->Shape().Size()) / conv_attrs_.group;
    const SafeInt<int64_t> kernel_dim = SafeInt<int64_t>(C) / conv_attrs_.group * kernel_size;
    const int64_t col_buffer_size = kernel_dim * output_image_size;

    auto col_data = IAllocator::MakeUniquePtr<float>(alloc, narrow<size_t>(col_buffer_size));
    auto w_data = W->DataAsSpan<float>();
    for (int image_id = 0; image_id < N; ++image_id) {
      for (int group_id = 0; group_id < conv_attrs_.group; ++group_id) {
        math::Im2col<float, StorageOrder::NCHW>()(
            &Xdata[group_id * X_offset],
            input_shape.GetDims().data(),
            output_shape.GetDims().data(),
            kernel_dim,
            kernel_shape.data(),
            strides.data(),
            dilations.data(),
            pads.data(),
            narrow<int>(kernel_shape.size()),
            col_data.get());

        math::Gemm<float>(
            CblasNoTrans,
            CblasNoTrans,
            narrow<ptrdiff_t>(M / conv_attrs_.group),
            narrow<ptrdiff_t>(output_image_size),
            narrow<ptrdiff_t>(kernel_dim),
            1,
            &w_data[group_id * W_offset],
            col_data.get(),
            Beta,
            &Ydata[group_id * Y_offset],
            thread_pool);
      }

      MlasActivation(&activation_, Ydata.data(), Bdata, narrow<size_t>(M), narrow<size_t>(output_image_size), narrow<size_t>(output_image_size));

      Xdata = Xdata.subspan(X_offset * conv_attrs_.group);
      Ydata = Ydata.subspan(Y_offset * conv_attrs_.group);
    }
  }

  return Status::OK();
}

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Conv,
    1, 10,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);

ONNX_CPU_OPERATOR_KERNEL(
    Conv,
    11,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);

}  // namespace onnxruntime
