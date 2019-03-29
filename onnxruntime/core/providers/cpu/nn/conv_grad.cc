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
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
namespace contrib {

template <typename T>
Status ConvGrad<T>::Compute(OpKernelContext* context) const {
  const Tensor* dY = context->Input<Tensor>(0);
  const Tensor* X = context->Input<Tensor>(1);
  const Tensor* W = context->Input<Tensor>(2);

  const int64_t N = X->Shape()[0];
  const int64_t C = X->Shape()[1];
  const int64_t M = W->Shape()[0];

  // TODO: validataion might not be needed, since it's already done once in the fw pass
  ORT_RETURN_IF_ERROR(ValidateInputShape(X, W));

  // Copied from conv_impl.h, maybe refactor
  std::vector<int64_t> kernel_shape;
  ORT_RETURN_IF_ERROR(ComputeKernelShape(W->Shape(), kernel_shape));

  bool Is2DKernel = kernel_shape.size() == 2;
  std::vector<int64_t> pads(pads_);
  if (pads.empty()) {
    pads.resize(kernel_shape.size() * 2, 0);
  }
  std::vector<int64_t> dilations(dilations_);
  if (dilations.empty()) {
    dilations.resize(kernel_shape.size(), 1);
  }
  std::vector<int64_t> strides(strides_);
  if (strides.empty()) {
    strides.resize(kernel_shape.size(), 1);
  }

  Tensor* dW = context->Output(1, W->Shape());
  T* dWdata = dW->template MutableData<T>();

  TensorShape input_shape = X->Shape().Slice(2);
  TensorShape output_shape = dY->Shape().Slice(2);

  const int64_t input_image_size = input_shape.Size();
  const int64_t output_image_size = output_shape.Size();
  const int64_t kernel_size = TensorShape(kernel_shape).Size();
  const int64_t X_offset = C / group_ * input_image_size;
  const int64_t Y_offset = dY->Shape().Size() / dY->Shape()[0] / group_;
  const int64_t W_offset = W->Shape().Size() / group_;
  const int64_t kernel_dim = C / group_ * kernel_size;
  const int64_t col_buffer_size = kernel_dim * output_image_size;

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  BufferUniquePtr col_buffer(alloc->Alloc(sizeof(T) * col_buffer_size), BufferDeleter(alloc));
  T* col_buffer_data = static_cast<T*>(col_buffer.get());

  const T* Xdata = X->template Data<T>();
  const T* Wdata = W->template Data<T>();
  const T* dYdata = dY->template Data<T>();

  // Pre-setting the gradients to zero.
  math::Set<T, CPUMathUtil>(dW->Size(), 0, dWdata, &CPUMathUtil::Instance());

  BufferUniquePtr bias_multiplier(alloc->Alloc(sizeof(T) * output_image_size), BufferDeleter(alloc));
  T* bias_multiplier_data = nullptr;
  Tensor* dB = context->Output(2, TensorShape({M}));
  T* dBdata = nullptr;
  if (dB) {
    dBdata = dB->template MutableData<T>();
    math::Set<T, CPUMathUtil>(dB->Size(), static_cast<T>(0), dBdata, &CPUMathUtil::Instance());

    bias_multiplier_data = static_cast<T*>(bias_multiplier.get());
    math::Set<T, CPUMathUtil>(output_image_size,
                              static_cast<T>(1),
                              bias_multiplier_data,
                              &CPUMathUtil::Instance());
  }

  TensorShape image_shape = X->Shape().Slice(1);
  std::vector<int64_t> col_buffer_shape{kernel_dim};
  col_buffer_shape.insert(col_buffer_shape.end(), output_shape.GetDims().begin(),
                          output_shape.GetDims().end());

  for (int image_id = 0; image_id < N; ++image_id) {
    for (int group_id = 0; group_id < group_; ++group_id) {
      if (Is2DKernel) {
        math::Im2col<T, CPUMathUtil, StorageOrder::NCHW>(
            Xdata + group_id * X_offset,
            C / group_,
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
            col_buffer_data,
            &CPUMathUtil::Instance());
      } else {
        math::Im2colNd<T, CPUMathUtil, StorageOrder::NCHW>()(
            Xdata + group_id * X_offset,
            image_shape.GetDims().data(),
            col_buffer_shape.data(),
            C * input_image_size,
            col_buffer_size,
            kernel_shape.data(),
            strides.data(),
            dilations.data(),
            pads.data(),
            static_cast<int>(kernel_shape.size()),
            col_buffer_data,
            &CPUMathUtil::Instance());
      }
      // Gradient with respect to W, filter.
      math::Gemm<T, CPUMathUtil>(
          CblasNoTrans,
          CblasTrans,
          M / group_,
          kernel_dim,
          output_image_size,
          1,
          dYdata + group_id * Y_offset,
          col_buffer_data,
          1,
          dWdata + group_id * W_offset,
          &CPUMathUtil::Instance());
    }
    if (dB) {
      // Gradient with respect to bias can be computed independent from group.
      math::Gemv<T, CPUMathUtil>(
          CblasNoTrans,
          static_cast<int>(M),
          static_cast<int>(output_image_size),
          1,
          dYdata,
          bias_multiplier_data,
          1,
          dBdata,
          &CPUMathUtil::Instance());
    }
    Xdata += X_offset * group_;
    dYdata += Y_offset * group_;
  }

  Tensor* dX = context->Output(0, X->Shape());
  if (dX) {
    T* dXdata = dX->template MutableData<T>();
    for (int image_id = 0; image_id < N; ++image_id) {
      for (int group_id = 0; group_id < group_; ++group_id) {
        // Compute gradient into col_buffer.
        math::Gemm<T, CPUMathUtil>(
            CblasTrans,
            CblasNoTrans,
            kernel_dim,
            output_image_size,
            M / group_,
            1,
            Wdata + group_id * W_offset,
            dYdata,
            0,
            col_buffer_data,
            &CPUMathUtil::Instance());

        if (Is2DKernel) {
          math::Col2im<T, CPUMathUtil, StorageOrder::NCHW>(
              col_buffer_data,
              C / group_,
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
              dXdata,
              &CPUMathUtil::Instance());
        } else {
          math::Col2imNd<T, CPUMathUtil, StorageOrder::NCHW>(
              col_buffer_data,
              image_shape.GetDims().data(),
              col_buffer_shape.data(),
              C * input_image_size,
              col_buffer_size,
              kernel_shape.data(),
              strides.data(),
              dilations.data(),
              pads.data(),
              static_cast<int>(kernel_shape.size()),
              dXdata,
              &CPUMathUtil::Instance());
        }
        dXdata += X_offset;
        dYdata += Y_offset;
      }
    }
  }
  return Status::OK();
}

ONNX_CPU_OPERATOR_KERNEL(
    ConvGrad,
    9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ConvGrad<float>);

}  // namespace contrib
}  // namespace onnxruntime
