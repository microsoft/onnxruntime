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

#include <vector>

#include "core/providers/cpu/nn/conv.h"

#include "core/common/narrow.h"
#include "core/common/safeint.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {
using ConvPadVector = ConvAttributes::ConvPadVector;

namespace {

template <typename T>
void ConvertNHWCToNCHW(const T* src, T* dst,
                       int64_t n, int64_t c, int64_t h, int64_t w,
                       concurrency::ThreadPool* thread_pool) {
  const size_t n_count = narrow<size_t>(n);
  const size_t c_count = narrow<size_t>(c);
  const size_t hw = narrow<size_t>(SafeInt<int64_t>(h) * w);
  for (size_t n_idx = 0; n_idx < n_count; ++n_idx) {
    const size_t n_src_offset = SafeInt<size_t>(SafeInt<size_t>(n_idx) * hw) * c_count;
    const size_t n_dst_offset = SafeInt<size_t>(SafeInt<size_t>(n_idx) * c_count) * hw;
    MlasTranspose(src + n_src_offset, dst + n_dst_offset, hw, c_count, thread_pool);
  }
}

template <typename T>
void ConvertNCHWToNHWC(const T* src, T* dst,
                       int64_t n, int64_t c, int64_t h, int64_t w,
                       concurrency::ThreadPool* thread_pool) {
  const size_t n_count = narrow<size_t>(n);
  const size_t c_count = narrow<size_t>(c);
  const size_t hw = narrow<size_t>(SafeInt<int64_t>(h) * w);
  for (size_t n_idx = 0; n_idx < n_count; ++n_idx) {
    const size_t n_src_offset = SafeInt<size_t>(SafeInt<size_t>(n_idx) * c_count) * hw;
    const size_t n_dst_offset = SafeInt<size_t>(SafeInt<size_t>(n_idx) * hw) * c_count;
    MlasTranspose(src + n_src_offset, dst + n_dst_offset, c_count, hw, thread_pool);
  }
}

}  // namespace

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
  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X->Shape(), W->Shape(), channels_last_));
  const int64_t N = X->Shape()[0];
  // If channels_last_ we should get the back dim for channels instead of [1]
  const int64_t C = channels_last_ ? X->Shape().GetDims().back() : X->Shape()[1];
  const int64_t M = W->Shape()[0];

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
  TensorShape input_shape = channels_last_ ? X->Shape().Slice(1, 3) : X->Shape().Slice(2);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(input_shape, kernel_shape, strides, dilations, pads, Y_dims));
  if (channels_last_) {
    Y_dims = {Y_dims[0], Y_dims[2], Y_dims[3], Y_dims[1]};
  }
  Tensor* Y = context->Output(0, TensorShape(Y_dims));
  TensorShape output_shape = channels_last_ ? TensorShape(Y_dims).Slice(1, 3) : Y->Shape().Slice(2);

  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  auto Xdata = X->DataAsSpan<float>();
  const auto* Bdata = B != nullptr ? B->Data<float>() : nullptr;
  auto Ydata = Y->MutableDataAsSpan<float>();
  const size_t kernel_rank = kernel_shape.size();
  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();

  if (channels_last_) {
    ORT_RETURN_IF_NOT(kernel_rank == 2, "Conv with channels_last layout currently supports 2D kernels.");
  }

  const bool wants_channels_last = channels_last_;
  const bool sum_present = Sum != nullptr;
  std::array<size_t, 2> input_shape_size_t{};
  std::array<size_t, 2> kernel_shape_size_t{};
  std::array<size_t, 2> dilations_size_t{};
  std::array<size_t, 4> pads_size_t{};
  std::array<size_t, 2> strides_size_t{};
  if (wants_channels_last) {
    ORT_RETURN_IF_NOT(input_shape.NumDimensions() == 2, "Nhwc Conv fast-path expects 2D input shape.");
    for (size_t i = 0; i < 2; ++i) {
      input_shape_size_t[i] = narrow<size_t>(input_shape[i]);
      kernel_shape_size_t[i] = narrow<size_t>(kernel_shape[i]);
      dilations_size_t[i] = narrow<size_t>(dilations[i]);
      strides_size_t[i] = narrow<size_t>(strides[i]);
      pads_size_t[i] = narrow<size_t>(pads[i]);
      pads_size_t[i + 2] = narrow<size_t>(pads[i + 2]);
    }
  }
  const bool nhwc_fastpath =
      wants_channels_last && !sum_present &&
      MlasConvSupportsSymmetricChannelsLast2DFloatKernel(
          kernel_rank,
          narrow<size_t>(N),
          narrow<size_t>(conv_attrs_.group),
          input_shape_size_t.data(),
          kernel_shape_size_t.data(),
          dilations_size_t.data(),
          pads_size_t.data(),
          strides_size_t.data(),
          narrow<size_t>(M / conv_attrs_.group),
          /*Beta*/ 0.0f);
  const bool manual_sum = wants_channels_last && !nhwc_fastpath && sum_present;
  MLAS_ACTIVATION pre_sum_activation = activation_;
  if (manual_sum) {
    pre_sum_activation.ActivationKind = MlasIdentityActivation;
  }

  std::vector<float> sum_manual_buffer;
  const float* sum_manual_data = nullptr;

  float Beta = 0.0f;
  if (sum_present) {
    const auto& sum_shape = Sum->Shape();
    ORT_RETURN_IF_NOT(Y->Shape() == sum_shape, "output and sum shape must match");
    if (manual_sum) {
      auto sum_span = Sum->DataAsSpan<float>();
      sum_manual_buffer.assign(sum_span.begin(), sum_span.end());
      sum_manual_data = sum_manual_buffer.data();
    } else {
      auto sum_span = Sum->DataAsSpan<float>();
      if (Ydata.data() != sum_span.data()) {
        gsl::copy(sum_span, Ydata);
      }
      Beta = 1.0f;
    }
  }

  if (kernel_rank >= 1 && kernel_rank <= 3) {
    MLAS_CONV_PARAMETERS Parameters;
    Parameters.BackendKernelSelectorConfig = &mlas_backend_kernel_selector_config_;

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
                    manual_sum ? &pre_sum_activation : &activation_,
                    &WorkingBufferSize,
                    nhwc_fastpath,
                    nhwc_fastpath ? 0.0f : Beta,
                    thread_pool);

    float* working_data = nullptr;
    BufferUniquePtr working_buffer;
    if (WorkingBufferSize > 0) {
      working_data = static_cast<float*>(alloc->Alloc(sizeof(float) * SafeInt<size_t>(WorkingBufferSize)));
      working_buffer = BufferUniquePtr(working_data, BufferDeleter(alloc));
    }

    float* output_compute = Ydata.data();
    BufferUniquePtr output_temp;
    if (wants_channels_last && !nhwc_fastpath) {
      const SafeInt<size_t> output_compute_size =
          SafeInt<size_t>(Y->Shape()[0]) * SafeInt<size_t>(M) *
          SafeInt<size_t>(output_shape[0]) * SafeInt<size_t>(output_shape[1]);
      float* temp_output = static_cast<float*>(alloc->Alloc(sizeof(float) * output_compute_size));
      output_temp = BufferUniquePtr(temp_output, BufferDeleter(alloc));
      output_compute = temp_output;
    }

    const float* input_compute = Xdata.data();
    BufferUniquePtr input_temp;
    if (wants_channels_last && !nhwc_fastpath) {
      ORT_RETURN_IF_NOT(X->Shape().NumDimensions() == 4, "Nhwc fallback expects 4D input.");
      const auto& x_dims = X->Shape().GetDims();
      const int64_t input_n = x_dims[0];
      const int64_t input_h = x_dims[1];
      const int64_t input_w = x_dims[2];
      const int64_t input_c = x_dims[3];
      const SafeInt<size_t> input_elements = SafeInt<size_t>(X->Shape().Size());
      float* temp_input = static_cast<float*>(alloc->Alloc(sizeof(float) * input_elements));
      input_temp = BufferUniquePtr(temp_input, BufferDeleter(alloc));
      ConvertNHWCToNCHW(X->Data<float>(), temp_input,
                        input_n, input_c, input_h, input_w, thread_pool);
      input_compute = temp_input;
    }

    MlasConv(&Parameters,
             input_compute,
             W->Data<float>(),
             Bdata,
             working_data,
             output_compute,
             thread_pool);

    if (wants_channels_last && !nhwc_fastpath) {
      const auto& y_dims = Y->Shape().GetDims();
      ORT_RETURN_IF_NOT(y_dims.size() == 4, "Nhwc fallback expects 4D output.");
      if (manual_sum) {
        const SafeInt<size_t> output_elements = SafeInt<size_t>(Y->Shape().Size());
        float* sum_nchw = static_cast<float*>(alloc->Alloc(sizeof(float) * output_elements));
        BufferUniquePtr sum_nchw_buffer(sum_nchw, BufferDeleter(alloc));
        ConvertNHWCToNCHW(sum_manual_data,
                          sum_nchw,
                          y_dims[0], y_dims[3], y_dims[1], y_dims[2], thread_pool);

        auto output_span = gsl::make_span(output_compute, static_cast<size_t>(output_elements));
        auto sum_span = gsl::make_span(sum_nchw, static_cast<size_t>(output_elements));
        for (size_t i = 0; i < output_span.size(); ++i) {
          output_span[i] += sum_span[i];
        }

        const auto activation_rows = narrow<size_t>(SafeInt<int64_t>(y_dims[0]) * y_dims[3]);
        const auto activation_cols = narrow<size_t>(output_shape.Size());
        MlasActivation(&activation_, output_compute, nullptr, activation_rows,
                       activation_cols, activation_cols);
      }

      ConvertNCHWToNHWC(output_compute,
                        Ydata.data(),
                        y_dims[0], y_dims[3], y_dims[1], y_dims[2], thread_pool);
    }
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
            thread_pool,
            &mlas_backend_kernel_selector_config_);
      }

      MlasActivation(&activation_, Ydata.data(), Bdata, narrow<size_t>(M),
                     narrow<size_t>(output_image_size), narrow<size_t>(output_image_size));

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

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    Conv,
    11,
    21,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);

// Opset 22 starts to support bfloat16
ONNX_CPU_OPERATOR_KERNEL(
    Conv,
    22,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Conv<float>);

}  // namespace onnxruntime
