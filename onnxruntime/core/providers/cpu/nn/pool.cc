// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel_context_internal.h"
#include "core/providers/cpu/nn/pool.h"
#include <cmath>
using namespace ::onnxruntime::common;

namespace onnxruntime {

template <typename T, typename PoolType>
Status Pool<T, PoolType>::Compute(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  ORT_RETURN_IF_NOT(x_shape.NumDimensions() >= 3, "Input dimension cannot be less than 3.");

  std::vector<int64_t> pads = pads_;
  std::vector<int64_t> kernel_shape = kernel_shape_;

  if (global_pooling_) {
    const auto& input_dims = x_shape.GetDims();
    kernel_shape.assign(input_dims.begin() + 2, input_dims.end());
    pads.assign(kernel_shape.size(), 0);
  }

  std::vector<int64_t> output_dims = PoolBase::SetOutputSize(x_shape, x_shape[1], &pads, dilations_, ceil_mode_);
  Tensor* Y = context->Output(0, TensorShape(output_dims));

  const float* X_data = X->template Data<float>();
  float* Y_data = Y->template MutableData<float>();

  // Get access to the internal threadpool
  auto ctx_internal = static_cast<OpKernelContextInternal*>(context);
  auto thread_pool = ctx_internal->GetOperatorThreadPool();

  // The main loop
  int64_t channels = x_shape[1];
  int64_t height = x_shape[2];
  int64_t width = kernel_shape.size() > 1 ? x_shape[3] : 1;
  int64_t depth = kernel_shape.size() > 2 ? x_shape[4] : 1;
  int64_t pooled_height = output_dims[2];
  int64_t pooled_width = kernel_shape.size() > 1 ? output_dims[3] : 1;
  int64_t pooled_depth = kernel_shape.size() > 2 ? output_dims[4] : 1;

  switch (kernel_shape.size()) {
    case 1: {
      int64_t x_step = height;
      int64_t y_step = pooled_height;
      const int64_t total_channels = x_shape[0] * channels;

      std::function<void(int32_t)> work_object = [&](int32_t c) {
        const float* x_d = X_data + c * x_step;
        float* y_d = Y_data + c * y_step;

        for (int64_t ph = 0; ph < pooled_height; ++ph) {
          int64_t hstart = ph * stride_h() - pads[0];
          int64_t hend = std::min(hstart + kernel_shape[0], height);
          hstart = std::max(hstart, static_cast<int64_t>(0));
          T Yh = PoolType::Initialize();
          for (int64_t h = hstart; h < hend; ++h) {
            PoolType::Process(x_d[h], Yh, pool_context_);
          }
          if (count_include_pad_) {
            PoolType::Finalize(kernel_shape[0], Yh, pool_context_);
          } else {
            PoolType::Finalize(hend - hstart, Yh, pool_context_);
          }
          y_d[ph] = Yh;
        }
      };
      const_cast<concurrency::ThreadPool*>(thread_pool)->ParallelFor((int32_t)total_channels, work_object);

      break;
    }

    case 2: {
      int64_t x_step = height * width;
      int64_t y_step = pooled_height * pooled_width;
      const int64_t total_channels = x_shape[0] * channels;

      std::function<void(int32_t)> work_object = [&](int32_t c) {
        const float* x_d = X_data + c * x_step;
        float* y_d = Y_data + c * y_step;

        for (int64_t ph = 0; ph < pooled_height; ++ph) {
          int64_t hstart = ph * stride_h() - pads[0];
          int64_t hend = std::min(hstart + kernel_shape[0], height);
          hstart = std::max(hstart, static_cast<int64_t>(0));
          for (int64_t pw = 0; pw < pooled_width; ++pw) {
            int64_t wstart = pw * stride_w() - pads[1];
            int64_t wend = std::min(wstart + kernel_shape[1], width);
            wstart = std::max(wstart, static_cast<int64_t>(0));
            const int64_t pool_index = ph * pooled_width + pw;
            T Yh = PoolType::Initialize();
            for (int64_t h = hstart; h < hend; ++h) {
              for (int64_t w = wstart; w < wend; ++w) {
                const int64_t input_index = h * width + w;
                PoolType::Process(x_d[input_index], Yh, pool_context_);
              }
            }
            if (count_include_pad_) {
              PoolType::Finalize(kernel_shape[0] * kernel_shape[1], Yh, pool_context_);
            } else {
              PoolType::Finalize((hend - hstart) * (wend - wstart), Yh, pool_context_);
            }
            y_d[pool_index] = Yh;
          }
        }
      };
      const_cast<concurrency::ThreadPool*>(thread_pool)->ParallelFor((int32_t)total_channels, work_object);

      break;
    }
    case 3: {
      int64_t x_step = height * width * depth;
      int64_t y_step = pooled_height * pooled_width * pooled_depth;
      const int64_t total_channels = x_shape[0] * channels;

      std::function<void(int32_t)> work_object = [&](int32_t c) {
        const float* x_d = X_data + c * x_step;
        float* y_d = Y_data + c * y_step;

        for (int64_t ph = 0; ph < pooled_height; ++ph) {
          int64_t hstart = ph * stride_h() - pads[0];
          int64_t hend = std::min(hstart + kernel_shape[0], height);
          hstart = std::max(hstart, static_cast<int64_t>(0));
          for (int64_t pw = 0; pw < pooled_width; ++pw) {
            int64_t wstart = pw * stride_w() - pads[1];
            int64_t wend = std::min(wstart + kernel_shape[1], width);
            wstart = std::max(wstart, static_cast<int64_t>(0));
            for (int64_t pd = 0; pd < pooled_depth; ++pd) {
              int64_t dstart = pd * stride_d() - pads[2];
              int64_t dend = std::min(dstart + kernel_shape[2], depth);
              dstart = std::max(dstart, static_cast<int64_t>(0));
              const int64_t pool_index =
                  ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
              T Yh = PoolType::Initialize();
              for (int64_t h = hstart; h < hend; ++h) {
                for (int64_t w = wstart; w < wend; ++w) {
                  for (int64_t d = dstart; d < dend; ++d) {
                    const int64_t input_index = h * width * depth + w * depth + d;
                    PoolType::Process(x_d[input_index], Yh, pool_context_);
                  }
                }
              }
              if (count_include_pad_) {
                PoolType::Finalize(kernel_shape[0] * kernel_shape[1] * kernel_shape[2], Yh, pool_context_);
              } else {
                PoolType::Finalize(
                    (hend - hstart) * (wend - wstart) * (dend - dstart), Yh, pool_context_);
              }
              y_d[pool_index] = Yh;
            }
          }
        }
      };
      const_cast<concurrency::ThreadPool*>(thread_pool)->ParallelFor((int32_t)total_channels, work_object);

      break;
    }
    default:
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported pooling size : ");
  }

  return Status::OK();
}

Status PoolBase::Compute(OpKernelContext* context, MLAS_POOLING_KIND kind) const {
  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  size_t input_dims = x_shape.NumDimensions();
  ORT_RETURN_IF_NOT(input_dims >= 3, "Input dimension cannot be less than 3.");

  size_t pooling_dims = input_dims - 2;
  if (pooling_dims > 3) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported pooling size.");
  }
  if (!global_pooling_) {
    ORT_RETURN_IF_NOT(pooling_dims == kernel_shape_.size(), "kernel_shape num_dims is not compatible with X num_dims.");
  }

  std::vector<int64_t> pads = pads_;
  std::vector<int64_t> output_dims = PoolBase::SetOutputSize(x_shape, x_shape[1], &pads, dilations_, ceil_mode_);
  Tensor* Y = context->Output(0, TensorShape(output_dims));

  // Get access to the internal threadpool
  // Temporarily derive concurrency parameters without access to session state
  auto ctx_internal = static_cast<OpKernelContextInternal*>(context);
  auto thread_pool = ctx_internal->GetOperatorThreadPool();

  MlasPool(kind,
           pooling_dims,
           X->Shape().GetDims().data(),
           global_pooling_ ? nullptr : kernel_shape_.data(),
           global_pooling_ ? nullptr : pads.data(),
           global_pooling_ ? nullptr : strides_.data(),
           output_dims.data(),
           X->template Data<float>(),
           Y->template MutableData<float>(),
           const_cast<concurrency::ThreadPool*>(thread_pool));

  return Status::OK();
}

template <>
Status Pool<float, MaxPool<1 /*VERSION*/>>::Compute(OpKernelContext* context) const {
  return PoolBase::Compute(context, MlasMaximumPooling);
}

template <>
Status Pool<float, AveragePool>::Compute(OpKernelContext* context) const {
  return PoolBase::Compute(context, count_include_pad_ ? MlasAveragePoolingIncludePad : MlasAveragePoolingExcludePad);
}

template <>
Status Pool<float, MaxPool<8 /*VERSION*/>>::Compute(OpKernelContext* context) const {
  // Use MLAS pooling if the index output tensor is not used
  // and also if dilation is not required

  bool need_dilation = false;
  for (auto n : dilations_)
    need_dilation |= n > 1;

  if (OpKernel::Node().OutputDefs().size() == 1 && !need_dilation) {
		return PoolBase::Compute(context, MlasMaximumPooling);
  }

  const Tensor* X = context->Input<Tensor>(0);
  const TensorShape& x_shape = X->Shape();

  ORT_RETURN_IF_NOT(x_shape.NumDimensions() >= 3, "Input dimension cannot be less than 3.");

  std::vector<int64_t> pads = pads_;
  std::vector<int64_t> kernel_shape = kernel_shape_;

  std::vector<int64_t> output_dims = PoolBase::SetOutputSize(x_shape, x_shape[1], &pads, dilations_, ceil_mode_);
  Tensor* Y = context->Output(0, TensorShape(output_dims));
  Tensor* I = context->Output(1, TensorShape(output_dims));

  const float* X_data = X->template Data<float>();
  float* Y_data = Y->template MutableData<float>();
  int64_t* I_data = I != nullptr ? I->template MutableData<int64_t>() : nullptr;

  // Get access to the internal threadpool
  auto ctx_internal = static_cast<OpKernelContextInternal*>(context);
  auto thread_pool = ctx_internal->GetOperatorThreadPool();

  // The main loop
  int64_t channels = x_shape[1];
  int64_t height = x_shape[2];
  int64_t width = kernel_shape.size() > 1 ? x_shape[3] : 1;
  int64_t depth = kernel_shape.size() > 2 ? x_shape[4] : 1;
  int64_t pooled_height = output_dims[2];
  int64_t pooled_width = kernel_shape.size() > 1 ? output_dims[3] : 1;
  int64_t pooled_depth = kernel_shape.size() > 2 ? output_dims[4] : 1;

  switch (kernel_shape.size()) {
    case 1: {
      int64_t x_step = height;
      int64_t y_step = pooled_height;
      const int64_t total_channels = x_shape[0] * channels;
      const int64_t dilation_h = dilations_[0];

      std::function<void(int32_t)> work_object = [&](int32_t c) {
        const float* x_d = X_data + c * x_step;
        float* y_d = Y_data + c * y_step;
        int64_t* i_d = I_data ? I_data + c * y_step : nullptr;
        for (int64_t ph = 0; ph < pooled_height; ++ph) {
          int64_t hstart = ph * stride_h() - pads[0];
          int64_t hend = std::min(hstart + kernel_shape[0] * dilation_h - dilation_h + 1, height);
          hstart = std::max(hstart, static_cast<int64_t>(0));
          float Yh = std::numeric_limits<float>::lowest();
          int64_t h_index = -1;
          for (int64_t h = hstart; h < hend; h += dilation_h) {
            if (x_d[h] > Yh) {
              Yh = x_d[h];
              h_index = h;
            }
          }
          y_d[ph] = Yh;
          if (i_d != nullptr) i_d[ph] = c * x_step + h_index;
        }
      };
      const_cast<concurrency::ThreadPool*>(thread_pool)->ParallelFor((int32_t)total_channels, work_object);

      break;
    }

    case 2: {
      int64_t x_step = height * width;
      int64_t y_step = pooled_height * pooled_width;
      const int64_t total_channels = x_shape[0] * channels;
      const int64_t dilation_h = dilations_[0];
      const int64_t dilation_w = dilations_[1];

      std::function<void(int32_t)> work_object = [&](int32_t c) {
        const float* x_d = X_data + c * x_step;
        float* y_d = Y_data + c * y_step;
        int64_t* i_d = I_data ? I_data + c * y_step : nullptr;

        for (int64_t ph = 0; ph < pooled_height; ++ph) {
          int64_t hstart = ph * stride_h() - pads[0];
          int64_t hend = std::min(hstart + kernel_shape[0] * dilation_h - dilation_h + 1, height);
          hstart = std::max(hstart, static_cast<int64_t>(0));
          for (int64_t pw = 0; pw < pooled_width; ++pw) {
            int64_t wstart = pw * stride_w() - pads[1];
            int64_t wend = std::min(wstart + kernel_shape[1] * dilation_w - dilation_w + 1, width);
            wstart = std::max(wstart, static_cast<int64_t>(0));
            const int64_t pool_index = ph * pooled_width + pw;
            float Yh = std::numeric_limits<float>::lowest();
            int64_t h_index = -1;
            int64_t w_index = -1;
            for (int64_t h = hstart; h < hend; h += dilation_h) {
              for (int64_t w = wstart; w < wend; w += dilation_w) {
                const int64_t input_index = h * width + w;
                if (x_d[input_index] > Yh) {
                  Yh = x_d[input_index];
                  h_index = h;
                  w_index = w;
                }
              }
            }
            y_d[pool_index] = Yh;
            if (i_d != nullptr)
              i_d[pool_index] = storage_order_ == 0 ? c * x_step + h_index * width + w_index
                                                    : c * x_step + h_index + w_index * height;
          }
        }
      };
      const_cast<concurrency::ThreadPool*>(thread_pool)->ParallelFor((int32_t)total_channels, work_object);

      break;
    }
    case 3: {
      int64_t x_step = height * width * depth;
      int64_t y_step = pooled_height * pooled_width * pooled_depth;
      const int64_t total_channels = x_shape[0] * channels;
      const int64_t dilation_h = dilations_[0];
      const int64_t dilation_w = dilations_[1];
      const int64_t dilation_d = dilations_[2];

      std::function<void(int32_t)> work_object = [&](int32_t c) {
        const float* x_d = X_data + c * x_step;
        float* y_d = Y_data + c * y_step;
        int64_t* i_d = I_data ? I_data + c * y_step : nullptr;

        for (int64_t ph = 0; ph < pooled_height; ++ph) {
          int64_t hstart = ph * stride_h() - pads[0];
          int64_t hend = std::min(hstart + kernel_shape[0] * dilation_h - dilation_h + 1, height);
          hstart = std::max(hstart, static_cast<int64_t>(0));
          for (int64_t pw = 0; pw < pooled_width; ++pw) {
            int64_t wstart = pw * stride_w() - pads[1];
            int64_t wend = std::min(wstart + kernel_shape[1] * dilation_w - dilation_w + 1, width);
            wstart = std::max(wstart, static_cast<int64_t>(0));
            for (int64_t pd = 0; pd < pooled_depth; ++pd) {
              int64_t dstart = pd * stride_d() - pads[2];
              int64_t dend = std::min(dstart + kernel_shape[2] * dilation_d - dilation_d + 1, depth);
              dstart = std::max(dstart, static_cast<int64_t>(0));
              const int64_t pool_index =
                  ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
              float Yh = std::numeric_limits<float>::lowest();
              int64_t h_index = -1;
              int64_t w_index = -1;
              int64_t d_index = -1;
              for (int64_t h = hstart; h < hend; h += dilation_h) {
                for (int64_t w = wstart; w < wend; w += dilation_w) {
                  for (int64_t d = dstart; d < dend; d += dilation_d) {
                    const int64_t input_index = h * width * depth + w * depth + d;
                    if (x_d[input_index] > Yh) {
                      Yh = x_d[input_index];
                      h_index = h;
                      w_index = w;
                      d_index = d;
                    }
                  }
                }
              }
              y_d[pool_index] = Yh;
              if (i_d != nullptr)
                i_d[pool_index] = storage_order_ == 0 ? c * x_step + h_index * width * depth + w_index * depth + d_index
                                                      : c * x_step + h_index + w_index * height + d_index * height * width;
            }
          }
        }
      };
      const_cast<concurrency::ThreadPool*>(thread_pool)->ParallelFor((int32_t)total_channels, work_object);

      break;
    }
    default:
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported pooling size : ");
  }

  return Status::OK();
}  // namespace onnxruntime

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    AveragePool,
    7, 9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pool<float, AveragePool>);

ONNX_CPU_OPERATOR_KERNEL(
    AveragePool,
    10,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pool<float, AveragePool>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    MaxPool,
    1, 7,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pool<float, MaxPool<1 /*VERSION*/>>);

ONNX_CPU_OPERATOR_VERSIONED_KERNEL(
    MaxPool,
    8, 9,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
    Pool<float, MaxPool<8 /*VERSION*/>>);

ONNX_CPU_OPERATOR_KERNEL(
    MaxPool,
    10,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).TypeConstraint("I", DataTypeImpl::GetTensorType<int64_t>()),
    Pool<float, MaxPool<8 /*VERSION*/>>);

ONNX_CPU_OPERATOR_KERNEL(
    LpPool,
    2,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pool<float, LpPool>);

ONNX_CPU_OPERATOR_KERNEL(
    GlobalLpPool,
    2,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pool<float, LpPool>);

ONNX_CPU_OPERATOR_KERNEL(
    GlobalAveragePool,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pool<float, AveragePool>);

ONNX_CPU_OPERATOR_KERNEL(
    GlobalMaxPool,
    1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Pool<float, MaxPool<1 /*VERSION*/>>);

}  // namespace onnxruntime
