// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/upsample.h"
#include <math.h>  //for fabs

using namespace ::onnxruntime::common;
using namespace std;
namespace onnxruntime {

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Upsample,
    7, 9,
    float,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Upsample<float>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Upsample,
    7, 9,
    int32_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<int32_t>()),
    Upsample<int32_t>);

ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(
    Upsample,
    7, 9,
    uint8_t,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()),
    Upsample<uint8_t>);

template <typename T>
void UpsampleNearest2x(
    int64_t batch_size,
    int64_t num_channels,
    int64_t input_height,
    int64_t input_width,
    const T* input,
    T* output) {
  const int64_t output_height = input_height * 2;
  const int64_t output_width = input_width * 2;
  for (int64_t n = 0; n < batch_size; ++n) {
    for (int64_t c = 0; c < num_channels; ++c) {
      for (int64_t y = 0; y < output_height; ++y) {
        const int64_t in_y = y / 2;
        for (int64_t x = 0; x < input_width; ++x) {
          const T v = input[in_y * input_width + x];
          const int64_t oidx = output_width * y + x * 2;
          output[oidx + 0] = v;
          output[oidx + 1] = v;
        }
      }
      input += input_height * input_width;
      output += output_height * output_width;
    }
  }
}

template <typename T>
Status UpsampleNearest(const T* input,
                       T* output,
                       const TensorShape& input_shape,
                       const TensorShape& output_shape,
                       const vector<float>& scales) {
  if (!input || !output)
    return Status(ONNXRUNTIME, FAIL, "Upsample: input/output value is nullptr");
  if (input_shape.NumDimensions() != output_shape.NumDimensions())
    return Status(ONNXRUNTIME, FAIL, "Upsample: input/output value's dimension mismatch");
  auto n_dim = input_shape.NumDimensions();
  if (scales.size() == 4 && scales[0] == 1 && scales[1] == 1 && scales[2] == 2 && scales[3] == 2) {
    UpsampleNearest2x<T>(input_shape[0], input_shape[1], input_shape[2], input_shape[3], input, output);
  } else {
    for (size_t i = 0, size = output_shape.Size(); i < size; i++) {
      size_t old_idx = 0;
      size_t cur_idx = i;

      int64_t base = 1;
      for (int64_t j = static_cast<int64_t>(n_dim - 1); j >= 0; j--) {
        auto tmp = cur_idx % output_shape[j];

        if (scales[j] < 1) {  //downsample
          old_idx += (std::min(static_cast<int64_t>(std::ceil(tmp / scales[j])), input_shape[j] - 1)) * base;
        } else {  //upsample
          old_idx += (std::min(static_cast<int64_t>(tmp / scales[j]), input_shape[j] - 1)) * base;
        }
        base *= input_shape[j];
        cur_idx /= output_shape[j];
      }

      output[i] = input[old_idx];
    }
  }
  return Status::OK();
}

//This is a generic upsample in linear mode for N-D tensor.
//But what's the correct behavior for linear mode is not clear right now.
//this function is not enabled yet.
template <typename T>
Status upsampleLiner(const T* input,
                     T* output,
                     const TensorShape& input_shape,
                     const TensorShape& output_shape,
                     const vector<float>& scales) {
  if (!input || !output)
    return Status(ONNXRUNTIME, FAIL, "Upsample: input/output value is nullptr");
  if (input_shape.NumDimensions() != output_shape.NumDimensions())
    return Status(ONNXRUNTIME, FAIL, "Upsample: input/output value's dimension mismatch");
  auto n_dim = input_shape.NumDimensions();
  for (size_t i = 0, size = output_shape.Size(); i < size; i++) {
    std::vector<int64_t> val1, val2;
    std::vector<float> d1, d2;
    size_t cur_idx = i;
    //val1, vla2, d1, d2 are in reverse order
    for (int64_t j = static_cast<int64_t>(n_dim - 1); j >= 0; j--) {
      T v = std::min((cur_idx % output_shape[j]) / scales[j], static_cast<float>(input_shape[j] - 1));
      auto v1 = std::min(static_cast<int64_t>(v), input_shape[j] - 1);
      auto v2 = std::min(v1 + 1, input_shape[j] - 1);
      if (v1 == v2) {
        d1.push_back(0.5f);
        d2.push_back(0.5f);
      } else {
        d1.push_back(std::abs(v - v1));
        d2.push_back(std::abs(v - v2));
      }
      val1.push_back(v1);
      val2.push_back(v2);
      cur_idx /= output_shape[j];
    }

    output[i] = 0;
    int64_t step = static_cast<int64_t>(1 << n_dim) - 1;
    while (step >= 0) {
      auto cur = step;
      float w = 1.0f;
      size_t old_idx = 0;
      size_t base = 1;
      for (int64_t j = static_cast<int64_t>(n_dim - 1); j >= 0; j--) {
        int64_t reverse_idx = static_cast<int64_t>(n_dim - 1) - j;
        w *= (cur % 2) ? d1[reverse_idx] : d2[reverse_idx];
        old_idx += ((cur % 2) ? val2[reverse_idx] : val1[reverse_idx]) * base;
        base *= input_shape[j];
        cur >>= 1;
      }
      output[i] += input[old_idx] * w;
      step--;
    }
  }
  return Status::OK();
}

template <typename T>
void upsampleBilinear(
    int64_t batch_size,
    int64_t num_channels,
    int64_t input_height,
    int64_t input_width,
    float height_scale,
    float width_scale,
    const T* Xdata,
    T* Ydata) {
  int64_t output_width = static_cast<int64_t>(input_width * width_scale);
  int64_t output_height = static_cast<int64_t>(input_height * height_scale);

  for (int64_t n = 0; n < batch_size; ++n) {
    for (int64_t c = 0; c < num_channels; ++c) {
      for (int64_t y = 0; y < output_height; ++y) {
        float in_y = std::min(y / height_scale, static_cast<float>(input_height - 1));
        const int64_t in_y1 = std::min(static_cast<int64_t>(in_y), input_height - 1);
        const int64_t in_y2 = std::min(in_y1 + 1, input_height - 1);
        float dy1 = fabs(in_y - in_y1);
        float dy2 = fabs(in_y - in_y2);
        if (in_y1 == in_y2) {
          dy1 = 0.5f;
          dy2 = 0.5f;
        }

        const int64_t input_width_mul_y1 = input_width * in_y1;
        const int64_t input_width_mul_y2 = input_width * in_y2;

        for (int64_t x = 0; x < output_width; ++x) {
          float in_x = std::min(x / width_scale, static_cast<float>(input_width - 1));
          const int64_t in_x1 = std::min(static_cast<int64_t>(in_x), input_width - 1);
          const int64_t in_x2 = std::min(in_x1 + 1, input_width - 1);

          float dx1 = std::abs(in_x - in_x1);
          float dx2 = std::abs(in_x - in_x2);
          if (in_x1 == in_x2) {
            dx1 = 0.5f;
            dx2 = 0.5f;
          }

          T X11 = Xdata[input_width_mul_y1 + in_x1];
          T X21 = Xdata[input_width_mul_y1 + in_x2];
          T X12 = Xdata[input_width_mul_y2 + in_x1];
          T X22 = Xdata[input_width_mul_y2 + in_x2];

          Ydata[output_width * y + x] = static_cast<T>(dx2 * dy2 * X11 +
                                                       dx1 * dy2 * X21 +
                                                       dx2 * dy1 * X12 +
                                                       dx1 * dy1 * X22);
        }
      }
      Xdata += input_height * input_width;
      Ydata += output_width * output_height;
    }
  }
}

template <typename T>
Status Upsample<T>::BaseCompute(OpKernelContext* context, const std::vector<float>& scales) const {
  const Tensor* X = context->Input<Tensor>(0);
  ORT_ENFORCE(X != nullptr);

  const std::vector<int64_t>& dims = X->Shape().GetDims();
  if (dims.size() != scales.size()) {
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Upsample: input tensor's dimension does not match the scales.");
  }

  std::vector<int64_t> Y_dims;
  for (std::size_t i = 0; i < dims.size(); i++) {
    Y_dims.push_back(static_cast<int64_t>(scales[i] * dims[i]));
  }
  Tensor* Y = context->Output(0, Y_dims);

  switch (mode_) {
    case UpsampleMode::NN:
      return UpsampleNearest<T>(X->template Data<T>(), Y->template MutableData<T>(), X->Shape(), Y->Shape(), scales);
    case UpsampleMode::LINEAR: {
      //What's the correct behavior of linear mode is not clear right now,
      //Only support bilinear with 4D tensor to keep consistent with previous behavior
      if (dims.size() != 4)
        return Status(ONNXRUNTIME, FAIL, "Upsample: linear mode upsample only support 4-D tensor with NCHW layout");

      const int64_t batch_size = dims[0], num_channels = dims[1];
      const int64_t input_height = dims[2], input_width = dims[3];

      upsampleBilinear(batch_size, num_channels, input_height, input_width,
                       scales[2], scales[3], X->template Data<T>(), Y->template MutableData<T>());
      return Status::OK();
    }
    default:
      return Status(ONNXRUNTIME, FAIL, "Upsample: unexpected mode");
  }
}

template <typename T>
Status Upsample<T>::Compute(OpKernelContext* context) const {
  if (OpKernel::Node().InputDefs().size() == 1 || scales_cached_) {
    return BaseCompute(context, scales_);
  }

  const Tensor* scales = context->Input<Tensor>(1);
  ORT_ENFORCE(scales != nullptr);
  int64_t scales_size = scales->Shape().Size();
  std::vector<float> scales_arrary(scales_size);
  ParseScalesData(scales, scales_arrary);
  return BaseCompute(context, scales_arrary);
}

}  // namespace onnxruntime
