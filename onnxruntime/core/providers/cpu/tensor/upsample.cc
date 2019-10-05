// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/tensor/upsample.h"
#include <cmath>
#include <sstream>

using namespace onnxruntime::common;
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
                       const vector<float>& scales,
                       bool is_resize) {
  if (!input || !output)
    return Status(ONNXRUNTIME, FAIL, is_resize ? "Resize: input/output value is nullptr" : 
                                                 "Upsample: input/output value is nullptr");
  if (input_shape.NumDimensions() != output_shape.NumDimensions())
    return Status(ONNXRUNTIME, FAIL, is_resize ? "Resize: input/output value's dimension mismatch" : 
                                                 "Upsample: input/output value's dimension mismatch");
  if (input_shape.NumDimensions() == 0) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  is_resize ? "Resize: input shape needs to be at least a single dimension" : 
                              "Upsample: input shape needs to be at least a single dimension.");
  }

  int64_t n_dim = static_cast<int64_t>(input_shape.NumDimensions());

  std::vector<int64_t> input_dim_counters(n_dim);
  std::vector<int64_t> input_dim_factor(n_dim);
  input_dim_factor[n_dim - 1] = 1;  // initialize dimension factor
  for (int64_t dim_idx = n_dim - 2; dim_idx >= 0; dim_idx--) {
    input_dim_factor[dim_idx] = input_dim_factor[dim_idx + 1] * input_shape[dim_idx + 1];
  }

  int64_t output_idx = 0;
  int64_t input_idx = 0;

#define OneDemensionProcessor(dim_inx)                                                                                                                  \
  int64_t input_dim##dim_inx##_inx =                                                                                                                    \
      static_cast<int64_t>(scales[dim_inx] < 1 ? std::ceil(output_dim##dim_inx##_inx / scales[dim_inx]) : output_dim##dim_inx##_inx / scales[dim_inx]); \
  if (input_dim##dim_inx##_inx > input_shape[dim_inx] - 1) input_dim##dim_inx##_inx = input_shape[dim_inx] - 1;                                         \
  if (input_dim##dim_inx##_inx != input_dim_counters[dim_inx]) {                                                                                        \
    input_idx += (input_dim##dim_inx##_inx - input_dim_counters[dim_inx]) * input_dim_factor[dim_inx];                                                  \
    input_dim_counters[dim_inx] = input_dim##dim_inx##_inx;                                                                                             \
  }

  if (n_dim == 1) {
    for (int64_t output_dim0_inx = 0; output_dim0_inx < output_shape[0]; output_dim0_inx++) {
      OneDemensionProcessor(0);
      output[output_idx++] = input[input_idx];
    }
    return Status::OK();
  }

  if (n_dim == 2) {
    for (int64_t output_dim0_inx = 0; output_dim0_inx < output_shape[0]; output_dim0_inx++) {
      OneDemensionProcessor(0);
      for (int64_t output_dim1_inx = 0; output_dim1_inx < output_shape[1]; output_dim1_inx++) {
        OneDemensionProcessor(1);
        output[output_idx++] = input[input_idx];
      }
    }
    return Status::OK();
  }

  if (n_dim == 3) {
    for (int64_t output_dim0_inx = 0; output_dim0_inx < output_shape[0]; output_dim0_inx++) {
      OneDemensionProcessor(0);
      for (int64_t output_dim1_inx = 0; output_dim1_inx < output_shape[1]; output_dim1_inx++) {
        OneDemensionProcessor(1);
        for (int64_t output_dim2_inx = 0; output_dim2_inx < output_shape[2]; output_dim2_inx++) {
          OneDemensionProcessor(2);
          output[output_idx++] = input[input_idx];
        }
      }
    }
    return Status::OK();
  }

  if (n_dim == 4) {
    if (scales[0] == 1 && scales[1] == 1 && scales[2] == 2 && scales[3] == 2) {
      UpsampleNearest2x<T>(input_shape[0], input_shape[1], input_shape[2], input_shape[3], input, output);
      return Status::OK();
    }
    for (int64_t output_dim0_inx = 0; output_dim0_inx < output_shape[0]; output_dim0_inx++) {
      OneDemensionProcessor(0);
      for (int64_t output_dim1_inx = 0; output_dim1_inx < output_shape[1]; output_dim1_inx++) {
        OneDemensionProcessor(1);
        for (int64_t output_dim2_inx = 0; output_dim2_inx < output_shape[2]; output_dim2_inx++) {
          OneDemensionProcessor(2);
          for (int64_t output_dim3_inx = 0; output_dim3_inx < output_shape[3]; output_dim3_inx++) {
            OneDemensionProcessor(3);
            output[output_idx++] = input[input_idx];
          }
        }
      }
    }
    return Status::OK();
  }

#undef OneDemensionProcessor

  std::vector<int64_t> output_dim_counter(n_dim);
  output_dim_counter[n_dim - 1] = -1;  // initialize dimension counter

  for (; output_idx < output_shape.Size(); output_idx++) {
    for (int64_t dim_idx = n_dim - 1; dim_idx >= 0; dim_idx--) {
      if (++output_dim_counter[dim_idx] < output_shape[dim_idx]) {
        int64_t current_input_dim_counter = 0;
        if (scales[dim_idx] < 1)  //downsample
        {
          current_input_dim_counter = static_cast<int64_t>(std::ceil(output_dim_counter[dim_idx] / scales[dim_idx]));
        } else  //upsample
        {
          current_input_dim_counter = static_cast<int64_t>(output_dim_counter[dim_idx] / scales[dim_idx]);
        }

        if (current_input_dim_counter >= input_shape[dim_idx] - 1)
          current_input_dim_counter = input_shape[dim_idx] - 1;

        if (current_input_dim_counter != input_dim_counters[dim_idx]) {
          input_idx += (current_input_dim_counter - input_dim_counters[dim_idx]) * input_dim_factor[dim_idx];
          input_dim_counters[dim_idx] = current_input_dim_counter;
        }
        break;
      } else {
        output_dim_counter[dim_idx] = 0;
        input_idx += (0 - input_dim_counters[dim_idx]) * input_dim_factor[dim_idx];
        input_dim_counters[dim_idx] = 0;
      }
    }

    output[output_idx] = input[input_idx];
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
                     const vector<float>& scales,
                     bool is_resize) {
  if (!input || !output)
    return Status(ONNXRUNTIME, FAIL, is_resize ? "Resize: input / output value is nullptr" : 
                                                 "Upsample: input / output value is nullptr");
  if (input_shape.NumDimensions() != output_shape.NumDimensions())
    return Status(ONNXRUNTIME, FAIL, is_resize ? "Resize: input/output value's dimension mismatch" : 
                                                 "Upsample: input/output value's dimension mismatch");
  auto n_dim = input_shape.NumDimensions();
  for (size_t i = 0, size = output_shape.Size(); i < size; i++) {
    std::vector<int64_t> val1;
    std::vector<int64_t> val2;
    std::vector<float> d1;
    std::vector<float> d2;
    size_t cur_idx = i;
    //val1, vla2, d1, d2 are in reverse order
    for (auto j = static_cast<int64_t>(n_dim - 1); j >= 0; j--) {
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
      for (auto j = static_cast<int64_t>(n_dim - 1); j >= 0; j--) {
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

// The following method supports a 4-D input in 'Linear mode' 
// that amounts to 'Bilinear' Upsampling/Resizing in the sense that it assumes
// the scale values for the outermost 2 dimensions are 1.
// This is the common use-case where the 4-D input (batched multi-channel images) 
// is usually of shape [N, C, H, W] and the scales are [1.0, 1.0, height_scale, width_scale]
template <typename T>
void upsampleBilinear(
    int64_t batch_size,
    int64_t num_channels,
    int64_t input_height,
    int64_t input_width,
    float height_scale,
    float width_scale,
    const T* Xdata,
    T* Ydata,
    AllocatorPtr& alloc) {
  auto output_width = static_cast<int64_t>(input_width * width_scale);
  auto output_height = static_cast<int64_t>(input_height * height_scale);

  size_t idx_buffer_size = 2 * sizeof(int64_t) * (output_height + output_width);
  size_t scale_buffer_size = 2 * sizeof(float_t) * (output_height + output_width);
  auto inx_scale_data_buffer = alloc->Alloc(idx_buffer_size + scale_buffer_size);
  BufferUniquePtr idx_scale_data_buffer_holder(inx_scale_data_buffer, BufferDeleter(alloc));
  auto* idx_data = static_cast<int64_t*>(idx_scale_data_buffer_holder.get());
  int64_t* input_width_mul_y1 = idx_data;
  int64_t* input_width_mul_y2 = idx_data + output_height;
  int64_t* in_x1 = idx_data + 2 * output_height;
  int64_t* in_x2 = idx_data + 2 * output_height + output_width;

  auto* scale_data = reinterpret_cast<float*>(in_x2 + output_width);
  float* dy1 = scale_data;
  float* dy2 = scale_data + output_height;
  float* dx1 = scale_data + 2 * output_height;
  float* dx2 = scale_data + 2 * output_height + output_width;

  for (int64_t y = 0; y < output_height; ++y) {
    float in_y = std::min(y / height_scale, static_cast<float>(input_height - 1));
    const int64_t in_y1 = std::min(static_cast<int64_t>(in_y), input_height - 1);
    const int64_t in_y2 = std::min(in_y1 + 1, input_height - 1);
    dy1[y] = std::fabs(in_y - in_y1);
    dy2[y] = std::fabs(in_y - in_y2);
    if (in_y1 == in_y2) {
      dy1[y] = 0.5f;
      dy2[y] = 0.5f;
    }

    input_width_mul_y1[y] = input_width * in_y1;
    input_width_mul_y2[y] = input_width * in_y2;
  }

  for (int64_t x = 0; x < output_width; ++x) {
    float in_x = std::min(x / width_scale, static_cast<float>(input_width - 1));
    in_x1[x] = std::min(static_cast<int64_t>(in_x), input_width - 1);
    in_x2[x] = std::min(in_x1[x] + 1, input_width - 1);

    dx1[x] = std::abs(in_x - in_x1[x]);
    dx2[x] = std::abs(in_x - in_x2[x]);
    if (in_x1[x] == in_x2[x]) {
      dx1[x] = 0.5f;
      dx2[x] = 0.5f;
    }
  }

  for (int64_t n = 0; n < batch_size; ++n) {
    for (int64_t c = 0; c < num_channels; ++c) {
      for (int64_t y = 0; y < output_height; ++y) {
        for (int64_t x = 0; x < output_width; ++x) {
          T X11 = Xdata[input_width_mul_y1[y] + in_x1[x]];
          T X21 = Xdata[input_width_mul_y1[y] + in_x2[x]];
          T X12 = Xdata[input_width_mul_y2[y] + in_x1[x]];
          T X22 = Xdata[input_width_mul_y2[y] + in_x2[x]];

          Ydata[output_width * y + x] = static_cast<T>(dx2[x] * dy2[y] * X11 +
                                                       dx1[x] * dy2[y] * X21 +
                                                       dx2[x] * dy1[y] * X12 +
                                                       dx1[x] * dy1[y] * X22);
        }
      }
      Xdata += input_height * input_width;
      Ydata += output_width * output_height;
    }
  }
}

template <typename T>
Status Upsample<T>::BaseCompute(OpKernelContext* context, const std::vector<float>& scales) const {
  const auto* X = context->Input<Tensor>(0);
  ORT_ENFORCE(X != nullptr);

  const std::vector<int64_t>& dims = X->Shape().GetDims();
  if (dims.size() != scales.size())
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, 
                  is_resize ? "Resize: input tensor's dimension does not match the scales." : 
                              "Upsample: input tensor's dimension does not match the scales.");

  bool no_scale = true;
  std::vector<int64_t> Y_dims;
  Y_dims.reserve( dims.size() );
  for (std::size_t i = 0; i < dims.size(); i++) {
    int64_t dim_y = static_cast<int64_t>(scales[i] * dims[i]);
    if (no_scale && dim_y != dims[i]) no_scale = false;
    Y_dims.push_back(dim_y);
  }
  Tensor* Y = context->Output(0, Y_dims);

  if (no_scale) {
    memcpy(Y->MutableDataRaw(), X->DataRaw(), Y->SizeInBytes());
    return Status::OK();
  }

  switch (mode_) {
    case UpsampleMode::NN:
      return UpsampleNearest<T>(X->template Data<T>(), Y->template MutableData<T>(), X->Shape(), Y->Shape(), scales, is_resize);
    case UpsampleMode::LINEAR: {
      //The correct behavior of 'linear' mode for an N-D input is not clear right now,
      //so only support 'bilinear' with 2-D or 4-D input tensor with outermost 2 scales as 1 in the 4-D case 
      if (dims.size() != 2 && dims.size() != 4) {
        std::ostringstream oss;
        oss << "'Linear' mode only support 2-D inputs ('Bilinear') or 4-D inputs "
               "with the corresponding outermost 2 scale values being 1 in the ";
        oss << (is_resize ? "Resize operator" : "Upsample operator");
        return Status(ONNXRUNTIME, FAIL, oss.str());      
      }

      bool is_2D = dims.size() == 2;
      const int64_t batch_size = is_2D ? 1 : dims[0];
      const int64_t num_channels = is_2D ? 1 : dims[1];
      const int64_t input_height = is_2D ? dims[0] : dims[2];
      const int64_t input_width = is_2D ? dims[1] : dims[3];

      AllocatorPtr alloc;
      ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
      upsampleBilinear(batch_size, num_channels, input_height, input_width,
                       is_2D ? scales[0] : scales[2], is_2D ? scales[1] : scales[3], 
                       X->template Data<T>(), Y->template MutableData<T>(), alloc);
      return Status::OK();
    }
    default:
      return Status(ONNXRUNTIME, FAIL, is_resize ? "Resize: unexpected mode" : "Upsample: unexpected mode");
  }
}

template <typename T>
Status Upsample<T>::Compute(OpKernelContext* context) const {
  if (OpKernel::Node().InputDefs().size() == 1 || scales_cached_) {
    return BaseCompute(context, scales_);
  }

  const auto* scales = context->Input<Tensor>(1);
  ORT_ENFORCE(scales != nullptr);
  int64_t scales_size = scales->Shape().Size();
  std::vector<float> scales_array(scales_size);
  ParseScalesData(scales, scales_array);
  return BaseCompute(context, scales_array);
}

}  // namespace onnxruntime
