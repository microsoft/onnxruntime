// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

constexpr const char* UpsampleModeNN = "nearest";
constexpr const char* UpsampleModeLinear = "linear";

enum UpsampleMode {
  NN = 0,      // nearest neighbour
  LINEAR = 1,  // linear interpolation
};

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
                       const std::vector<float>& scales) {
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
        old_idx += (std::min(static_cast<int64_t>(tmp / scales[j]), input_shape[j] - 1)) * base;
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
                     const std::vector<float>& scales) {
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

class UpsampleBase {
 protected:
  UpsampleBase(OpKernelInfo info) : scales_cached_(false) {
    std::string mode;
    ORT_ENFORCE(info.GetAttr<std::string>("mode", &mode).IsOK());
    mode_ = StringToUpsampleMode(mode);

    auto input_count = info.GetInputCount();
    if (input_count == 1) {
      ORT_ENFORCE(info.GetAttrs<float>("scales", scales_).IsOK());
      ScalesValidation(scales_, mode_);
    }

    // opset 9
    if (input_count > 1) {
      const Tensor* scale;
      bool get_scale = info.TryGetConstantInput(1, &scale);

      if (get_scale) {
        ParseScalesData(scale, scales_);
        scales_cached_ = true;
      }
    }
  }

  UpsampleMode mode_;
  std::vector<float> scales_;
  bool scales_cached_;

  UpsampleMode StringToUpsampleMode(const std::string& mode) {
    if (strcmp(mode.c_str(), UpsampleModeNN) == 0) {
      return UpsampleMode::NN;
    } else if (strcmp(mode.c_str(), UpsampleModeLinear) == 0) {
      return UpsampleMode::LINEAR;
    } else {
      ORT_THROW("mode attribute is " + mode + ". It can only be " +
                UpsampleModeNN + "(default) or " + UpsampleModeLinear + ".");
    }
  }

  void ScalesValidation(const std::vector<float>& scales, const UpsampleMode mode) const {
    for (auto& scale : scales) {
      ORT_ENFORCE(scale >= 1, "Scale value should be greater than or equal to 1.");
    }

    if (UpsampleMode::LINEAR == mode) {
      ORT_ENFORCE(scales.size() == 4, "Upsample: linear mode upsample only support bilinear with 4 dimension.");
      ORT_ENFORCE(((scales[0] == 1) && (scales[1] == 1)),
                  "Upsample: linear mode upsample only support bilinear, the first 2 scales should be 1.");
    }
  }

  void ParseScalesData(const Tensor* scale, std::vector<float>& scales) const {
    const float* scale_data = scale->template Data<float>();
    int64_t scales_size = scale->Shape().Size();
    ORT_ENFORCE(scales_size > 0, "scales size should be greater than 0.");
    if (scales.size() == 0) {
      scales.resize(scales_size);
    }
    memcpy(scales.data(), scale_data, scales_size * sizeof(float));
    ScalesValidation(scales, mode_);
  }
};

template <typename T>
class Upsample : public UpsampleBase, public OpKernel {
 public:
  Upsample(OpKernelInfo info) : UpsampleBase(info), OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

  Status BaseCompute(OpKernelContext* context, const std::vector<float>& scales) const;
};

}  // namespace onnxruntime
