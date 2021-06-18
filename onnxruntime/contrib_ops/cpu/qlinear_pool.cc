// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qlinear_pool.h"

#include "core/util/math_cpuonly.h"
#include "core/providers/common.h"
#include "core/platform/threadpool.h"
#include "core/util/math.h"
#include "core/mlas/inc/mlas.h"

#include <functional>

namespace onnxruntime {

using concurrency::ThreadPool;

namespace contrib {

template <typename T8Bits>
static inline float dequantize_value(T8Bits x, float x_scale, T8Bits x_zero_point);

template <typename T8Bits>
static inline T8Bits quantize_value(float y, float y_scale, T8Bits y_zero_point);

template <>
inline float dequantize_value<uint8_t>(uint8_t x, float x_scale, uint8_t x_zero_point) {
  return x_scale * (static_cast<int>(x) - x_zero_point);
}

template <>
inline uint8_t quantize_value<uint8_t>(float y, float y_scale, uint8_t y_zero_point) {
  return static_cast<uint8_t>(std::max(0.0f, std::min(std::nearbyintf(y / y_scale + y_zero_point), 255.0f)));
}

static void SwitchDimsNchwNhwc(std::vector<int64_t>& dims, bool from_nchw_to_nhwc) {
  if (from_nchw_to_nhwc) {
    int64_t channel = dims[1];
    dims.erase(dims.begin() + 1);
    dims.push_back(channel);
  } else {
    int64_t channel = dims.back();
    dims.insert(dims.begin() + 1, channel);
    dims.pop_back();
  }
}
template <typename T8Bits, typename PoolType>
struct QLinearPool1DTask final {
  const float* X_data;
  T8Bits* Y_data;
  float y_scale;
  T8Bits y_zero_point;
  int64_t x_image_size;
  int64_t y_image_size;
  int64_t pooled_height;
  int64_t stride_h;
  int64_t height;
  const std::vector<int64_t>& kernel_shape;
  const std::vector<int64_t>& pads;
  const PoolProcessContext& pool_context_;
  const PoolAttributes& pool_attrs_;

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * kernel_shape[0]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    for (int64_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }

  void operator()(std::ptrdiff_t c) const {
    const float* x_d = X_data + c * x_image_size;
    T8Bits* y_d = Y_data + c * y_image_size;

    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = std::min(hstart + kernel_shape[0], height);
      hstart = std::max(hstart, static_cast<int64_t>(0));
      float Yh = PoolType::Initialize();
      for (int64_t h = hstart; h < hend; ++h) {
        PoolType::Process(x_d[h], Yh, pool_context_);
      }
      if (pool_attrs_.count_include_pad) {
        PoolType::Finalize(kernel_shape[0], Yh, pool_context_);
      } else {
        PoolType::Finalize(hend - hstart, Yh, pool_context_);
      }
      y_d[ph] = quantize_value(Yh, y_scale, y_zero_point);
    }
  }
};

template <typename T8Bits, typename PoolType>
struct QLinearPoolNhwc1DTask final {
  const float* X_data;
  T8Bits* Y_data;
  float y_scale;
  T8Bits y_zero_point;
  int64_t channels;
  int64_t pooled_height;
  int64_t stride_h;
  int64_t height;
  const std::vector<int64_t>& kernel_shape;
  const std::vector<int64_t>& pads;
  const PoolProcessContext& pool_context_;
  const PoolAttributes& pool_attrs_;

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(channels * kernel_shape[0]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    int64_t y_image_size = pooled_height;
    int64_t batch = begin / y_image_size;
    int64_t offset = begin % y_image_size;

    for (int64_t remains = end - begin; remains > 0; offset = 0, batch++) {
      if (offset + remains <= y_image_size) {
        operator()(std::ptrdiff_t(batch), std::ptrdiff_t(offset), std::ptrdiff_t(offset + remains));
        remains = 0;
      } else {
        operator()(std::ptrdiff_t(batch), std::ptrdiff_t(offset), std::ptrdiff_t(y_image_size));
        remains -= (y_image_size - offset);
      }
    }
  }

  void operator()(std::ptrdiff_t batch, std::ptrdiff_t begin, std::ptrdiff_t end) const {
    const float* x_d = X_data + batch * height * channels;
    T8Bits* y_d = Y_data + batch * pooled_height * channels;
    std::vector<float> Yh(channels, PoolType::Initialize());

    for (int64_t ph = begin, phc = begin * channels; ph < end; ++ph, phc += channels) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = std::min(hstart + kernel_shape[0], height);
      hstart = std::max(hstart, static_cast<int64_t>(0));

      std::fill(Yh.begin(), Yh.end(), PoolType::Initialize());
      for (int64_t h = hstart, hc = hstart * channels; h < hend; ++h, hc += channels) {
        for (int64_t c = 0; c < channels; ++c) {
          PoolType::Process(x_d[hc + c], Yh[c], pool_context_);
        }
      }

      int64_t element_count = (pool_attrs_.count_include_pad) ? kernel_shape[0] : hend - hstart;
      for (int64_t c = 0; c < channels; ++c) {
        PoolType::Finalize(element_count, Yh[c], pool_context_);
        y_d[phc + c] = quantize_value(Yh[c], y_scale, y_zero_point);
      }
    }
  }
};

template <typename T8Bits, typename PoolType>
struct QLinearPool2DTask final {
  const float* X_data;
  T8Bits* Y_data;
  float y_scale;
  T8Bits y_zero_point;
  int64_t x_image_size;
  int64_t y_image_size;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t stride_h;
  int64_t stride_w;
  int64_t height;
  int64_t width;
  const std::vector<int64_t>& kernel_shape;
  const std::vector<int64_t>& pads;
  const PoolProcessContext& pool_context_;
  const PoolAttributes& pool_attrs_;

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * pooled_width * kernel_shape[0] * kernel_shape[1]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    for (int64_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }

  void operator()(std::ptrdiff_t c) const {
    const float* x_d = X_data + c * x_image_size;
    T8Bits* y_d = Y_data + c * y_image_size;

    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = std::min(hstart + kernel_shape[0], height);
      hstart = std::max(hstart, static_cast<int64_t>(0));
      for (int64_t pw = 0; pw < pooled_width; ++pw) {
        int64_t wstart = pw * stride_w - pads[1];
        int64_t wend = std::min(wstart + kernel_shape[1], width);
        wstart = std::max(wstart, static_cast<int64_t>(0));
        const int64_t pool_index = ph * pooled_width + pw;
        float Yh = PoolType::Initialize();
        for (int64_t h = hstart; h < hend; ++h) {
          int64_t input_index = h * width + wstart;
          for (int64_t w = wstart; w < wend; ++w) {
            PoolType::Process(x_d[input_index++], Yh, pool_context_);
          }
        }
        if (pool_attrs_.count_include_pad) {
          PoolType::Finalize(kernel_shape[0] * kernel_shape[1], Yh, pool_context_);
        } else {
          PoolType::Finalize((hend - hstart) * (wend - wstart), Yh, pool_context_);
        }
        y_d[pool_index] = quantize_value(Yh, y_scale, y_zero_point);
      }
    }
  }
};

template <typename T8Bits, typename PoolType>
struct QLinearPoolNhwc2DTask final {
  const float* X_data;
  T8Bits* Y_data;
  float y_scale;
  T8Bits y_zero_point;
  int64_t x_image_size;
  int64_t y_image_size;
  int64_t kernel_size;
  int64_t channels;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t stride_h;
  int64_t stride_w;
  int64_t height;
  int64_t width;
  const std::vector<int64_t>& kernel_shape;
  const std::vector<int64_t>& pads;
  const PoolProcessContext& pool_context_;
  const PoolAttributes& pool_attrs_;

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(channels * kernel_size);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    int64_t batch = begin / y_image_size;
    int64_t offset = begin % y_image_size;

    for (int64_t remains = end - begin; remains > 0; offset = 0, batch++) {
      if (offset + remains <= y_image_size) {
        operator()(std::ptrdiff_t(batch), std::ptrdiff_t(offset), std::ptrdiff_t(offset + remains));
        remains = 0;
      } else {
        operator()(std::ptrdiff_t(batch), std::ptrdiff_t(offset), std::ptrdiff_t(y_image_size));
        remains -= (y_image_size - offset);
      }
    }
  }

  void operator()(std::ptrdiff_t batch, std::ptrdiff_t begin, std::ptrdiff_t end) const {
    const float* x_d = X_data + batch * x_image_size * channels;
    T8Bits* y_d = Y_data + batch * y_image_size * channels;

    // Calculate starting pooled_h, pooled_w, pooled_d
    int64_t start_pw = begin;
    int64_t start_ph = start_pw / pooled_width;
    start_pw -= (start_ph * pooled_width);

    int64_t pool_index = channels * begin;
    int64_t remains = end - begin;
    std::vector<float> Yh(channels);

    for (int64_t ph = start_ph; remains > 0 && ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = std::min(hstart + kernel_shape[0], height);
      hstart = std::max(hstart, static_cast<int64_t>(0));
      for (int64_t pw = start_pw; remains > 0 && pw < pooled_width; ++pw) {
        int64_t wstart = pw * stride_w - pads[1];
        int64_t wend = std::min(wstart + kernel_shape[1], width);
        wstart = std::max(wstart, static_cast<int64_t>(0));

        // do the pooling here
        float pool_init_value = PoolType::Initialize();
        std::fill(Yh.data(), Yh.data() + channels, pool_init_value);
        for (int64_t h = hstart; h < hend; ++h) {
          int64_t input_index = channels * (h * width + wstart);
          for (int64_t w = wstart; w < wend; ++w) {
            for (int64_t c = 0; c < channels; c++) {
              PoolType::Process(x_d[input_index + c], Yh[c], pool_context_);
            }
            input_index += channels;
          }
        }

        int64_t elements_count = (pool_attrs_.count_include_pad) ? kernel_size : (hend - hstart) * (wend - wstart);
        for (int64_t c = 0; c < channels; c++) {
          PoolType::Finalize(elements_count, Yh[c], pool_context_);
          auto y_value = quantize_value(Yh[c], y_scale, y_zero_point);
          y_d[pool_index + c] = y_value;
        }

        pool_index += channels;
        remains--;
      }
      start_pw = 0;
    }
  }
};

template <typename T8Bits, typename PoolType>
struct QLinearPool3DTask final {
  const float* X_data;
  T8Bits* Y_data;
  float y_scale;
  T8Bits y_zero_point;
  int64_t x_image_size;
  int64_t y_image_size;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t pooled_depth;
  int64_t stride_h;
  int64_t stride_w;
  int64_t stride_d;
  int64_t height;
  int64_t width;
  int64_t depth;
  const std::vector<int64_t>& kernel_shape;
  const std::vector<int64_t>& pads;
  const PoolProcessContext& pool_context_;
  const PoolAttributes& pool_attrs_;

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * pooled_width * pooled_depth * kernel_shape[0] *
                                            kernel_shape[1] * kernel_shape[2]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    for (int64_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }

  void operator()(std::ptrdiff_t c) const {
    const float* x_d = X_data + c * x_image_size;
    T8Bits* y_d = Y_data + c * y_image_size;

    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = std::min(hstart + kernel_shape[0], height);
      hstart = std::max(hstart, static_cast<int64_t>(0));
      for (int64_t pw = 0; pw < pooled_width; ++pw) {
        int64_t wstart = pw * stride_w - pads[1];
        int64_t wend = std::min(wstart + kernel_shape[1], width);
        wstart = std::max(wstart, static_cast<int64_t>(0));
        for (int64_t pd = 0; pd < pooled_depth; ++pd) {
          int64_t dstart = pd * stride_d - pads[2];
          int64_t dend = std::min(dstart + kernel_shape[2], depth);
          dstart = std::max(dstart, static_cast<int64_t>(0));
          const int64_t pool_index = ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
          float Yh = PoolType::Initialize();
          for (int64_t h = hstart; h < hend; ++h) {
            const int64_t input_index_h = h * width * depth;
            for (int64_t w = wstart; w < wend; ++w) {
              int64_t input_index = input_index_h + w * depth + dstart;
              for (int64_t d = dstart; d < dend; ++d) {
                PoolType::Process(x_d[input_index++], Yh, pool_context_);
              }
            }
          }
          if (pool_attrs_.count_include_pad) {
            PoolType::Finalize(kernel_shape[0] * kernel_shape[1] * kernel_shape[2], Yh, pool_context_);
          } else {
            PoolType::Finalize((hend - hstart) * (wend - wstart) * (dend - dstart), Yh, pool_context_);
          }
          auto y_value = quantize_value(Yh, y_scale, y_zero_point);
          y_d[pool_index] = y_value;
        }
      }
    }
  }
};

template <typename T8Bits, typename PoolType>
struct QLinearPoolNhwc3DTask final {
  const float* X_data;
  T8Bits* Y_data;
  float y_scale;
  T8Bits y_zero_point;
  int64_t x_image_size;
  int64_t y_image_size;
  int64_t kernel_size;
  int64_t channels;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t pooled_depth;
  int64_t stride_h;
  int64_t stride_w;
  int64_t stride_d;
  int64_t height;
  int64_t width;
  int64_t depth;
  const std::vector<int64_t>& kernel_shape;
  const std::vector<int64_t>& pads;
  const PoolProcessContext& pool_context_;
  const PoolAttributes& pool_attrs_;

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(channels * kernel_size);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    int64_t batch = begin / y_image_size;
    int64_t offset = begin % y_image_size;

    for (int64_t remains = end - begin; remains > 0; offset = 0, batch++) {
      if (offset + remains <= y_image_size) {
        operator()(std::ptrdiff_t(batch), std::ptrdiff_t(offset), std::ptrdiff_t(offset + remains));
        remains = 0;
      } else {
        operator()(std::ptrdiff_t(batch), std::ptrdiff_t(offset), std::ptrdiff_t(y_image_size));
        remains -= (y_image_size - offset);
      }
    }
  }

  void operator()(std::ptrdiff_t batch, std::ptrdiff_t begin, std::ptrdiff_t end) const {
    const float* x_d = X_data + batch * x_image_size * channels;
    T8Bits* y_d = Y_data + batch * y_image_size * channels;

    // Calculate starting pooled_h, pooled_w, pooled_d
    int64_t start_pd = begin;
    int64_t start_ph = start_pd / (pooled_width * pooled_depth);
    start_pd = start_pd - (start_ph * pooled_width * pooled_depth);
    int64_t start_pw = start_pd / pooled_depth;
    start_pd = start_pd - start_pw * pooled_depth;
    int64_t pool_index = channels * begin;
    int64_t remains = end - begin;

    std::vector<float> Yh(channels);

    for (int64_t ph = start_ph; remains > 0 && ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = std::min(hstart + kernel_shape[0], height);
      hstart = std::max(hstart, static_cast<int64_t>(0));
      for (int64_t pw = start_pw; remains > 0 && pw < pooled_width; ++pw) {
        int64_t wstart = pw * stride_w - pads[1];
        int64_t wend = std::min(wstart + kernel_shape[1], width);
        wstart = std::max(wstart, static_cast<int64_t>(0));
        for (int64_t pd = start_pd; remains > 0 && pd < pooled_depth; ++pd) {
          int64_t dstart = pd * stride_d - pads[2];
          int64_t dend = std::min(dstart + kernel_shape[2], depth);
          dstart = std::max(dstart, static_cast<int64_t>(0));

          // do the pooling here
          std::fill(Yh.begin(), Yh.end(), PoolType::Initialize());
          for (int64_t h = hstart; h < hend; ++h) {
            const int64_t input_index_h = h * width * depth;
            for (int64_t w = wstart; w < wend; ++w) {
              int64_t input_index = channels * (input_index_h + w * depth + dstart);
              for (int64_t d = dstart; d < dend; ++d) {
                for (int64_t c = 0; c < channels; c++) {
                  PoolType::Process(x_d[input_index + c], Yh[c], pool_context_);
                }
                input_index += channels;
              }
            }
          }

          int64_t elements_count = (pool_attrs_.count_include_pad) ? kernel_size : (hend - hstart) * (wend - wstart) * (dend - dstart);
          for (int64_t c = 0; c < channels; c++) {
            PoolType::Finalize(elements_count, Yh[c], pool_context_);
            auto y_value = quantize_value(Yh[c], y_scale, y_zero_point);
            y_d[pool_index + c] = y_value;
          }

          pool_index += channels;
          remains--;
        }
        start_pd = 0;
      }
      start_pw = 0;
    }
  }
};

Status QLinearAveragePool::Compute(OpKernelContext* context) const {
  const auto tensor_x_scale = context->Input<Tensor>(1);
  const auto tensor_x_zero_point = context->Input<Tensor>(2);
  const auto tensor_y_scale = context->Input<Tensor>(3);
  const auto tensor_y_zero_point = context->Input<Tensor>(4);

  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_x_scale),
              "Input x_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(tensor_x_zero_point == nullptr || IsScalarOr1ElementVector(tensor_x_zero_point),
              "input x_zero_point must be a scalar or 1D tensor of size 1 if given");
  ORT_ENFORCE(IsScalarOr1ElementVector(tensor_y_scale),
              "input y_scale must be a scalar or 1D tensor of size 1");
  ORT_ENFORCE(tensor_y_zero_point == nullptr || IsScalarOr1ElementVector(tensor_y_zero_point),
              "input y_zero_point must be a scalar or 1D tensor of size 1 if given");

  const auto* X = context->Input<Tensor>(0);
  auto dtype = X->GetElementType();
  if (dtype != ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    ORT_THROW("Unsupported 'dtype' in QLinear Pooling:", dtype);
  }

  TensorShape x_shape = X->Shape();
  const float x_scale = *(tensor_x_scale->Data<float>());
  const float y_scale = *(tensor_y_scale->Data<float>());
  uint8_t x_zero_point = (tensor_x_zero_point ? *(tensor_x_zero_point->Data<uint8_t>()) : (uint8_t)0);
  uint8_t y_zero_point = (tensor_y_zero_point ? *(tensor_y_zero_point->Data<uint8_t>()) : (uint8_t)0);

  ORT_RETURN_IF_NOT(x_shape.NumDimensions() >= 3, "Input dimension cannot be less than 3.");
  std::vector<int64_t> pads = pool_attrs_.pads;
  std::vector<int64_t> strides = pool_attrs_.strides;
  std::vector<int64_t> kernel_shape = pool_attrs_.kernel_shape;

  if (channels_last_) {
    std::vector<int64_t> x_dims = x_shape.GetDims();
    SwitchDimsNchwNhwc(x_dims, false);
    x_shape = TensorShape(x_dims);
  }
  std::vector<int64_t> output_dims = pool_attrs_.SetOutputSize(x_shape, x_shape[1], &pads);

  int64_t batch_count = x_shape[0];
  const int64_t channels = x_shape[1];
  const int64_t height = x_shape[2];
  const int64_t width = kernel_shape.size() > 1 ? x_shape[3] : 1;
  const int64_t depth = kernel_shape.size() > 2 ? x_shape[4] : 1;
  const int64_t pooled_height = output_dims[2];
  const int64_t pooled_width = kernel_shape.size() > 1 ? output_dims[3] : 1;
  const int64_t pooled_depth = kernel_shape.size() > 2 ? output_dims[4] : 1;
  const int64_t total_channels = batch_count * channels;
  const int64_t x_image_size = height * width * depth;
  const int64_t y_image_size = pooled_height * pooled_width * pooled_depth;
  const int64_t kernel_size = std::accumulate(kernel_shape.begin(), kernel_shape.end(), 1LL, std::multiplies<int64_t>());

  if (channels_last_) {
    SwitchDimsNchwNhwc(output_dims, true);
  }
  Tensor* Y = context->Output(0, output_dims);
  const auto* X_data = X->Data<uint8_t>();
  auto* Y_data = Y->MutableData<uint8_t>();

  ThreadPool* tp = context->GetOperatorThreadPool();
  std::vector<float> x_data_fp32;
  if (kernel_shape.size() <= 3) {
    x_data_fp32.resize(x_shape.Size());
    ThreadPool::TryParallelFor(tp, x_shape.Size(), 1.0f, [=, &x_data_fp32](ptrdiff_t first, ptrdiff_t last) {
      const auto* x8 = X_data + first;
      float* x32 = x_data_fp32.data() + first;
      for (ptrdiff_t i = 0, sz = last - first; i < sz; ++i) {
        *x32++ = dequantize_value(x8[i], x_scale, x_zero_point);
      }
    });
  }

  switch (kernel_shape.size()) {
    case 1: {
      if (channels_last_) {
        QLinearPoolNhwc1DTask<uint8_t, onnxruntime::AveragePool> avg_pool_task_1d = {
            x_data_fp32.data(), Y_data, y_scale, y_zero_point, channels,
            pooled_height, strides[0], height, kernel_shape, pads, pool_context_, pool_attrs_};
        ThreadPool::TryParallelFor(tp, y_image_size * batch_count, avg_pool_task_1d.Cost(), avg_pool_task_1d);
      } else {
        QLinearPool1DTask<uint8_t, onnxruntime::AveragePool> avg_pool_task_1d = {
            x_data_fp32.data(), Y_data, y_scale, y_zero_point, x_image_size, y_image_size,
            pooled_height, strides[0], height, kernel_shape, pads, pool_context_, pool_attrs_};
        ThreadPool::TryParallelFor(tp, total_channels, avg_pool_task_1d.Cost(), avg_pool_task_1d);
      }
      break;
    }

    case 2: {
      if (channels_last_) {
        QLinearPoolNhwc2DTask<uint8_t, onnxruntime::AveragePool> avg_pool_task_2d = {
            x_data_fp32.data(), Y_data, y_scale, y_zero_point, x_image_size, y_image_size, kernel_size, channels,
            pooled_height, pooled_width, strides[0], strides[1], height, width, kernel_shape, pads, pool_context_, pool_attrs_};
        ThreadPool::TryParallelFor(tp, y_image_size * batch_count, avg_pool_task_2d.Cost(), avg_pool_task_2d);

      } else {
        QLinearPool2DTask<uint8_t, onnxruntime::AveragePool> avg_pool_task_2d = {
            x_data_fp32.data(), Y_data, y_scale, y_zero_point, x_image_size, y_image_size,
            pooled_height, pooled_width, strides[0], strides[1], height, width, kernel_shape, pads, pool_context_, pool_attrs_};
        ThreadPool::TryParallelFor(tp, total_channels, avg_pool_task_2d.Cost(), avg_pool_task_2d);
      }
      break;
    }

    case 3: {
      if (channels_last_) {
        QLinearPoolNhwc3DTask<uint8_t, onnxruntime::AveragePool> avg_pool_task_3d = {
            x_data_fp32.data(), Y_data, y_scale, y_zero_point, x_image_size, y_image_size, kernel_size, channels,
            pooled_height, pooled_width, pooled_depth, strides[0], strides[1], strides[2], height, width, depth,
            kernel_shape, pads, pool_context_, pool_attrs_};
        ThreadPool::TryParallelFor(tp, y_image_size * batch_count, avg_pool_task_3d.Cost(), avg_pool_task_3d);

      } else {
        QLinearPool3DTask<uint8_t, onnxruntime::AveragePool> avg_pool_task_3d = {
            x_data_fp32.data(), Y_data, y_scale, y_zero_point, x_image_size, y_image_size,
            pooled_height, pooled_width, pooled_depth, strides[0], strides[1], strides[2], height, width, depth,
            kernel_shape, pads, pool_context_, pool_attrs_};
        ThreadPool::TryParallelFor(tp, total_channels, avg_pool_task_3d.Cost(), avg_pool_task_3d);
      }
      break;
    }

    default: {
      return onnxruntime::common::Status(
          onnxruntime::common::ONNXRUNTIME,
          onnxruntime::common::INVALID_ARGUMENT,
          "QLinear Pooling unsupported pooling size!");
    }
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(QLinearAveragePool, kMSDomain, 1, kCpuExecutionProvider, KernelDefBuilder(), QLinearAveragePool);

}  // namespace contrib

}  // namespace onnxruntime
