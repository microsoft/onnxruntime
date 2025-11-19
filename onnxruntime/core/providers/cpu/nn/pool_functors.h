// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/platform/threadpool.h"
#include "core/providers/cpu/nn/pool_base.h"
namespace onnxruntime {

template <typename T, typename PoolType>
struct Pool1DTask final {
  const T* X_data;
  T* Y_data;
  int64_t x_step;
  int64_t y_step;
  int64_t pooled_height;
  int64_t stride_h;
  int64_t height;
  gsl::span<const int64_t> kernel_shape;
  gsl::span<const int64_t> pads;
  const PoolProcessContext& pool_context_;
  const PoolAttributes& pool_attrs_;
  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * kernel_shape[0]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    for (std::ptrdiff_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }
  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;

    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = std::min(hstart + kernel_shape[0], height);
      hstart = std::max(hstart, static_cast<int64_t>(0));
      T Yh = PoolType::Initialize();
      for (int64_t h = hstart; h < hend; ++h) {
        PoolType::Process(x_d[h], Yh, pool_context_);
      }
      if (pool_attrs_.count_include_pad) {
        PoolType::Finalize(kernel_shape[0], Yh, pool_context_);
      } else {
        PoolType::Finalize(hend - hstart, Yh, pool_context_);
      }
      y_d[ph] = Yh;
    }
  }
};

template <typename T, typename PoolType>
struct Pool2DTask final {
  const T* X_data;
  T* Y_data;
  int64_t x_step;
  int64_t y_step;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t stride_h;
  int64_t stride_w;
  int64_t height;
  int64_t width;
  gsl::span<const int64_t> kernel_shape;
  gsl::span<const int64_t> pads;
  const PoolProcessContext& pool_context_;
  const PoolAttributes& pool_attrs_;

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * pooled_width * kernel_shape[0] * kernel_shape[1]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    for (std::ptrdiff_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }

  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;

    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = std::min(hstart + kernel_shape[0], height);
      hstart = std::max(hstart, static_cast<int64_t>(0));
      for (int64_t pw = 0; pw < pooled_width; ++pw) {
        int64_t wstart = pw * stride_w - pads[1];
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
        if (pool_attrs_.count_include_pad) {
          PoolType::Finalize(kernel_shape[0] * kernel_shape[1], Yh, pool_context_);
        } else {
          PoolType::Finalize((hend - hstart) * (wend - wstart), Yh, pool_context_);
        }
        y_d[pool_index] = Yh;
      }
    }
  }
};

template <typename T, typename PoolType>
struct Pool3DTask final {
  const T* X_data;
  T* Y_data;
  int64_t x_step;
  int64_t y_step;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t pooled_depth;
  int64_t stride_h;
  int64_t stride_w;
  int64_t stride_d;
  int64_t height;
  int64_t width;
  int64_t depth;
  gsl::span<const int64_t> kernel_shape;
  gsl::span<const int64_t> pads;
  const PoolProcessContext& pool_context_;
  const PoolAttributes& pool_attrs_;

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * pooled_width * pooled_depth * kernel_shape[0] *
                                            kernel_shape[1] * kernel_shape[2]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    for (std::ptrdiff_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }

  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;

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
          T Yh = PoolType::Initialize();
          for (int64_t h = hstart; h < hend; ++h) {
            for (int64_t w = wstart; w < wend; ++w) {
              for (int64_t d = dstart; d < dend; ++d) {
                const int64_t input_index = h * width * depth + w * depth + d;
                PoolType::Process(x_d[input_index], Yh, pool_context_);
              }
            }
          }
          if (pool_attrs_.count_include_pad) {
            PoolType::Finalize(kernel_shape[0] * kernel_shape[1] * kernel_shape[2], Yh, pool_context_);
          } else {
            PoolType::Finalize((hend - hstart) * (wend - wstart) * (dend - dstart), Yh, pool_context_);
          }
          y_d[pool_index] = Yh;
        }
      }
    }
  }
};

template <typename T>
struct MaxPool1DTask final {
  const T* X_data;
  T* Y_data;
  int64_t* I_data;
  int64_t x_step;
  int64_t y_step;
  int64_t dilation_h;
  int64_t pooled_height;
  int64_t stride_h;
  int64_t height;
  gsl::span<const int64_t> kernel_shape;
  gsl::span<const int64_t> pads;
  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * kernel_shape[0]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    for (std::ptrdiff_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }
  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;
    int64_t* i_d = I_data ? I_data + c * y_step : nullptr;
    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = hstart + kernel_shape[0] * dilation_h;
      T Yh = std::numeric_limits<T>::lowest();
      int64_t h_index = -1;
      for (int64_t h = hstart; h < hend; h += dilation_h) {
        if (math::is_a_ge_zero_and_a_lt_b(h, height)) {
          if (x_d[h] > Yh) {
            Yh = x_d[h];
            h_index = h;
          }
        }
      }
      y_d[ph] = Yh;
      if (i_d != nullptr)
        i_d[ph] = c * x_step + h_index;
    }
  }
};

template <typename T>
struct MaxPool2DTask final {
  const T* X_data;
  T* Y_data;
  int64_t* I_data;
  int64_t x_step;
  int64_t y_step;
  int64_t dilation_h;
  int64_t dilation_w;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t stride_h;
  int64_t stride_w;
  int64_t height;
  int64_t width;
  gsl::span<const int64_t> kernel_shape;
  gsl::span<const int64_t> pads;
  int64_t storage_order;

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * pooled_width * kernel_shape[0] * kernel_shape[1]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    for (std::ptrdiff_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }

  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;
    int64_t* i_d = I_data ? I_data + c * y_step : nullptr;
    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = hstart + kernel_shape[0] * dilation_h;
      for (int64_t pw = 0; pw < pooled_width; ++pw) {
        int64_t wstart = pw * stride_w - pads[1];
        int64_t wend = wstart + kernel_shape[1] * dilation_w;
        const int64_t pool_index = ph * pooled_width + pw;
        T Yh = std::numeric_limits<T>::lowest();
        int64_t h_index = -1;
        int64_t w_index = -1;
        for (int64_t h = hstart; h < hend; h += dilation_h) {
          if (math::is_a_ge_zero_and_a_lt_b(h, height)) {
            for (int64_t w = wstart; w < wend; w += dilation_w) {
              if (math::is_a_ge_zero_and_a_lt_b(w, width)) {
                const int64_t input_index = h * width + w;
                if (x_d[input_index] > Yh) {
                  Yh = x_d[input_index];
                  h_index = h;
                  w_index = w;
                }
              }
            }
          }
        }
        y_d[pool_index] = Yh;
        if (i_d != nullptr)
          i_d[pool_index] =
              storage_order == 0 ? c * x_step + h_index * width + w_index : c * x_step + h_index + w_index * height;
      }
    }
  }
};

template <typename T>
struct MaxPool3DTask {
  const T* X_data;
  T* Y_data;
  int64_t* I_data;
  int64_t x_step;
  int64_t y_step;
  int64_t dilation_h;
  int64_t dilation_w;
  int64_t dilation_d;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t pooled_depth;
  int64_t stride_h;
  int64_t stride_w;
  int64_t stride_d;
  int64_t height;
  int64_t width;
  int64_t depth;
  gsl::span<const int64_t> kernel_shape;
  gsl::span<const int64_t> pads;
  int64_t storage_order;

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    for (std::ptrdiff_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * pooled_width * pooled_depth * kernel_shape[0] *
                                            kernel_shape[1] * kernel_shape[2]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;
    int64_t* i_d = I_data ? I_data + c * y_step : nullptr;

    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = hstart + kernel_shape[0] * dilation_h;
      for (int64_t pw = 0; pw < pooled_width; ++pw) {
        int64_t wstart = pw * stride_w - pads[1];
        int64_t wend = wstart + kernel_shape[1] * dilation_w;
        for (int64_t pd = 0; pd < pooled_depth; ++pd) {
          int64_t dstart = pd * stride_d - pads[2];
          int64_t dend = dstart + kernel_shape[2] * dilation_d;
          const int64_t pool_index = ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
          T Yh = std::numeric_limits<T>::lowest();
          int64_t h_index = -1;
          int64_t w_index = -1;
          int64_t d_index = -1;
          for (int64_t h = hstart; h < hend; h += dilation_h) {
            if (math::is_a_ge_zero_and_a_lt_b(h, height)) {
              for (int64_t w = wstart; w < wend; w += dilation_w) {
                if (math::is_a_ge_zero_and_a_lt_b(w, width)) {
                  for (int64_t d = dstart; d < dend; d += dilation_d) {
                    if (math::is_a_ge_zero_and_a_lt_b(d, depth)) {
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
              }
            }
          }
          y_d[pool_index] = Yh;
          if (i_d != nullptr)
            i_d[pool_index] = storage_order == 0 ? c * x_step + h_index * width * depth + w_index * depth + d_index : c * x_step + h_index + w_index * height + d_index * height * width;
        }
      }
    }
  }
};

template <typename T>
struct AveragePool1DTask final {
  const T* X_data;
  T* Y_data;
  int64_t x_step;
  int64_t y_step;
  int64_t dilation_h;
  int64_t pooled_height;
  int64_t stride_h;
  int64_t height;
  gsl::span<const int64_t> kernel_shape;
  gsl::span<const int64_t> pads;
  bool count_include_pad;
  int64_t p;
  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * kernel_shape[0]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    for (std::ptrdiff_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }
  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;
    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = hstart + kernel_shape[0] * dilation_h;
      hend = std::min(hend, height + pads[1]);
      y_d[ph] = 0;
      int total_elements = 0;
      for (int64_t h = hstart; h < hend; h += dilation_h) {
        if (math::is_a_ge_zero_and_a_lt_b(h, height)) {
          y_d[ph] += x_d[h];
          total_elements++;
        }
      }
      if (total_elements > 0) {
        if (count_include_pad) {
          y_d[ph] /= (1 + (hend - hstart - 1) / dilation_h);
        } else {
          y_d[ph] /= total_elements;
        }
      }
    }
  }
};

template <typename T>
struct AveragePool2DTask final {
  const T* X_data;
  T* Y_data;
  int64_t x_step;
  int64_t y_step;
  int64_t dilation_h;
  int64_t dilation_w;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t stride_h;
  int64_t stride_w;
  int64_t height;
  int64_t width;
  gsl::span<const int64_t> kernel_shape;
  gsl::span<const int64_t> pads;
  bool count_include_pad;
  int64_t p;

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * pooled_width * kernel_shape[0] * kernel_shape[1]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    for (std::ptrdiff_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }

  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;
    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = hstart + kernel_shape[0] * dilation_h;
      hend = std::min(hend, height + pads[1]);
      for (int64_t pw = 0; pw < pooled_width; ++pw) {
        int64_t wstart = pw * stride_w - pads[1];
        int64_t wend = wstart + kernel_shape[1] * dilation_w;
        wend = std::min(wend, width + pads[3]);
        const int64_t pool_index = ph * pooled_width + pw;
        y_d[pool_index] = 0;
        int total_elements = 0;
        for (int64_t h = hstart; h < hend; h += dilation_h) {
          if (math::is_a_ge_zero_and_a_lt_b(h, height)) {
            for (int64_t w = wstart; w < wend; w += dilation_w) {
              if (math::is_a_ge_zero_and_a_lt_b(w, width)) {
                const int64_t input_index = h * width + w;
                y_d[pool_index] += x_d[input_index];
                total_elements++;
              }
            }
          }
        }
        if (total_elements > 0) {
          if (count_include_pad) {
            y_d[pool_index] /= ((1 + (hend - hstart - 1) / dilation_h) * (1 + (wend - wstart - 1) / dilation_w));
          } else {
            y_d[pool_index] /= total_elements;
          }
        }
      }
    }
  }
};

template <typename T>
struct AveragePool3DTask {
  const T* X_data;
  T* Y_data;
  int64_t x_step;
  int64_t y_step;
  int64_t dilation_h;
  int64_t dilation_w;
  int64_t dilation_d;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t pooled_depth;
  int64_t stride_h;
  int64_t stride_w;
  int64_t stride_d;
  int64_t height;
  int64_t width;
  int64_t depth;
  gsl::span<const int64_t> kernel_shape;
  gsl::span<const int64_t> pads;
  bool count_include_pad;
  int64_t p;

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    for (std::ptrdiff_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * pooled_width * pooled_depth * kernel_shape[0] *
                                            kernel_shape[1] * kernel_shape[2]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;

    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = hstart + kernel_shape[0] * dilation_h;
      hend = std::min(hend, height + pads[1]);
      for (int64_t pw = 0; pw < pooled_width; ++pw) {
        int64_t wstart = pw * stride_w - pads[1];
        int64_t wend = wstart + kernel_shape[1] * dilation_w;
        wend = std::min(wend, width + pads[3]);
        for (int64_t pd = 0; pd < pooled_depth; ++pd) {
          int64_t dstart = pd * stride_d - pads[2];
          int64_t dend = dstart + kernel_shape[2] * dilation_d;
          dend = std::min(dend, depth + pads[5]);
          const int64_t pool_index = ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
          y_d[pool_index] = 0;
          int total_elements = 0;
          for (int64_t h = hstart; h < hend; h += dilation_h) {
            if (math::is_a_ge_zero_and_a_lt_b(h, height)) {
              for (int64_t w = wstart; w < wend; w += dilation_w) {
                if (math::is_a_ge_zero_and_a_lt_b(w, width)) {
                  for (int64_t d = dstart; d < dend; d += dilation_d) {
                    if (math::is_a_ge_zero_and_a_lt_b(d, depth)) {
                      const int64_t input_index = h * width * depth + w * depth + d;
                      y_d[pool_index] += x_d[input_index];
                      total_elements++;
                    }
                  }
                }
              }
            }
          }
          if (total_elements > 0) {
            if (count_include_pad) {
              y_d[pool_index] /= ((1 + (hend - hstart - 1) / dilation_h) * (1 + (wend - wstart - 1) / dilation_w) * (1 + (dend - dstart - 1) / dilation_d));
            } else {
              y_d[pool_index] /= total_elements;
            }
          }
        }
      }
    }
  }
};

template <typename T>
struct LpPool1DTask final {
  const T* X_data;
  T* Y_data;
  int64_t x_step;
  int64_t y_step;
  int64_t dilation_h;
  int64_t pooled_height;
  int64_t stride_h;
  int64_t height;
  gsl::span<const int64_t> kernel_shape;
  gsl::span<const int64_t> pads;
  int64_t p;
  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * kernel_shape[0]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    for (std::ptrdiff_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }
  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;
    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = hstart + kernel_shape[0] * dilation_h;
      y_d[ph] = 0;
      for (int64_t h = hstart; h < hend; h += dilation_h) {
        if (math::is_a_ge_zero_and_a_lt_b(h, height)) {
          y_d[ph] += static_cast<T>(std::pow(std::abs(x_d[h]), p));
        }
      }
      y_d[ph] = static_cast<T>(std::pow(y_d[ph], 1.0f / p));
    }
  }
};

template <typename T>
struct LpPool2DTask final {
  const T* X_data;
  T* Y_data;
  int64_t x_step;
  int64_t y_step;
  int64_t dilation_h;
  int64_t dilation_w;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t stride_h;
  int64_t stride_w;
  int64_t height;
  int64_t width;
  gsl::span<const int64_t> kernel_shape;
  gsl::span<const int64_t> pads;
  int64_t p;

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * pooled_width * kernel_shape[0] * kernel_shape[1]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    for (std::ptrdiff_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }

  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;
    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = hstart + kernel_shape[0] * dilation_h;
      for (int64_t pw = 0; pw < pooled_width; ++pw) {
        int64_t wstart = pw * stride_w - pads[1];
        int64_t wend = wstart + kernel_shape[1] * dilation_w;
        const int64_t pool_index = ph * pooled_width + pw;
        y_d[pool_index] = 0;
        for (int64_t h = hstart; h < hend; h += dilation_h) {
          if (math::is_a_ge_zero_and_a_lt_b(h, height)) {
            for (int64_t w = wstart; w < wend; w += dilation_w) {
              if (math::is_a_ge_zero_and_a_lt_b(w, width)) {
                const int64_t input_index = h * width + w;
                y_d[pool_index] += static_cast<T>(std::pow(std::abs(x_d[input_index]), p));
              }
            }
          }
        }
        y_d[pool_index] = static_cast<T>(std::pow(y_d[pool_index], 1.0f / p));
      }
    }
  }
};

template <typename T>
struct LpPool3DTask {
  const T* X_data;
  T* Y_data;
  int64_t x_step;
  int64_t y_step;
  int64_t dilation_h;
  int64_t dilation_w;
  int64_t dilation_d;
  int64_t pooled_height;
  int64_t pooled_width;
  int64_t pooled_depth;
  int64_t stride_h;
  int64_t stride_w;
  int64_t stride_d;
  int64_t height;
  int64_t width;
  int64_t depth;
  gsl::span<const int64_t> kernel_shape;
  gsl::span<const int64_t> pads;
  int64_t p;

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
    for (std::ptrdiff_t c = begin; c < end; ++c) {
      operator()(c);
    }
  }

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * pooled_width * pooled_depth * kernel_shape[0] *
                                            kernel_shape[1] * kernel_shape[2]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;

    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      int64_t hstart = ph * stride_h - pads[0];
      int64_t hend = hstart + kernel_shape[0] * dilation_h;
      for (int64_t pw = 0; pw < pooled_width; ++pw) {
        int64_t wstart = pw * stride_w - pads[1];
        int64_t wend = wstart + kernel_shape[1] * dilation_w;
        for (int64_t pd = 0; pd < pooled_depth; ++pd) {
          int64_t dstart = pd * stride_d - pads[2];
          int64_t dend = dstart + kernel_shape[2] * dilation_d;
          const int64_t pool_index = ph * pooled_width * pooled_depth + pw * pooled_depth + pd;
          y_d[pool_index] = 0;
          for (int64_t h = hstart; h < hend; h += dilation_h) {
            if (math::is_a_ge_zero_and_a_lt_b(h, height)) {
              for (int64_t w = wstart; w < wend; w += dilation_w) {
                if (math::is_a_ge_zero_and_a_lt_b(w, width)) {
                  for (int64_t d = dstart; d < dend; d += dilation_d) {
                    if (math::is_a_ge_zero_and_a_lt_b(d, depth)) {
                      const int64_t input_index = h * width * depth + w * depth + d;
                      y_d[pool_index] += static_cast<T>(std::pow(std::abs(x_d[input_index]), p));
                    }
                  }
                }
              }
            }
          }
          y_d[pool_index] = static_cast<T>(std::pow(y_d[pool_index], 1.0f / p));
        }
      }
    }
  }
};

}  // namespace onnxruntime
