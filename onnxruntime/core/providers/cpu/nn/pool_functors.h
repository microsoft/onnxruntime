// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/platform/threadpool.h"
#include "core/providers/cpu/nn/pool_base.h"
#include <deque>
#include <iostream>
#include <functional>

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
  const std::vector<int64_t>& kernel_shape;
  const std::vector<int64_t>& pads;
  const PoolProcessContext& pool_context_;
  const PoolAttributes& pool_attrs_;
  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * kernel_shape[0]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t c = begin; c < end; ++c) {
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
  const std::vector<int64_t>& kernel_shape;
  const std::vector<int64_t>& pads;
  const PoolProcessContext& pool_context_;
  const PoolAttributes& pool_attrs_;

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * pooled_width * kernel_shape[0] * kernel_shape[1]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t c = begin; c < end; ++c) {
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
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t c = begin; c < end; ++c) {
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
  const std::vector<int64_t>& kernel_shape;
  const std::vector<int64_t>& pads;
  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * kernel_shape[0]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
#ifdef _OPENMP
#pragma omp parallel for
    for (int64_t c = begin; c < end; ++c) {
      std::unique_ptr<T[]> que_ptr{new T[kernel_shape[0]]};
#else
    std::unique_ptr<T[]> que_ptr{new T[kernel_shape[0]]};
    for (int64_t c = begin; c < end; ++c) {
#endif
      operator()(c, que_ptr);
    }
  }

  void operator()(std::ptrdiff_t c, std::unique_ptr<T[]>& que_ptr) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;
    int64_t* i_d = I_data ? I_data + c * y_step : nullptr;

    if (stride_h == dilation_h && !i_d) {
      //std::cout << "in optimized maxpool" << std::endl;
      auto que = que_ptr.get();
      auto end = que + kernel_shape[0] - 1;
      auto tail = que;
      auto head = que;
      int64_t cnt = 0;

      T padding = std::numeric_limits<T>::lowest();
      int64_t global_indice = 0;
      int64_t global_end = 2 * pads[0] + height;
      int64_t first_end = dilation_h * kernel_shape[0] - 1;

      for (; global_indice < first_end; global_indice += dilation_h) {
        auto x_indice = global_indice - pads[0];
        if (x_indice < 0) {
          *tail++ = padding;
          ++cnt;
        } else {
          auto back = tail == que ? end : tail - 1;
          while (cnt > 0) {
            if (*x_d > *back) {
               tail = back;
               back = tail == que ? end : tail - 1;
               --cnt; 
            } else {
              break;
            }
          }//while
          *tail = *x_d;
          tail = tail == end ? que : tail + 1;
          ++cnt;
          x_d += dilation_h; 
        }//else
      }//for
      
      for (int64_t pool_start = 0; global_indice < global_end;
           pool_start += dilation_h, global_indice += dilation_h) {

        auto x_indice = global_indice - pads[0];
        if (x_indice >= pads[0] + height) {
          *y_d = *head;
          break;
        } else {
          auto back = tail == que ? end : tail - 1;
          while (cnt > 0) {
            if (*x_d > *back) {
               tail = back;
               back = tail == que ? end : tail - 1;
               --cnt;
            } else {
              break;
            }
          }//while
          *tail = *x_d;
          tail = tail == end ? que : tail + 1;
          ++cnt;
          x_d += dilation_h; 
          *y_d++ = *head;
          if (*head == x_d[pool_start]) {
            head = head == end ? que : head + 1;
            --cnt;
          }//if
        } //else
      } //for
    } else {
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
      }//for
    }//else
  }

/*
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

  using SaveMaxFunctor = std::function<void(T,int64_t,int64_t)>;

  void operator()(std::ptrdiff_t c) const {
    const T* x_d = X_data + c * x_step;
    T* y_d = Y_data + c * y_step;
    int64_t* i_d = I_data ? I_data + c * y_step : nullptr;
    std::unordered_set<int64_t> filled;

    SaveMaxFunctor save_max_functor = [this, &filled, c, y_d, i_d] (T max_value, int64_t max_indice, int64_t x_indice) {
      if (x_indice % stride_h == 0) {
        auto y_indice = x_indice / stride_h;
        filled.insert(y_indice);
        y_d[y_indice] = max_value;
        //std::cout << "save y data at " << y_indice << std::endl;
        if (i_d) {
          i_d[y_indice] = c * x_step + max_indice;
        }//if
      }//if
    };//save_max_functor
    
    for (int64_t ph = 0; ph < pooled_height; ++ph) {
      if (filled.find(ph) == filled.end()) {
        MaxPool1DVector(x_d, ph * stride_h, height, dilation_h, pads[0], kernel_shape[0], save_max_functor);
      }
    }
  }

  static void MaxPool1DVector(const T* from_vector, int64_t start_at, int64_t len, int64_t dilation,
                              int64_t pads, int64_t pool_size, SaveMaxFunctor& save_max_fn)
  {
    struct Value {
      T value;
      int64_t indice;
      bool operator == (const Value& V) const { return V.value == value && V.indice == indice; }
      bool operator > (const Value& V) const { return value > V.value; }
    };

    std::deque<Value> values;
    std::deque<Value> maxes;

    auto enqueue = [&] (Value& value) {
      values.push_back(value);
      while (!maxes.empty() && value > maxes.back()) {
        maxes.pop_back();
      }
      maxes.push_back(value);
      if (values.size() == static_cast<size_t>(pool_size)) {
        auto max_indice = maxes.front().indice - pads;
        save_max_fn(maxes.front().value,
                    max_indice < 0 || max_indice >= len ? len : max_indice,
                    values.front().indice);
        auto popped_value = values.front();
        values.pop_front();
        if (popped_value == maxes.front()) {
          maxes.pop_front();
        }//if
      }//if
    };//update_deque

    int64_t global_indice = start_at;
    auto padding = std::numeric_limits<T>::lowest();

    for (int32_t indice = start_at; indice < pads; indice += dilation) {
      Value value = {padding, global_indice};
      enqueue(value);
      global_indice += dilation;
    }//for

    for (auto indice = global_indice - pads; indice < len; indice += dilation) {
      Value value = {from_vector[indice], global_indice};
      enqueue(value);
      global_indice += dilation;
    }//for 

    for (int32_t indice = global_indice - pads - len; indice < pads; indice += dilation) {
      Value value = {padding, global_indice};
      enqueue(value);
      global_indice += dilation;
    }//for
  }//MaxPoolVector
*/
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
  const std::vector<int64_t>& kernel_shape;
  const std::vector<int64_t>& pads;
  int64_t storage_order;

  TensorOpCost Cost() {
    double loop_count = static_cast<double>(pooled_height * pooled_width * kernel_shape[0] * kernel_shape[1]);
    return TensorOpCost{loop_count, loop_count, loop_count};
  }

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t c = begin; c < end; ++c) {
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
  const std::vector<int64_t>& kernel_shape;
  const std::vector<int64_t>& pads;
  int64_t storage_order;

  void operator()(std::ptrdiff_t begin, std::ptrdiff_t end) const {
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int64_t c = begin; c < end; ++c) {
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
            i_d[pool_index] = storage_order == 0 ? c * x_step + h_index * width * depth + w_index * depth + d_index :
                                                   c * x_step + h_index + w_index * height + d_index * height * width;
        }
      }
    }
  }
};

}  // namespace onnxruntime
