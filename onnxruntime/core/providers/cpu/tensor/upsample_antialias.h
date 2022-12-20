// Copyright c Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>  // for round
#include <vector>
#include "core/framework/tensor.h"
#include "gsl/span"
#ifndef SHARED_PROVIDER
#include "core/framework/op_kernel.h"
#endif
#include "core/providers/cpu/tensor/upsamplebase.h"

namespace onnxruntime {

namespace ConstValue {
constexpr int32_t mag_factor = 1 << (22 - 1);
}

template <typename T>
struct FilterParamsBaseAntiAlias {
  std::vector<int64_t> bound;
  std::vector<float> original;
  std::vector<int64_t> out_of_bound_idx;
  int64_t window_size = 2;
  IAllocatorUniquePtr<T> weight_coefficients;
};

template <typename T>
struct FilterParamsAntiAlias {
  float support_size = 2.0f;
  float cubic_coeff_a = -0.75f;

  /* Handles values form -640 to 639. */
  uint8_t* clip8_lookups_table{nullptr};

  FilterParamsBaseAntiAlias<T> dim_x;
  FilterParamsBaseAntiAlias<T> dim_y;
  FilterParamsBaseAntiAlias<T> dim_z;

  void init_clip_lookup() {
    // if we have already initialized the lookup table, just return
    // ideally we could have a global lookup table, but that account for too much space.
    if (clip8_lookups_table[1279] == 255) {
      return;
    }

    // taken from https://github.com/python-pillow/Pillow/blob/66add095a50d76c35c7f58643461f2edf78a3f05/src/libImaging/Resample.c#L94
    //  we need to handle negative values
    //  it's equivalent to :x = np.clip(x, 0, 255) where x \in [-640, 639]
    // we will accept a negative x for (&clip8_lookups_table[640])[x] means clip8_lookups_table +640 -x
    for (int i = 0; i < 1280; ++i) {
      clip8_lookups_table[i] = static_cast<uint8_t>(std::min(std::max(i - 640, 0), 255));
    }
  }
  virtual ~FilterParamsAntiAlias() = default;
  virtual float filter(float x) const = 0;
};

template <typename T>
struct BilinearParamsAntiAlias : FilterParamsAntiAlias<T> {
  // taken from
  // https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/
  // src/libImaging/Resample.c#L20-L29
  float filter(float x) const override {
    if (x < 0.0f) {
      x = -x;
    }
    if (x < 1.0f) {
      return 1.0f - x;
    }
    return 0.0f;
  }
};

template <typename T>
struct BiCubicParamsAntiAlias : FilterParamsAntiAlias<T> {
  BiCubicParamsAntiAlias() {
    this->support_size = 4.0f;
  }

  // taken from
  // https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/
  // src/libImaging/Resample.c
  float filter(float x) const override {
    /* https://en.wikipedia.org/wiki/Bicubic_interpolation#Bicubic_convolution_algorithm
     */
    if (x < 0.0f) {
      x = -x;
    }
    if (x < 1.0f) {
      return ((this->cubic_coeff_a + 2.0f) * x - (this->cubic_coeff_a + 3.0f)) * x * x + 1;
    }
    if (x < 2.0f) {
      return (((x - 5.0f) * x + 8.f) * x - 4.f) * this->cubic_coeff_a;
    }
    return 0.0f;
  }
};

template <typename T>
struct TriLinearParamsAntiAlias : FilterParamsAntiAlias<T> {
  float filter(float x) const override {
    if (x < 0.0f) {
      x = -x;
    }
    if (x < 1.0f) {
      return 1.0f - x;
    }
    return 0.0f;
  }
};

template <typename T>
struct AccumulateType {
  using type = int32_t;
  using Dtype = T;
};

template <>
struct AccumulateType<int32_t> {
  using type = float;
};

template <>
struct AccumulateType<float> {
  using type = float;
};

template <>
struct AccumulateType<double> {
  using type = double;
};

// The following method supports a 3/4/5-D input in 'Linear mode, cubic mode'
// that amounts to 'Bilinear,TriLinear, Bicubic/Tricubic' Upsampling/Resizing in the sense that it assumes
// A N-D tensor has
// 1. the scale values for the outermost 2 dimensions are 1 or
// 2. the scale values for the outermost and innermost dimensions are 1
// This is the common use-case where the 4-D input (batched multi-channel images)
// is usually of shapes:
// - [N, C, H, W] and the scales are [1.0, 1.0, height_scale, width_scale]
// - [N, H, W, C] and the scales are [1.0, height_scale, width_scale, 1.0]
template <class T>
void SetupUpsampleFilterAntiAlias(FilterParamsAntiAlias<T>& p,
                                  const gsl::span<int64_t> input_h_w_c,
                                  const gsl::span<int64_t> output_h_w_c,
                                  const gsl::span<float> scale_h_w_c,
                                  const std::vector<float>& roi,
                                  AllocatorPtr& alloc,
                                  const GetOriginalCoordinateFunc& get_original_coordinate,
                                  bool exclude_outside, const bool is_nchw) {
  auto compute_weight_coefficients = [&alloc, &roi, &get_original_coordinate, exclude_outside](const FilterParamsAntiAlias<T>& p,
                                                                                               const int64_t input_size,
                                                                                               const int64_t output_size,
                                                                                               size_t rindex,
                                                                                               FilterParamsBaseAntiAlias<T>& param_base,
                                                                                               const float rscale) -> int64_t {
    param_base.bound.reserve(static_cast<size_t>(output_size) * 2);
    param_base.original.reserve(static_cast<size_t>(output_size));
    param_base.out_of_bound_idx.reserve(static_cast<size_t>(output_size));

    float scale = 1.0f / rscale;
    float support = (scale >= 1.0f) ? (p.support_size * 0.5f) * scale : p.support_size * 0.5f;

    int32_t window_size = narrow<int32_t>(ceilf(support)) * 2 + 1;
    const size_t scale_buffer_size = window_size * output_size;

    param_base.weight_coefficients = IAllocator::MakeUniquePtr<T>(alloc, scale_buffer_size);
    // Get pointers to appropriate memory locations in the scratch buffer
    auto* scale_data = reinterpret_cast<float*>(param_base.weight_coefficients.get());
    int64_t xmin = 0, xmax = 0;
    float inv_scale = (scale >= 1.0f) ? 1.0f / scale : 1.0f;

    const auto roi_start = roi.size() / 2 - (rindex + 1);
    const auto roi_end = roi.size() - (rindex + 1);

    for (int32_t i = 0; i < output_size; i++) {
      // double center = (i + 0.5) * scale;
      float center = 0.5f + (scale == 1.0f ? static_cast<float>(i)
                                           : get_original_coordinate(static_cast<float>(i), rscale,
                                                                     static_cast<float>(output_size),
                                                                     static_cast<float>(input_size),
                                                                     roi[roi_start], roi[roi_end]));
      param_base.original.emplace_back(center - 0.5f);
      if (center - 0.5f < 0) {
        param_base.out_of_bound_idx.emplace_back(i);
      }
      float total_weight = 0.0;

      int64_t xmin_real = static_cast<int64_t>(floor(center - support + 0.5));
      int64_t xmax_real = static_cast<int64_t>(floor(center + support + 0.5));
      int64_t xmin_cut = std::max<int64_t>(xmin_real, (0));
      int64_t xmax_cut = std::min<int64_t>(xmax_real, input_size);

      xmin = exclude_outside ? xmin_cut : xmin_real;
      xmax = exclude_outside ? xmax_cut : xmax_real;
      param_base.bound.push_back(xmin_cut);
      param_base.bound.push_back(xmax_cut);

      auto* scale_buffer = &scale_data[i * window_size];
      int64_t x = 0;
      xmax -= xmin;
      for (; x < xmax; x++) {
        float w = p.filter((x + xmin - center + 0.5f) * inv_scale);
        scale_buffer[x] = w;
        total_weight += w;
      }

      if (!exclude_outside) {
        int64_t neg_xsize = xmin < 0 ? -xmin : 0;
        for (x = 0; x < neg_xsize; x++) {
          scale_buffer[neg_xsize] += scale_buffer[x];
        }

        int64_t bound_xsize =
            xmax + xmin > input_size ? xmax + xmin - input_size : 0;
        for (x = xmax - bound_xsize; x < xmax; x++) {
          scale_buffer[xmax - bound_xsize - 1] +=
              scale_buffer[x];
        }

        for (x = 0; (neg_xsize | bound_xsize) > 0 && x < xmax_cut - xmin_cut; x++) {
          scale_buffer[x] = scale_buffer[x + neg_xsize];
        }
      }

      float total_weight_inv = total_weight == 0.0f ? 1.f : 1.0f / total_weight;
      auto* scale_buffer_int = reinterpret_cast<int32_t*>(scale_buffer);
      for (x = 0; x < xmax; x++) {
        scale_buffer[x] *= total_weight_inv;

        // normalize the scale to 1 << 22 for int8/uint8
        if constexpr (std::is_same<T, int32_t>::value) {
          scale_buffer_int[x] = static_cast<int32_t>(std::round(scale_buffer[x] * ConstValue::mag_factor * 2.f));
        }
      }
      /*for (; x < window_size; x++) {
        scale_buffer[x] = 0;
      }*/
    }
    return window_size;
  };

  const size_t width_rindex = is_nchw ? 0 : 1;
  const size_t height_rindex = is_nchw ? 1 : 2;
  const size_t channel_rindex = is_nchw ? 2 : 2;  // only works for trilinear NC(chw)

  /* Handles values form -640 to 639. */
  static uint8_t clip8_lookups_table[1280];
  p.clip8_lookups_table = clip8_lookups_table;

  p.init_clip_lookup();
  p.dim_y.window_size = compute_weight_coefficients(p, input_h_w_c[0], output_h_w_c[0], height_rindex,
                                                    p.dim_y, scale_h_w_c[0]);
  p.dim_x.window_size = compute_weight_coefficients(p, input_h_w_c[1], output_h_w_c[1], width_rindex,
                                                    p.dim_x, scale_h_w_c[1]);
  if (input_h_w_c.size() == 3) {
    p.dim_z.window_size = compute_weight_coefficients(p, input_h_w_c[2], output_h_w_c[2], channel_rindex,
                                                      p.dim_z, scale_h_w_c[2]);
  }
}

template <class T>
inline constexpr bool is_8bit_v = std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;

template <typename T, typename T1>
void UpsampleBaseAntiAlias(FilterParamsAntiAlias<T1>& p,
                           const int64_t batch_size,
                           const int64_t num_channels,
                           const int64_t input_height,
                           const int64_t input_width,
                           const int64_t output_height,
                           const int64_t output_width,
                           const bool use_extrapolation,
                           const float extrapolation_value,
                           const T* Xdata_base,
                           T* Ydata_base,
                           AllocatorPtr& alloc,
                           concurrency::ThreadPool* tp) {
  const uint8_t* clip8_lookups = &p.clip8_lookups_table[640];

  IAllocatorUniquePtr<T> image_temp_buffer = IAllocator::MakeUniquePtr<T>(
      alloc, static_cast<size_t>(input_height * output_width * num_channels));

  using ACtype = T1;

  for (int64_t n = 0; n < batch_size; ++n) {
    auto* temp_buffer = image_temp_buffer.get();
    // horizon interpolate

    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, narrow<std::ptrdiff_t>(num_channels),
        [&](std::ptrdiff_t c) {
          auto x_start = (n * num_channels + c) * (input_height * input_width);
          auto y_start = c * (input_height * output_width);

          const T* Xdata = Xdata_base + x_start;
          T* Ydata = temp_buffer + y_start;
          // no need to do scale
          if (output_width == input_width) {
            auto output_size = narrow<size_t>(input_height * output_width);

            auto hc_prod = input_height * num_channels;
            auto xdata_span = gsl::make_span(Xdata_base, batch_size * hc_prod * input_width);
            auto ydata_span = gsl::make_span(temp_buffer, hc_prod * output_width);

            std::copy_n(xdata_span.begin() + narrow<size_t>(x_start), narrow<size_t>(output_size),
                        ydata_span.begin() + narrow<size_t>(y_start));
            return;
          }

          for (size_t y = 0; y < narrow<size_t>(input_height); ++y) {
            auto* Ydata_offset = Ydata + output_width * y;
            auto* bound = p.dim_x.bound.data();
            for (size_t x = 0; x < narrow<size_t>(output_width); ++x) {
              // when use_extrapolation is set and original index of x or y is out of the dim range
              // then use extrapolation_value as the output value.
              if (use_extrapolation &&
                  ((p.dim_y.original[y] < 0 || p.dim_y.original[y] > static_cast<float>(input_height - 1)) ||
                   (p.dim_x.original[x] < 0 || p.dim_x.original[x] > static_cast<float>(input_width - 1)))) {
                *Ydata_offset++ = static_cast<T>(extrapolation_value);
                continue;
              }
              ACtype output = is_8bit_v<T> ? ConstValue::mag_factor : 0;

              const auto* weight_coeff = p.dim_x.weight_coefficients.get() + p.dim_x.window_size * x;
              int64_t xmin = *bound++;
              int64_t xmax = *bound++;
              const auto* Xdata_offset = Xdata + y * input_width + xmin;
              for (; xmin < xmax; ++xmin) {
                output += (*Xdata_offset++) * (*weight_coeff++);
              }

              if constexpr (is_8bit_v<T>) {
                *Ydata_offset++ = static_cast<T>(clip8_lookups[output >> 22]);
              } else if constexpr (std::is_same<T, int32_t>::value) {
                *Ydata_offset++ = narrow<int32_t>(std::round(output));
              } else {
                *Ydata_offset++ = output;
              }
            }
          }
        });

    // vertical interpolate
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, narrow<std::ptrdiff_t>(num_channels),
        [&](std::ptrdiff_t c) {
          auto x_start = c * (input_height * output_width);
          auto y_start = (n * num_channels + c) * (output_height * output_width);

          const T* Xdata = temp_buffer + x_start;
          T* Ydata = Ydata_base + y_start;

          if (output_height == input_height) {
            memcpy(Ydata, Xdata, sizeof(T) * narrow<size_t>(output_height * output_width));
            auto output_size = output_height * output_width;

            auto hc_prod = input_height * num_channels;
            auto xdata_span = gsl::make_span(temp_buffer, hc_prod * input_width);
            auto ydata_span = gsl::make_span(Ydata_base, batch_size * hc_prod * output_width);

            std::copy_n(xdata_span.begin() + narrow<size_t>(x_start), narrow<size_t>(output_size),
                        ydata_span.begin() + narrow<size_t>(y_start));
            return;
          }

          const auto* y_bound = p.dim_y.bound.data();
          for (size_t y = 0; y < narrow<size_t>(output_height); ++y) {
            const auto* weight_coeff = p.dim_y.weight_coefficients.get() + p.dim_y.window_size * y;
            int64_t ymin = *y_bound++;
            int64_t ymax = *y_bound++;
            auto* Ydata_offset = Ydata + output_width * y;
            for (size_t x = 0; x < narrow<size_t>(output_width); ++x) {
              if (use_extrapolation &&
                  ((p.dim_y.original[y] < 0 || p.dim_y.original[y] > static_cast<float>(input_height - 1)) ||
                   (p.dim_x.original[x] < 0 || p.dim_x.original[x] > static_cast<float>(input_width - 1)))) {
                *Ydata_offset++ = static_cast<T>(extrapolation_value);
                continue;
              }

              ACtype output = is_8bit_v<T> ? ConstValue::mag_factor : 0;
              auto* weight_coeff_start = weight_coeff;

              const auto* Xdata_offset = Xdata + ymin * output_width + x;
              for (auto idx = ymin; idx < ymax; ++idx) {
                output += *Xdata_offset * (*weight_coeff_start++);
                Xdata_offset += output_width;
              }
              if constexpr (is_8bit_v<T>) {
                *Ydata_offset++ = static_cast<T>(clip8_lookups[output >> 22]);
              } else if constexpr (std::is_same<T, int32_t>::value) {
                *Ydata_offset++ = narrow<int32_t>(std::round(output));
              } else {  // float double
                *Ydata_offset++ = output;
              }
            }
          }
        });
  }
}

template <typename T>
void UpsampleBilinearAntiAlias(const int64_t batch_size,
                               const int64_t num_channels,
                               const int64_t input_height,
                               const int64_t input_width,
                               const int64_t output_height,
                               const int64_t output_width,
                               const float height_scale,
                               const float width_scale,
                               const std::vector<float>& roi,
                               const bool use_extrapolation,
                               const float extrapolation_value,
                               bool exclude_outside,
                               const Tensor* X,
                               T* Ydata_base,
                               AllocatorPtr& alloc,
                               const GetOriginalCoordinateFunc& get_original_coordinate,
                               concurrency::ThreadPool* tp) {
  int64_t input_paras[] = {input_height, input_width};
  int64_t output_paras[] = {output_height, output_width};
  float scale_paras[] = {height_scale, width_scale};
  BilinearParamsAntiAlias<typename AccumulateType<T>::type> p;
  SetupUpsampleFilterAntiAlias(p, input_paras, output_paras, scale_paras, roi,
                               alloc, get_original_coordinate, exclude_outside, true);
  return UpsampleBaseAntiAlias<T>(p, batch_size, num_channels, input_height, input_width, output_height, output_width,
                                  use_extrapolation, extrapolation_value,
                                  X->Data<T>(), Ydata_base, alloc, tp);
}

template <typename T>
void NhwcUpsampleBilinearAntiAlias(const int64_t batch_size,
                                   const int64_t num_channels,
                                   const int64_t input_height,
                                   const int64_t input_width,
                                   const int64_t output_height,
                                   const int64_t output_width,
                                   const float height_scale,
                                   const float width_scale,
                                   const std::vector<float>& roi,
                                   const bool use_extrapolation,
                                   const float extrapolation_value,
                                   bool exclude_outside,
                                   const Tensor* X,
                                   T* Ydata_base,
                                   AllocatorPtr& alloc,
                                   const GetOriginalCoordinateFunc& get_original_coordinate,
                                   concurrency::ThreadPool* tp) {
  int64_t input_paras[] = {input_height, input_width};
  int64_t output_paras[] = {output_height, output_width};
  float scale_paras[] = {height_scale, width_scale};
  BilinearParamsAntiAlias<typename AccumulateType<T>::type> p;
  SetupUpsampleFilterAntiAlias(p, input_paras, output_paras, scale_paras, roi,
                               alloc, get_original_coordinate, exclude_outside, false);
  return NhwcUpsampleBasicAntiAlias(p, batch_size, num_channels, input_height, input_width, output_height, output_width,
                                    use_extrapolation, extrapolation_value,
                                    X->Data<T>(), Ydata_base, alloc, tp);
}

template <typename T>
void NhwcResizeBiCubicAntiAlias(const int64_t batch_size,
                                const int64_t num_channels,
                                const int64_t input_height,
                                const int64_t input_width,
                                const int64_t output_height,
                                const int64_t output_width,
                                const float height_scale,
                                const float width_scale,
                                float cubic_coeff_a,
                                bool use_extrapolation,
                                float extrapolation_value,
                                bool exclude_outside,
                                const std::vector<float>& roi,
                                const Tensor* X,
                                T* Ydata_base,
                                AllocatorPtr& alloc,
                                const GetOriginalCoordinateFunc& get_original_coordinate,
                                concurrency::ThreadPool* tp) {
  int64_t input_paras[] = {input_height, input_width};
  int64_t output_paras[] = {output_height, output_width};
  float scale_paras[] = {height_scale, width_scale};
  BiCubicParamsAntiAlias<typename AccumulateType<T>::type> p;
  p.cubic_coeff_a = cubic_coeff_a;
  SetupUpsampleFilterAntiAlias(p, input_paras, output_paras, scale_paras, roi,
                               alloc, get_original_coordinate, exclude_outside, false);
  return NhwcUpsampleBasicAntiAlias(p, batch_size, num_channels, input_height, input_width, output_height, output_width,
                                    use_extrapolation, extrapolation_value,
                                    X->Data<T>(), Ydata_base, alloc, tp);
}

template <typename T, typename T1>
void NhwcUpsampleBasicAntiAlias(FilterParamsAntiAlias<T1>& p,
                                const int64_t batch_size,
                                const int64_t num_channels,
                                const int64_t input_height,
                                const int64_t input_width,
                                const int64_t output_height,
                                const int64_t output_width,
                                const bool use_extrapolation,
                                const float extrapolation_value,
                                const T* Xdata_base,
                                T* Ydata_base,
                                AllocatorPtr& alloc,
                                concurrency::ThreadPool* tp) {
  const uint8_t* clip8_lookups = &p.clip8_lookups_table[640];

  IAllocatorUniquePtr<T> image_temp_buffer = IAllocator::MakeUniquePtr<T>(
      alloc, static_cast<size_t>(input_height * output_width * num_channels));

  using ACtype = T1;

  for (int64_t n = 0; n < batch_size; ++n) {
    auto* temp_buffer = image_temp_buffer.get();

    // horizon interpolate
    concurrency::ThreadPool::TryParallelFor(
        tp, static_cast<std::ptrdiff_t>(input_height * output_width),
        static_cast<double>(num_channels * 2),
        [&](std::ptrdiff_t first, std::ptrdiff_t last) {
          const T* Xdata = Xdata_base + n * (input_height * input_width) * num_channels;
          T* Ydata = temp_buffer;
          for (std::ptrdiff_t i = first; i < last; ++i) {
            const auto x = static_cast<size_t>(i % output_width);
            const auto y = static_cast<size_t>(i / output_width);
            T* Ydata_with_offset = Ydata + i * num_channels;
            if (use_extrapolation && ((p.dim_y.original[y] < 0 || p.dim_y.original[y] > static_cast<float>(input_height - 1)) ||
                                      (p.dim_x.original[x] < 0 || p.dim_x.original[x] > static_cast<float>(input_width - 1)))) {
              for (size_t c = 0; c < narrow<size_t>(num_channels); ++c) {
                Ydata_with_offset[c] = static_cast<T>(extrapolation_value);
              }
              continue;
            }

            const auto* weight_coeff = p.dim_x.weight_coefficients.get() + p.dim_x.window_size * x;
            int64_t xmin = p.dim_x.bound[x * 2];
            int64_t xmax = p.dim_x.bound[x * 2 + 1];
            for (size_t c = 0; c < narrow<size_t>(num_channels); ++c) {
              const auto* weight_coeff_start = weight_coeff;
              ACtype output = is_8bit_v<T> ? ConstValue::mag_factor : 0;

              for (int64_t idx = xmin; idx < xmax; ++idx) {
                output += Xdata[narrow<size_t>((y * input_width + idx) * num_channels + c)] *
                          (*weight_coeff_start++);
              }
              if constexpr (is_8bit_v<T>) {
                Ydata_with_offset[c] = static_cast<T>(clip8_lookups[output >> 22]);
              } else if constexpr (std::is_same<T, int32_t>::value) {
                Ydata_with_offset[c] = narrow<int32_t>(std::round(output));
              } else {  // float double
                Ydata_with_offset[c] = output;
              }
            }
          }
        });

    // vertical interpolate
    concurrency::ThreadPool::TryParallelFor(
        tp, static_cast<std::ptrdiff_t>(output_height * output_width),
        static_cast<double>(num_channels * 2),
        [&](std::ptrdiff_t first, std::ptrdiff_t last) {
          const T* Xdata = temp_buffer;
          T* Ydata = Ydata_base + n * (output_height * output_width) * num_channels;

          for (std::ptrdiff_t i = first; i < last; ++i) {
            const auto x = static_cast<size_t>(i % output_width);
            const auto y = static_cast<size_t>(i / output_width);
            T* Ydata_with_offset = Ydata + (output_width * y + x) * num_channels;

            if (use_extrapolation && ((p.dim_y.original[y] < 0 || p.dim_y.original[y] > static_cast<float>(input_height - 1)) ||
                                      (p.dim_x.original[x] < 0 || p.dim_x.original[x] > static_cast<float>(input_width - 1)))) {
              for (size_t c = 0; c < narrow<size_t>(num_channels); ++c) {
                Ydata_with_offset[c] = static_cast<T>(extrapolation_value);
              }
              continue;
            }

            const auto* weight_coeff = p.dim_y.weight_coefficients.get() + p.dim_y.window_size * y;
            int64_t ymin = p.dim_y.bound[y * 2];
            int64_t ymax = p.dim_y.bound[y * 2 + 1];

            for (int64_t c = 0; c < num_channels; ++c) {
              const auto* weight_coeff_start = weight_coeff;
              ACtype output = is_8bit_v<T> ? ConstValue::mag_factor : 0;

              for (int64_t idy = ymin; idy < ymax; ++idy) {
                output += Xdata[narrow<size_t>((idy * output_width + x) * num_channels + c)] *
                          (*weight_coeff_start++);
              }
              if constexpr (is_8bit_v<T>) {
                Ydata_with_offset[c] = static_cast<T>(clip8_lookups[output >> 22]);
              } else if constexpr (std::is_same<T, int32_t>::value) {
                Ydata_with_offset[c] = narrow<int32_t>(std::round(output));
              } else {  // float double
                Ydata_with_offset[c] = output;
              }
            }
          }
        });
  }
}

template <typename T>
void ResizeBiCubicAntiAlias(int64_t batch_size,
                            int64_t num_channels,
                            int64_t input_height,
                            int64_t input_width,
                            int64_t output_height,
                            int64_t output_width,
                            float height_scale,
                            float width_scale,
                            float cubic_coeff_a,
                            bool use_extrapolation,
                            float extrapolation_value,
                            bool exclude_outside,
                            const std::vector<float>& roi,
                            const Tensor* X,
                            T* Ydata_base,
                            AllocatorPtr& alloc,
                            const GetOriginalCoordinateFunc& get_original_coordinate,
                            concurrency::ThreadPool* tp) {
  int64_t input_paras[] = {input_height, input_width};
  int64_t output_paras[] = {output_height, output_width};
  float scale_paras[] = {height_scale, width_scale};
  BiCubicParamsAntiAlias<typename AccumulateType<T>::type> p;
  p.cubic_coeff_a = cubic_coeff_a;
  SetupUpsampleFilterAntiAlias(p, input_paras, output_paras, scale_paras, roi,
                               alloc, get_original_coordinate, exclude_outside, false);

  return UpsampleBaseAntiAlias<T>(p, batch_size, num_channels, input_height, input_width, output_height, output_width,
                                  use_extrapolation, extrapolation_value,
                                  X->Data<T>(), Ydata_base, alloc, tp);
}

template <typename T>
void UpsampleTrilinearAntiAlias(int64_t batch_size,
                                int64_t num_channels,
                                int64_t input_depth,
                                int64_t input_height,
                                int64_t input_width,
                                int64_t output_depth,
                                int64_t output_height,
                                int64_t output_width,
                                float depth_scale,
                                float height_scale,
                                float width_scale,
                                const std::vector<float>& roi,
                                bool use_extrapolation,
                                float extrapolation_value,
                                bool exclude_outside,
                                const Tensor* X,
                                T* Ydata_base,
                                AllocatorPtr& alloc,
                                const GetOriginalCoordinateFunc& get_original_coordinate,
                                concurrency::ThreadPool* tp) {
  int64_t input_paras[] = {input_height, input_width, input_depth};
  int64_t output_paras[] = {output_height, output_width, output_depth};
  float scale_paras[] = {height_scale, width_scale, depth_scale};

  TriLinearParamsAntiAlias<typename AccumulateType<T>::type> p;
  SetupUpsampleFilterAntiAlias(p, input_paras, output_paras, scale_paras, roi,
                               alloc, get_original_coordinate, exclude_outside, true);
  const uint8_t* clip8_lookups = &p.clip8_lookups_table[640];

  IAllocatorUniquePtr<T> image_temp_buffer = IAllocator::MakeUniquePtr<T>(
      alloc, static_cast<size_t>(batch_size * output_height * output_width *
                                 input_depth * num_channels));

  UpsampleBaseAntiAlias<T>(p, batch_size, num_channels * input_depth, input_height, input_width, output_height, output_width,
                           use_extrapolation, extrapolation_value,
                           X->Data<T>(), image_temp_buffer.get(), alloc, tp);
  using ACtype = typename AccumulateType<T>::type;

  for (int64_t n = 0; n < batch_size; ++n) {
    // reuse it for each batch
    auto* temp_buffer = image_temp_buffer.get();

    // channel interpolate
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, narrow<std::ptrdiff_t>(num_channels),
        [&](std::ptrdiff_t c) {
          auto out_wh_prod = output_height * output_width;
          auto x_start = (n * num_channels + c) * (out_wh_prod * input_depth);
          auto y_start = (n * num_channels + c) * (out_wh_prod * output_depth);

          const T* Xdata = temp_buffer + x_start;
          T* Ydata = Ydata_base + y_start;

          if (output_depth == input_depth) {
            auto output_size = out_wh_prod * output_depth;
            auto bwhc_prod = batch_size * out_wh_prod * num_channels;
            auto xdata_span = gsl::make_span(temp_buffer, bwhc_prod * input_depth);
            auto ydata_span = gsl::make_span(Ydata_base, bwhc_prod * output_depth);

            std::copy_n(xdata_span.begin() + narrow<size_t>(x_start), narrow<size_t>(output_size),
                        ydata_span.begin() + narrow<size_t>(y_start));
            return;
          }

          const auto* z_bound = p.dim_z.bound.data();
          for (size_t z = 0; z < narrow<size_t>(output_depth); ++z) {
            const auto* weight_coeff = p.dim_z.weight_coefficients.get() + p.dim_z.window_size * z;
            int64_t zmin = *z_bound++;
            int64_t zmax = *z_bound++;
            auto* Ydata_base_z = Ydata + z * out_wh_prod;
            for (size_t y = 0; y < narrow<size_t>(output_height); ++y) {
              for (size_t x = 0; x < narrow<size_t>(output_width); ++x) {
                auto* Ydata_offset = Ydata_base_z + y * output_width + x;

                if (use_extrapolation &&
                    ((p.dim_y.original[y] < 0 || p.dim_y.original[y] > static_cast<float>(input_height - 1)) ||
                     (p.dim_x.original[x] < 0 || p.dim_x.original[x] > static_cast<float>(input_width - 1)) ||
                     ((p.dim_z.original[y] < 0 || p.dim_z.original[y] > static_cast<float>(input_depth - 1))))) {
                  *Ydata_offset = static_cast<T>(extrapolation_value);
                  continue;
                }

                ACtype output = is_8bit_v<T> ? ConstValue::mag_factor : 0;
                auto* weight_coeff_start = weight_coeff;

                const auto* Xdata_offset = Xdata + (zmin * output_height + y) * output_width + x;
                for (auto idx = zmin; idx < zmax; ++idx) {
                  output += *Xdata_offset * (*weight_coeff_start++);
                  Xdata_offset += out_wh_prod;
                }
                if constexpr (is_8bit_v<T>) {
                  *Ydata_offset = static_cast<T>(clip8_lookups[output >> 22]);
                } else if constexpr (std::is_same<T, int32_t>::value) {
                  *Ydata_offset = narrow<int32_t>(std::round(output));
                } else {  // float double
                  *Ydata_offset = output;
                }
              }
            }
          }
        });
  }
}

}  // namespace onnxruntime
