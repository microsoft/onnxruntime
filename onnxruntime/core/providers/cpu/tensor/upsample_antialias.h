// Copyright c Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/*
 * Pillow 's Resize is corresponding to ONNX op with exclude_outside equaling 1.
 * And, for cubic mode, PIllow has a default value of 0.5 for "cubic_coeff_a",
 * while ONNX op has a default value of 0.75.
 */

#pragma once

#include <algorithm>
#include <cmath>  // for round
#include <vector>
#include "core/framework/tensor.h"
#include "gsl/span"
#ifndef SHARED_PROVIDER
#include "core/framework/op_kernel.h"
#endif
#include "core/providers/cpu/tensor/upsamplebase.h"

namespace onnxruntime {

template <typename T>
struct FilterParamsBaseAntiAlias {
  std::vector<int64_t> bound;
  std::vector<int64_t> out_of_bound_idx;
  int64_t window_size = 2;
  IAllocatorUniquePtr<T> weight_coefficients;
};

template <typename T>
struct FilterParamsAntiAlias {
  float support_size = antialias_constants::kSupportSize;
  float cubic_coeff_a = antialias_constants::kCubicCoeffA;

  FilterParamsBaseAntiAlias<T> dim_x;
  FilterParamsBaseAntiAlias<T> dim_y;
  FilterParamsBaseAntiAlias<T> dim_z;

  const uint8_t* GetClip8LookupTable() const {
    return UpsampleBase::GetLookupTableShared();
  }
  virtual ~FilterParamsAntiAlias() = default;
  virtual float Filter(float x) const = 0;
};

template <typename T>
struct BilinearParamsAntiAlias : FilterParamsAntiAlias<T> {
  // taken from
  // https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/src/libImaging/Resample.c#L20-L29
  float Filter(float x) const override {
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
    this->support_size = antialias_constants::kBiCubicSupportSize;
  }

  // taken from
  // https://github.com/python-pillow/Pillow/blob/6812205f18ca4ef54372e87e1a13ce4a859434df/
  // src/libImaging/Resample.c
  float Filter(float x) const override {
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
  float Filter(float x) const override {
    if (x < 0.0f) {
      x = -x;
    }
    if (x < 1.0f) {
      return 1.0f - x;
    }
    return 0.0f;
  }
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
                                  gsl::span<const int64_t> input_h_w_c,
                                  gsl::span<const int64_t> output_h_w_c,
                                  gsl::span<const float> scale_h_w_c,
                                  gsl::span<const float> roi,
                                  AllocatorPtr& alloc,
                                  const GetOriginalCoordinateFunc& get_original_coordinate,
                                  bool exclude_outside, const bool is_nchw) {
  auto compute_weight_coefficients = [&alloc, roi, &get_original_coordinate, exclude_outside](
                                         const FilterParamsAntiAlias<T>& p,
                                         const int64_t input_size,
                                         const int64_t output_size,
                                         size_t rindex,
                                         FilterParamsBaseAntiAlias<T>& param_base,
                                         const float rscale) -> int64_t {
    param_base.bound.reserve(static_cast<size_t>(output_size) * 2);
    param_base.out_of_bound_idx.reserve(static_cast<size_t>(output_size));

    float scale = 1.0f / rscale;
    float support = (scale >= 1.0f) ? (p.support_size * 0.5f) * scale : p.support_size * 0.5f;

    int32_t window_size = narrow<int32_t>(ceilf(support)) * 2 + 1;
    const size_t scale_buffer_size = narrow<size_t>(window_size * output_size);

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
      if (center - 0.5f < 0 || center - 0.5f > narrow<float>(input_size - 1)) {
        param_base.out_of_bound_idx.emplace_back(i);
      }
      float total_weight = 0.0;

      auto fmin = std::floor(center - support + 0.5f);
      auto fmax = std::floor(center + support + 0.5f);
      int64_t xmin_real = static_cast<int64_t>(fmin);
      int64_t xmax_real = static_cast<int64_t>(fmax);
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
        float w = p.Filter((x + xmin - center + 0.5f) * inv_scale);
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
      for (x = 0; x < xmax_cut - xmin_cut; x++) {
        scale_buffer[x] *= total_weight_inv;

        // normalize the scale to 1 << 22 for int8/uint8
        if constexpr (std::is_same<T, int32_t>::value) {
          scale_buffer_int[x] = static_cast<int32_t>(std::round(scale_buffer[x] * ConstValue::mag_factor_x_2));
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

  p.dim_x.window_size = compute_weight_coefficients(p, input_h_w_c[1], output_h_w_c[1], width_rindex,
                                                    p.dim_x, scale_h_w_c[1]);
  p.dim_y.window_size = compute_weight_coefficients(p, input_h_w_c[0], output_h_w_c[0], height_rindex,
                                                    p.dim_y, scale_h_w_c[0]);
  if (input_h_w_c.size() == 3) {
    p.dim_z.window_size = compute_weight_coefficients(p, input_h_w_c[2], output_h_w_c[2], channel_rindex,
                                                      p.dim_z, scale_h_w_c[2]);
  }
}

/**
 * @brief To compute interpolation along with the last axis.
 * For brief,we assume the input tensor has 3 dimensions and we all it CHW for each character represent a dim.
 * But it doesn't mean the input tensor has semantic meaning of CHW in traditional.
 * we can treat a tensor with rank 4 NCHW as (NC)HW or CHW with a for loop in N dimension.
 * @param num_channels The number of C in CHW.
 * @param input_height The number of H in CHW.
 * @param input_width The number of W in CHW.
 * @param output_height The number of H in CHW.
 * @param output_width The number of W in CHW.
 * @param Xdata_span The input tensor data.
 * @param Ydata_span The output tensor data.
 * @param p The filter params.
 * @param p_dim The filter params for each dim.
 * @param tp The thread pool.
 *
 */
template <typename InputType, typename AccumulateType>
void ComputeInterpolationAtLevel1(int64_t num_channels, int64_t input_height, int64_t input_width,
                                  int64_t output_height, int64_t output_width,
                                  gsl::span<const InputType> Xdata_span, gsl::span<InputType> Ydata_span,
                                  const FilterParamsAntiAlias<AccumulateType>& p,
                                  const FilterParamsBaseAntiAlias<AccumulateType>& p_dim,
                                  concurrency::ThreadPool* tp) {
  const uint8_t* clip8_lookups = &p.GetClip8LookupTable()[640];

  concurrency::ThreadPool::TrySimpleParallelFor(
      tp, narrow<std::ptrdiff_t>(num_channels),
      [&](std::ptrdiff_t c) {
        auto x_start = c * (input_height * input_width);
        auto y_start = c * (output_height * output_width);

        const InputType* Xdata = Xdata_span.data() + x_start;
        InputType* Ydata = Ydata_span.data() + y_start;
        // no need to do scale
        if (output_width == input_width) {
          std::copy_n(Xdata_span.begin() + narrow<size_t>(x_start), narrow<size_t>(output_height * output_width),
                      Ydata_span.begin() + narrow<size_t>(y_start));
          return;
        }

        for (size_t y = 0; y < narrow<size_t>(output_height); ++y) {
          auto* Ydata_offset = Ydata + output_width * y;
          auto* bound = p_dim.bound.data();
          for (size_t x = 0; x < narrow<size_t>(output_width); ++x) {
            AccumulateType output = is_8bit_v<InputType> ? ConstValue::mag_factor : 0;

            const auto* weight_coeff = p_dim.weight_coefficients.get() + p_dim.window_size * x;
            int64_t xmin = *bound++;
            int64_t xmax = *bound++;
            const auto* Xdata_offset = Xdata + y * input_width + xmin;
            for (; xmin < xmax; ++xmin) {
              output += (*Xdata_offset++) * (*weight_coeff++);
            }

            if constexpr (is_8bit_v<InputType>) {
              *Ydata_offset++ = static_cast<InputType>(clip8_lookups[output >> 22]);
            } else if constexpr (std::is_same<InputType, int32_t>::value) {
              *Ydata_offset++ = narrow<int32_t>(std::round(output));
            } else {
              *Ydata_offset++ = output;
            }
          }
        }
      });
}

/**
 * @brief To calculate interpolation along with penultimate axis.
 * For brief, we assume the input tensor has 3 dimensions and we all it CHW for each character represent a dim.
 * But it doesn't mean the input tensor has semantic meaning of CHW in traditional.
 * we can transform a tensor in formats like NCHW,NHWC,NcHWD,CHW,HWC..etc to a rank-3 tensor,
 * then this function can be applied.
 * @param num_channels The number of C in CHW.
 * @param input_height The number of H in CHW.
 * @param input_width The number of W in CHW.
 * @param output_height The number of H in CHW.
 * @param output_width The number of W in CHW.
 * @param Xdata_span The input tensor data.
 * @param Ydata_span The output tensor data.
 * @param p The filter params.
 * @param p_dim The filter params for each dim.
 * @param tp The thread pool.
 */
template <typename InputType, typename AccumulateType>
void ComputeInterpolationAtLevel2(int64_t num_channels, int64_t input_height, int64_t input_width,
                                  int64_t output_height, int64_t output_width,
                                  gsl::span<const InputType> Xdata_span, gsl::span<InputType> Ydata_span,
                                  const FilterParamsAntiAlias<AccumulateType>& p,
                                  const FilterParamsBaseAntiAlias<AccumulateType>& p_dim,
                                  concurrency::ThreadPool* tp) {
  const uint8_t* clip8_lookups = &p.GetClip8LookupTable()[640];
  // This condition is set for higher performance.
  // Observed that TrySimpleParallelFor in dim num_channels is always have higher efficiency, so I would rather
  // choose the first path as long as num_channels is 3 or bigger.
  if (num_channels > 2 && num_channels >= tp->DegreeOfParallelism(tp)) {
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, narrow<std::ptrdiff_t>(num_channels),
        [&](std::ptrdiff_t c) {
          auto x_start = c * (input_height * input_width);
          auto y_start = c * (output_height * output_width);

          const InputType* Xdata = Xdata_span.data() + x_start;
          InputType* Ydata = Ydata_span.data() + y_start;

          if (output_height == input_height) {
            std::copy_n(Xdata_span.begin() + narrow<size_t>(x_start), narrow<size_t>(output_height * output_width),
                        Ydata_span.begin() + narrow<size_t>(y_start));
            return;
          }

          const auto* y_bound = p_dim.bound.data();
          for (size_t y = 0; y < narrow<size_t>(output_height); ++y) {
            const auto* weight_coeff = p_dim.weight_coefficients.get() + p_dim.window_size * y;
            int64_t ymin = *y_bound++;
            int64_t ymax = *y_bound++;
            auto* Ydata_offset = Ydata + output_width * y;
            for (size_t x = 0; x < narrow<size_t>(output_width); ++x) {
              AccumulateType output = is_8bit_v<InputType> ? ConstValue::mag_factor : 0;
              auto* weight_coeff_start = weight_coeff;

              const auto* Xdata_offset = Xdata + ymin * output_width + x;
              for (auto idx = ymin; idx < ymax; ++idx) {
                output += *Xdata_offset * (*weight_coeff_start++);
                Xdata_offset += output_width;
              }

              if constexpr (is_8bit_v<InputType>) {
                *Ydata_offset++ = static_cast<InputType>(clip8_lookups[output >> 22]);
              } else if constexpr (std::is_same<InputType, int32_t>::value) {
                *Ydata_offset++ = narrow<int32_t>(std::round(output));
              } else {  // float double
                *Ydata_offset++ = output;
              }
            }
          }
        });
  } else {
    concurrency::ThreadPool::TryParallelFor(
        tp, static_cast<std::ptrdiff_t>(output_height * num_channels),
        static_cast<double>(output_height * 2),
        [&](std::ptrdiff_t first, std::ptrdiff_t last) {
          if (output_height == input_height) {
            auto workload_in_thread = narrow<size_t>(last) - narrow<size_t>(first);
            std::copy_n(Xdata_span.begin() + narrow<size_t>(first * input_width), narrow<size_t>(workload_in_thread * output_width),
                        Ydata_span.begin() + narrow<size_t>(first * output_width));
            return;
          }

          for (auto start = first; start != last; start++) {
            auto c = start / output_height;
            auto y = start % output_height;

            auto x_start = c * (input_height * input_width);
            auto y_start = c * (output_height * output_width);

            const InputType* Xdata = Xdata_span.data() + x_start;
            InputType* Ydata = Ydata_span.data() + y_start;

            const auto* y_bound = p_dim.bound.data();
            const auto* weight_coeff = p_dim.weight_coefficients.get() + p_dim.window_size * y;
            int64_t ymin = y_bound[2 * narrow<size_t>(y)];
            int64_t ymax = y_bound[2 * narrow<size_t>(y) + 1];
            auto* Ydata_offset = Ydata + output_width * y;
            for (size_t x = 0; x < narrow<size_t>(output_width); ++x) {
              AccumulateType output = is_8bit_v<InputType> ? ConstValue::mag_factor : 0;
              auto* weight_coeff_start = weight_coeff;

              const auto* Xdata_offset = Xdata + ymin * output_width + x;
              for (auto idx = ymin; idx < ymax; ++idx) {
                output += *Xdata_offset * (*weight_coeff_start++);
                Xdata_offset += output_width;
              }

              if constexpr (is_8bit_v<InputType>) {
                *Ydata_offset++ = static_cast<InputType>(clip8_lookups[output >> 22]);
              } else if constexpr (std::is_same<InputType, int32_t>::value) {
                *Ydata_offset++ = narrow<int32_t>(std::round(output));
              } else {  // float double
                *Ydata_offset++ = output;
              }
            }
          }
        });
  }
}

template <typename InputType, typename AccumulateType>
void HandleExtrapolation(int64_t num_channels,
                         int64_t output_height, int64_t output_width, int64_t output_depth,
                         const float extrapolation_value, gsl::span<InputType> Ydata_span,
                         const FilterParamsAntiAlias<AccumulateType>& p,
                         concurrency::ThreadPool* tp) {
  concurrency::ThreadPool::TrySimpleParallelFor(
      tp, static_cast<std::ptrdiff_t>(num_channels),
      [&](std::ptrdiff_t nc) {
        InputType* Ydata_base_nc = Ydata_span.data() + (nc) * (output_depth * output_height * output_width);

        for (int64_t z = 0; z < output_depth && p.dim_x.out_of_bound_idx.size() > 0; ++z) {
          for (int64_t y = 0; y < output_height; ++y) {
            InputType* Ydata_offset = Ydata_base_nc + (z * output_height + y) * output_width;
            for (int64_t idx_x : p.dim_x.out_of_bound_idx) {
              Ydata_offset[narrow<size_t>(idx_x)] = static_cast<InputType>(extrapolation_value);
            }
          }
        }

        for (int64_t z = 0; z < output_depth && p.dim_y.out_of_bound_idx.size() > 0; ++z) {
          for (int64_t y : p.dim_y.out_of_bound_idx) {
            InputType* Ydata_offset = Ydata_base_nc + (z * output_height + y) * output_width;
            std::fill_n(Ydata_offset, narrow<size_t>(output_width), static_cast<InputType>(extrapolation_value));
          }
        }

        for (int64_t z : p.dim_z.out_of_bound_idx) {
          InputType* Ydata_offset = Ydata_base_nc + z * output_height * output_width;
          std::fill_n(Ydata_offset, narrow<size_t>(output_height * output_width), static_cast<InputType>(extrapolation_value));
        }
      });
}

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
  IAllocatorUniquePtr<T> image_temp_buffer = IAllocator::MakeUniquePtr<T>(
      alloc, static_cast<size_t>(input_height * output_width * num_channels));

  for (int64_t n = 0; n < batch_size; ++n) {
    {
      // horizon interpolate
      auto xdata_span = gsl::make_span(Xdata_base + n * (input_height * num_channels * input_width),
                                       narrow<size_t>(input_height * num_channels * input_width));
      auto ydata_span = gsl::make_span(image_temp_buffer.get(), narrow<size_t>(input_height * num_channels * output_width));

      // This computes only the width direction.Thus height keeps unchanged.
      ComputeInterpolationAtLevel1(num_channels, input_height, input_width, input_height, output_width,
                                   xdata_span, ydata_span, p, p.dim_x, tp);
    }
    {
      // vertical interpolate
      auto xdata_span = gsl::make_span<const T>(image_temp_buffer.get(),
                                                narrow<size_t>(input_height * num_channels * output_width));
      auto ydata_span = gsl::make_span<T>(Ydata_base + n * (output_height * num_channels * output_width),
                                          narrow<size_t>(output_height * num_channels * output_width));

      ComputeInterpolationAtLevel2(num_channels, input_height, output_width, output_height, output_width,
                                   xdata_span, ydata_span, p, p.dim_y, tp);
    }
  }
  if (use_extrapolation) {
    auto ydata_span = gsl::make_span<T>(Ydata_base,
                                        narrow<size_t>(batch_size * output_height * num_channels * output_width));
    HandleExtrapolation(batch_size * num_channels, output_height, output_width, 1,
                        extrapolation_value, ydata_span, p, tp);
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
                               gsl::span<const float> roi,
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
                                   gsl::span<const float> roi,
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
                                gsl::span<const float> roi,
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
  IAllocatorUniquePtr<T> image_temp_buffer = IAllocator::MakeUniquePtr<T>(
      alloc, static_cast<size_t>(input_height * output_width * num_channels));

  for (int64_t n = 0; n < batch_size; ++n) {
    // horizon interpolate
    {
      auto xdata_span = gsl::make_span(Xdata_base + n * (input_height * num_channels * input_width),
                                       narrow<size_t>(input_height * num_channels * input_width));
      auto ydata_span = gsl::make_span(image_temp_buffer.get(), narrow<size_t>(input_height * num_channels * output_width));

      ComputeInterpolationAtLevel2(input_height, input_width, num_channels, output_width, num_channels,
                                   xdata_span, ydata_span, p, p.dim_x, tp);
    }

    // vertical interpolate
    {
      // vertical interpolate
      auto xdata_span = gsl::make_span<const T>(image_temp_buffer.get(),
                                                narrow<size_t>(input_height * num_channels * output_width));
      auto ydata_span = gsl::make_span<T>(Ydata_base + n * (output_height * num_channels * output_width),
                                          narrow<size_t>(output_height * num_channels * output_width));

      ComputeInterpolationAtLevel2(1, input_height, output_width * num_channels, output_height, output_width * num_channels,
                                   xdata_span, ydata_span, p, p.dim_y, tp);
    }
  }

  if (use_extrapolation) {
    auto ydata_span = gsl::make_span<T>(Ydata_base,
                                        narrow<size_t>(batch_size * output_height * num_channels * output_width));
    HandleExtrapolation(batch_size * num_channels, output_height, output_width, 1,
                        extrapolation_value, ydata_span, p, tp);
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
                            gsl::span<const float> roi,
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
                               alloc, get_original_coordinate, exclude_outside, true);

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
                                gsl::span<const float> roi,
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

  IAllocatorUniquePtr<T> image_temp_buffer = IAllocator::MakeUniquePtr<T>(
      alloc, static_cast<size_t>(batch_size * output_height * output_width *
                                 input_depth * num_channels));

  UpsampleBaseAntiAlias<T>(p, batch_size, num_channels * input_depth, input_height, input_width, output_height, output_width,
                           false, extrapolation_value,
                           X->Data<T>(), image_temp_buffer.get(), alloc, tp);

  auto m_batch_size = batch_size * num_channels < tp->DegreeOfParallelism(tp) ? 1 : batch_size;
  auto m_channel_size = batch_size * num_channels < tp->DegreeOfParallelism(tp) ? num_channels * batch_size : num_channels;
  for (int64_t n = 0; n < m_batch_size; ++n) {
    // depth interpolate
    {
      // depth interpolate
      auto xdata_span = gsl::make_span<const T>(image_temp_buffer.get() + n * (output_height * num_channels * output_width * input_depth),
                                                narrow<size_t>(output_height * num_channels * output_width * input_depth));
      auto ydata_span = gsl::make_span<T>(Ydata_base + n * (output_height * num_channels * output_width * output_depth),
                                          narrow<size_t>(output_height * num_channels * output_width * output_depth));

      ComputeInterpolationAtLevel2(m_channel_size, input_depth, output_height * output_width, output_depth, output_height * output_width,
                                   xdata_span, ydata_span, p, p.dim_z, tp);
    }
  }

  if (use_extrapolation) {
    auto ydata_span = gsl::make_span<T>(Ydata_base,
                                        narrow<size_t>(batch_size * output_height * num_channels * output_width * output_depth));
    HandleExtrapolation(batch_size * num_channels, output_height, output_width, output_depth,
                        extrapolation_value, ydata_span, p, tp);
  }
}

}  // namespace onnxruntime
