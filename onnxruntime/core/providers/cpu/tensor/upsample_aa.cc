// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <vector>
#include "core/common/safeint.h"
#include "core/platform/threadpool.h"
#include "core/providers/cpu/tensor/upsample_aa.h"
#include "gsl/span"

using namespace onnxruntime::common;
using namespace std;
using onnxruntime::narrow;
namespace onnxruntime {

// The following method supports a 4-D input in 'Linear mode'
// that amounts to 'Bilinear' Upsampling/Resizing in the sense that it assumes
// 1. the scale values for the outermost 2 dimensions are 1 or
// 2. the scale values for the outermost and innermost dimensions are 1
// This is the common use-case where the 4-D input (batched multi-channel images)
// is usually of shapes:
// - [N, C, H, W] and the scales are [1.0, 1.0, height_scale, width_scale]
// - [N, H, W, C] and the scales are [1.0, height_scale, width_scale, 1.0]
void SetupUpsampleFilterAA(FilterParamsAA& p,
                           const gsl::span<int64_t> input_h_w_c,
                           const gsl::span<int64_t> output_h_w_c,
                           const gsl::span<float> scale_h_w_c,
                           const std::vector<float>& roi,
                           AllocatorPtr& alloc,
                           const GetOriginalCoordinateFunc& get_original_coordinate,
                           const int32_t dtype, bool exclude_outside, const bool is_nchw) {
  auto compute_weight_coefficients = [&alloc, &roi, &get_original_coordinate, dtype, exclude_outside](const FilterParamsAA& p,
                                                                                                      const int64_t input_size,
                                                                                                      const int64_t output_size,
                                                                                                      size_t rindex,
                                                                                                      std::vector<int64_t>& bound_idx,
                                                                                                      std::vector<float>& original_idx,
                                                                                                      const float rscale,
                                                                                                      BufferUniquePtr& weight_coefficients) -> int64_t {
    bound_idx.reserve(static_cast<size_t>(output_size) * 2);
    // For each index in the output height and output width, cache its corresponding "weights/scales" for its
    // corresponding indices in the input which proportionately indicates how much they will influence the final
    // pixel value in the output
    // (cache because we don't have to re-compute each time we come across the output width/output height
    // value while iterating the output image tensor
    float scale = 1.0f / rscale;
    float support =
        (scale >= 1.0f) ? (p.support_size * 0.5f) * scale : p.support_size * 0.5f;

    int32_t window_size = SafeInt<int32_t>(ceilf(support)) * 2 + 1;
    const SafeInt<size_t> scale_buffer_size = sizeof(float) * (window_size)*output_size;

    const auto scale_data_buffer = alloc->Alloc(scale_buffer_size);
    weight_coefficients = BufferUniquePtr(scale_data_buffer, BufferDeleter(alloc));

    // Get pointers to appropriate memory locations in the scratch buffer
    auto* scale_data = static_cast<float*>(weight_coefficients.get());
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
      original_idx.emplace_back(center);
      float total_weight = 0.0;

      int64_t xmin_real = static_cast<int64_t>(center - support + 0.5);
      int64_t xmax_real = static_cast<int64_t>(center + support + 0.5);
      int64_t xmin_cut = std::max<int64_t>(xmin_real, (0));
      int64_t xmax_cut = std::min<int64_t>(xmax_real, input_size);

      xmin = exclude_outside ? xmin_cut : xmin_real;
      xmax = exclude_outside ? xmax_cut : xmax_real;
      bound_idx.push_back(xmin_cut);
      bound_idx.push_back(xmax_cut);

      auto* scale_buffer = &scale_data[i * window_size];
      int32_t* scale_buffer_int = reinterpret_cast<int32_t*>(scale_buffer);
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

      for (x = 0; x < xmax; x++) {
        if (total_weight != 0.0 && total_weight != 1) {
          scale_buffer[x] /= total_weight;
        }

        // normalize the scale to 1 << 22 for int8/uint8
        if (dtype == ONNX_NAMESPACE::TensorProto_DataType_INT8 || dtype == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
          if (scale_buffer[x] < 0) {
            scale_buffer_int[x] =
                (int)(-0.5 + scale_buffer[x] * (1 << 22));
          } else {
            scale_buffer_int[x] =
                (int)(0.5 + scale_buffer[x] * (1 << 22));
          }
        }
      }

      for (; x < window_size; x++) {
        scale_buffer[x] = 0;
      }
    }
    return window_size;
  };

  const size_t width_rindex = is_nchw ? 0 : 1;
  const size_t height_rindex = is_nchw ? 1 : 2;
  const size_t channel_rindex = is_nchw ? 2 : 3;

  /* Handles values form -640 to 639. */
  static uint8_t clip8_lookups_table[1280];
  p.clip8_lookups_table = clip8_lookups_table;

  p.init_clip_lookup();
  p.dim_y.window_size = compute_weight_coefficients(p, input_h_w_c[0], output_h_w_c[0], height_rindex,
                                                    p.dim_y.bound, p.dim_y.original, scale_h_w_c[0], p.dim_y.weight_coefficients);
  p.dim_x.window_size = compute_weight_coefficients(p, input_h_w_c[1], output_h_w_c[1], width_rindex,
                                                    p.dim_x.bound, p.dim_x.original, scale_h_w_c[1], p.dim_x.weight_coefficients);
  if (input_h_w_c.size() == 3) {
    p.dim_z.window_size = compute_weight_coefficients(p, input_h_w_c[2], output_h_w_c[2], channel_rindex,
                                                      p.dim_z.bound, p.dim_z.original, scale_h_w_c[2], p.dim_z.weight_coefficients);
  }
}

}  // namespace onnxruntime
