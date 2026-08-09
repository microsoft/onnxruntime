// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>
#ifndef SHARED_PROVIDER
#include "core/framework/op_kernel.h"
#endif
#include "core/providers/cpu/tensor/upsamplebase.h"
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(push)
// Chance of arithmetic overflow could be reduced
#pragma warning(disable : 26451)
#endif
namespace onnxruntime {

// In case of cubic mode the grid used to calculate the interpolation value
// is a 4x4 matrix
constexpr size_t CubicModeGridLength = 4;

struct BilinearParams {
  std::vector<float> x_original;
  std::vector<float> y_original;

  BufferUniquePtr idx_scale_data_buffer_holder;

  int32_t* input_width_mul_y1{nullptr};
  int32_t* input_width_mul_y2{nullptr};

  int32_t* in_x1{nullptr};
  int32_t* in_x2{nullptr};

  float* dx1{nullptr};
  float* dx2{nullptr};

  float* dy1{nullptr};
  float* dy2{nullptr};
};

// Same as above, but doesn't use any floating-point for the coefficient (i.e., d*_scale_10)
struct BilinearParamsInteger {
  std::vector<float> x_original;
  std::vector<float> y_original;

  BufferUniquePtr idx_scale_data_buffer_holder;

  int32_t* input_width_mul_y1{nullptr};
  int32_t* input_width_mul_y2{nullptr};

  int32_t* in_x1{nullptr};
  int32_t* in_x2{nullptr};

  int32_t* dx1_scale_10{nullptr};
  int32_t* dx2_scale_10{nullptr};

  int32_t* dy1_scale_10{nullptr};
  int32_t* dy2_scale_10{nullptr};
};

template <typename T>
class Upsample : public UpsampleBase, public OpKernel {
 public:
  Upsample(OpKernelInfo info) : UpsampleBase(info), OpKernel(info) {
  }

  Status Compute(OpKernelContext* context) const override;

  Status BaseCompute(OpKernelContext* context, gsl::span<const float> roi, gsl::span<const float> scales,
                     gsl::span<const int64_t> output_dims) const;
};

BilinearParams SetupUpsampleBilinear(const int32_t input_height,
                                     const int32_t input_width,
                                     const int32_t output_height,
                                     const int32_t output_width,
                                     const float height_scale,
                                     const float width_scale,
                                     gsl::span<const float> roi,
                                     AllocatorPtr& alloc,
                                     const GetOriginalCoordinateFunc& get_original_coordinate,
                                     const bool is_nchw);

template <typename T>
void UpsampleBilinear(const int32_t batch_size,
                      const int32_t num_channels,
                      const int32_t input_height,
                      const int32_t input_width,
                      const int32_t output_height,
                      const int32_t output_width,
                      const float height_scale,
                      const float width_scale,
                      gsl::span<const float> roi,
                      const bool use_extrapolation,
                      const float extrapolation_value,
                      const T* const XdataBase,
                      T* const YdataBase,
                      AllocatorPtr& alloc,
                      const GetOriginalCoordinateFunc& get_original_coordinate,
                      concurrency::ThreadPool* tp) {
  BilinearParams p = SetupUpsampleBilinear(input_height, input_width, output_height, output_width,
                                           height_scale, width_scale, roi,
                                           alloc, get_original_coordinate, true);
  for (int32_t n = 0; n < batch_size; ++n) {
    concurrency::ThreadPool::TrySimpleParallelFor(
        tp, num_channels,
        [&](std::ptrdiff_t c) {
          const T* const Xdata =
              XdataBase + (n * num_channels + static_cast<int32_t>(c)) * (input_height * input_width);
          T* const Ydata = YdataBase + (n * num_channels + static_cast<int32_t>(c)) * (output_height * output_width);
          for (int32_t y = 0; y < output_height; ++y) {
            for (int32_t x = 0; x < output_width; ++x) {
              const int32_t output_offset = output_width * y + x;
              // when use_extrapolation is set and original index of x or y is out of the dim range
              // then use extrapolation_value as the output value.
              if (use_extrapolation &&
                  ((p.y_original[y] < 0 || p.y_original[y] > static_cast<float>(input_height - 1)) ||
                   (p.x_original[x] < 0 || p.x_original[x] > static_cast<float>(input_width - 1)))) {
                Ydata[output_offset] = static_cast<T>(extrapolation_value);
                continue;
              }

              T X11 = Xdata[p.input_width_mul_y1[y] + p.in_x1[x]];
              T X21 = Xdata[p.input_width_mul_y1[y] + p.in_x2[x]];
              T X12 = Xdata[p.input_width_mul_y2[y] + p.in_x1[x]];
              T X22 = Xdata[p.input_width_mul_y2[y] + p.in_x2[x]];

              Ydata[output_offset] = static_cast<T>(p.dx2[x] * p.dy2[y] * X11 +
                                                    p.dx1[x] * p.dy2[y] * X21 +
                                                    p.dx2[x] * p.dy1[y] * X12 +
                                                    p.dx1[x] * p.dy1[y] * X22);
            }
          }
        });
  }
}

template <typename T, bool UseExtrapolation>
void NhwcUpsampleBilinear(const int32_t batch_size,
                          const int32_t num_channels,
                          const int32_t input_height,
                          const int32_t input_width,
                          const int32_t output_height,
                          const int32_t output_width,
                          const float height_scale,
                          const float width_scale,
                          gsl::span<const float> roi,
                          const float extrapolation_value,
                          const T* const XdataBase,
                          T* const YdataBase,
                          AllocatorPtr& alloc,
                          const GetOriginalCoordinateFunc& get_original_coordinate,
                          concurrency::ThreadPool* tp) {
  BilinearParams p = SetupUpsampleBilinear(input_height, input_width, output_height, output_width,
                                           height_scale, width_scale, roi,
                                           alloc, get_original_coordinate, false);
  for (int32_t n = 0; n < batch_size; ++n) {
    const T* const Xdata = XdataBase + n * (input_height * input_width) * num_channels;
    T* const Ydata = YdataBase + n * (output_height * output_width) * num_channels;
    concurrency::ThreadPool::TryParallelFor(
        tp, static_cast<std::ptrdiff_t>(output_height) * output_width,
        static_cast<double>(num_channels * 2),
        [&](std::ptrdiff_t first, std::ptrdiff_t last) {
          for (std::ptrdiff_t i = first; i < last; ++i) {
            const int32_t x = static_cast<int32_t>(i % output_width);
            const int32_t y = static_cast<int32_t>(i / output_width);
            const int32_t output_offset = (output_width * y + x) * num_channels;

            // when use_extrapolation is set and original index of x or y is out of the dim range
            // then use extrapolation_value as the output value.
            if constexpr (UseExtrapolation) {
              if ((p.y_original[y] < 0 || p.y_original[y] > static_cast<float>(input_height - 1)) ||
                  (p.x_original[x] < 0 || p.x_original[x] > static_cast<float>(input_width - 1))) {
                for (int32_t c = 0; c < num_channels; ++c) {
                  Ydata[output_offset + c] = static_cast<T>(extrapolation_value);
                }
              } else {
                const int32_t X11_offset = (p.input_width_mul_y1[y] + p.in_x1[x]) * num_channels;
                const int32_t X21_offset = (p.input_width_mul_y1[y] + p.in_x2[x]) * num_channels;
                const int32_t X12_offset = (p.input_width_mul_y2[y] + p.in_x1[x]) * num_channels;
                const int32_t X22_offset = (p.input_width_mul_y2[y] + p.in_x2[x]) * num_channels;
                const float X11_coef = p.dx2[x] * p.dy2[y];
                const float X21_coef = p.dx1[x] * p.dy2[y];
                const float X12_coef = p.dx2[x] * p.dy1[y];
                const float X22_coef = p.dx1[x] * p.dy1[y];
                for (int32_t c = 0; c < num_channels; ++c) {
                  const T X11 = Xdata[X11_offset + c];
                  const T X21 = Xdata[X21_offset + c];
                  const T X12 = Xdata[X12_offset + c];
                  const T X22 = Xdata[X22_offset + c];

                  Ydata[output_offset + c] = static_cast<T>(X11_coef * X11 +
                                                            X21_coef * X21 +
                                                            X12_coef * X12 +
                                                            X22_coef * X22);
                }
              }
            } else {
              const int32_t X11_offset = (p.input_width_mul_y1[y] + p.in_x1[x]) * num_channels;
              const int32_t X21_offset = (p.input_width_mul_y1[y] + p.in_x2[x]) * num_channels;
              const int32_t X12_offset = (p.input_width_mul_y2[y] + p.in_x1[x]) * num_channels;
              const int32_t X22_offset = (p.input_width_mul_y2[y] + p.in_x2[x]) * num_channels;
              const float X11_coef = p.dx2[x] * p.dy2[y];
              const float X21_coef = p.dx1[x] * p.dy2[y];
              const float X12_coef = p.dx2[x] * p.dy1[y];
              const float X22_coef = p.dx1[x] * p.dy1[y];
              for (int32_t c = 0; c < num_channels; ++c) {
                const T X11 = Xdata[X11_offset + c];
                const T X21 = Xdata[X21_offset + c];
                const T X12 = Xdata[X12_offset + c];
                const T X22 = Xdata[X22_offset + c];

                Ydata[output_offset + c] = static_cast<T>(X11_coef * X11 +
                                                          X21_coef * X21 +
                                                          X12_coef * X12 +
                                                          X22_coef * X22);
              }
            }
          }
        });
  }
}

BilinearParamsInteger SetupUpsampleBilinearInteger(const int32_t input_height,
                                                   const int32_t input_width,
                                                   const int32_t output_height,
                                                   const int32_t output_width,
                                                   const float height_scale,
                                                   const float width_scale,
                                                   gsl::span<const float> roi,
                                                   AllocatorPtr& alloc,
                                                   const GetOriginalCoordinateFunc& get_original_coordinate,
                                                   const bool is_nchw);

template <typename T, bool UseExtrapolation>
void NhwcUpsampleBilinearInteger(const int32_t batch_size,
                                 const int32_t num_channels,
                                 const int32_t input_height,
                                 const int32_t input_width,
                                 const int32_t output_height,
                                 const int32_t output_width,
                                 const float height_scale,
                                 const float width_scale,
                                 gsl::span<const float> roi,
                                 const float extrapolation_value,
                                 const T* const XdataBase,
                                 T* const YdataBase,
                                 AllocatorPtr& alloc,
                                 const GetOriginalCoordinateFunc& get_original_coordinate,
                                 concurrency::ThreadPool* tp) {
  BilinearParamsInteger p = SetupUpsampleBilinearInteger(input_height, input_width, output_height, output_width,
                                                         height_scale, width_scale, roi,
                                                         alloc, get_original_coordinate, false);
  for (int32_t n = 0; n < batch_size; ++n) {
    const T* const Xdata = XdataBase + n * (input_height * input_width) * num_channels;
    T* const Ydata = YdataBase + n * (output_height * output_width) * num_channels;
    concurrency::ThreadPool::TryParallelFor(
        tp, static_cast<std::ptrdiff_t>(output_height) * output_width,
        static_cast<double>(num_channels * 2),
        [&](std::ptrdiff_t first, std::ptrdiff_t last) {
          for (std::ptrdiff_t i = first; i < last; ++i) {
            const int32_t x = static_cast<int32_t>(i % output_width);
            const int32_t y = static_cast<int32_t>(i / output_width);
            const int32_t output_offset = (output_width * y + x) * num_channels;

            // when use_extrapolation is set and original index of x or y is out of the dim range
            // then use extrapolation_value as the output value.
            if constexpr (UseExtrapolation) {
              if ((p.y_original[y] < 0 || p.y_original[y] > static_cast<float>(input_height - 1)) ||
                  (p.x_original[x] < 0 || p.x_original[x] > static_cast<float>(input_width - 1))) {
                for (int32_t c = 0; c < num_channels; ++c) {
                  Ydata[output_offset + c] = static_cast<T>(extrapolation_value);
                }
              } else {
                const int32_t X11_offset = (p.input_width_mul_y1[y] + p.in_x1[x]) * num_channels;
                const int32_t X21_offset = (p.input_width_mul_y1[y] + p.in_x2[x]) * num_channels;
                const int32_t X12_offset = (p.input_width_mul_y2[y] + p.in_x1[x]) * num_channels;
                const int32_t X22_offset = (p.input_width_mul_y2[y] + p.in_x2[x]) * num_channels;
                const int32_t X11_coef_scale_20 = p.dx2_scale_10[x] * p.dy2_scale_10[y];
                const int32_t X21_coef_scale_20 = p.dx1_scale_10[x] * p.dy2_scale_10[y];
                const int32_t X12_coef_scale_20 = p.dx2_scale_10[x] * p.dy1_scale_10[y];
                const int32_t X22_coef_scale_20 = p.dx1_scale_10[x] * p.dy1_scale_10[y];
                for (int32_t c = 0; c < num_channels; ++c) {
                  const T X11 = Xdata[X11_offset + c];
                  const T X21 = Xdata[X21_offset + c];
                  const T X12 = Xdata[X12_offset + c];
                  const T X22 = Xdata[X22_offset + c];

                  Ydata[output_offset + c] = static_cast<T>((X11_coef_scale_20 * X11 +
                                                             X21_coef_scale_20 * X21 +
                                                             X12_coef_scale_20 * X12 +
                                                             X22_coef_scale_20 * X22) /
                                                            (1 << 20));
                }
              }
            } else {
              const int32_t X11_offset = (p.input_width_mul_y1[y] + p.in_x1[x]) * num_channels;
              const int32_t X21_offset = (p.input_width_mul_y1[y] + p.in_x2[x]) * num_channels;
              const int32_t X12_offset = (p.input_width_mul_y2[y] + p.in_x1[x]) * num_channels;
              const int32_t X22_offset = (p.input_width_mul_y2[y] + p.in_x2[x]) * num_channels;
              const int32_t X11_coef_scale_20 = p.dx2_scale_10[x] * p.dy2_scale_10[y];
              const int32_t X21_coef_scale_20 = p.dx1_scale_10[x] * p.dy2_scale_10[y];
              const int32_t X12_coef_scale_20 = p.dx2_scale_10[x] * p.dy1_scale_10[y];
              const int32_t X22_coef_scale_20 = p.dx1_scale_10[x] * p.dy1_scale_10[y];
              for (int32_t c = 0; c < num_channels; ++c) {
                const T X11 = Xdata[X11_offset + c];
                const T X21 = Xdata[X21_offset + c];
                const T X12 = Xdata[X12_offset + c];
                const T X22 = Xdata[X22_offset + c];

                Ydata[output_offset + c] = static_cast<T>((X11_coef_scale_20 * X11 +
                                                           X21_coef_scale_20 * X21 +
                                                           X12_coef_scale_20 * X12 +
                                                           X22_coef_scale_20 * X22) /
                                                          (1 << 20));
              }
            }
          }
        });
  }
}

}  // namespace onnxruntime
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif
