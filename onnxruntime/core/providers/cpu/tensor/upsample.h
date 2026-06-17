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
#if defined(_M_ARM64) || defined(__aarch64__) || defined(__ARM_NEON)
#if !defined(_MSC_VER)
#include <arm_neon.h>
#endif
#define ORT_UPSAMPLE_USE_NEON 1
#endif
#include <algorithm>
#include <type_traits>
#ifndef ORT_RESTRICT
#if defined(_MSC_VER)
#define ORT_RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define ORT_RESTRICT __restrict__
#else
#define ORT_RESTRICT
#endif
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

// Horizontal lerp into a float scratch row.
// Gather-indexed loads preclude NEON; scalar is optimal here (same conclusion as reference).
template <typename T>
inline void UpsampleBilinearHorizontalRow(
    const T* ORT_RESTRICT src_row,
    const int32_t* ORT_RESTRICT in_x1,
    const int32_t* ORT_RESTRICT in_x2,
    const float* ORT_RESTRICT dx1,
    const float* ORT_RESTRICT dx2,
    float* ORT_RESTRICT h_row,
    int32_t output_width) {
  for (int32_t x = 0; x < output_width; ++x) {
    h_row[x] = dx2[x] * static_cast<float>(src_row[in_x1[x]]) +
               dx1[x] * static_cast<float>(src_row[in_x2[x]]);
  }
}

// Vertical lerp from two horizontally-interpolated rows into the output row.
// T=float:   NEON float32x4, vmlaq_f32, 4 pixels/iter.
// T=uint8_t: NEON float32x4 -> vcvt -> vqmovun, 8 pixels/iter.
// T=int8_t:  NEON float32x4 -> vcvt -> vqmovn,  8 pixels/iter.
// Scalar fallback covers any remainder and non-NEON builds.
template <typename T>
inline void UpsampleBilinearVerticalRow(
    const float* ORT_RESTRICT h_top,
    const float* ORT_RESTRICT h_bot,
    T* ORT_RESTRICT out,
    int32_t width,
    float dy1, float dy2) {
  int32_t x = 0;
#if defined(ORT_UPSAMPLE_USE_NEON)
  if constexpr (std::is_same<T, float>::value) {
    // float: fused multiply-add, 4 pixels/iter.
    const float32x4_t vdy1 = vdupq_n_f32(dy1);
    float* fout = reinterpret_cast<float*>(out);
    for (; x <= width - 4; x += 4) {
      const float32x4_t vt = vld1q_f32(h_top + x);
      const float32x4_t vb = vld1q_f32(h_bot + x);
      vst1q_f32(fout + x, vmlaq_f32(vt, vsubq_f32(vb, vt), vdy1));
    }
  } else if constexpr (std::is_same<T, uint8_t>::value || std::is_same<T, int8_t>::value) {
    // uint8/int8: compute in float32, convert to int32, saturating-narrow to 8-bit.
    // vcvtq_s32_f32 truncates toward zero — same behaviour as static_cast<T>.
    // vqmovun_s16 clamps to [0,255]; vqmovn_s16 clamps to [-128,127].
    const float32x4_t vdy1 = vdupq_n_f32(dy1);
    const float32x4_t vdy2 = vdupq_n_f32(dy2);
    for (; x <= width - 8; x += 8) {
      const float32x4_t vt0 = vld1q_f32(h_top + x);
      const float32x4_t vt1 = vld1q_f32(h_top + x + 4);
      const float32x4_t vb0 = vld1q_f32(h_bot + x);
      const float32x4_t vb1 = vld1q_f32(h_bot + x + 4);
      // dy2*htop + dy1*hbot
      const float32x4_t r0 = vaddq_f32(vmulq_f32(vdy2, vt0), vmulq_f32(vdy1, vb0));
      const float32x4_t r1 = vaddq_f32(vmulq_f32(vdy2, vt1), vmulq_f32(vdy1, vb1));
      // float32 -> int32 (truncate) -> int16 (saturating narrow) -> int8/uint8
      const int16x8_t i16 = vcombine_s16(vqmovn_s32(vcvtq_s32_f32(r0)),
                                          vqmovn_s32(vcvtq_s32_f32(r1)));
      if constexpr (std::is_same<T, uint8_t>::value) {
        vst1_u8(reinterpret_cast<uint8_t*>(out + x), vqmovun_s16(i16));
      } else {
        vst1_s8(reinterpret_cast<int8_t*>(out + x), vqmovn_s16(i16));
      }
    }
  }
#endif
  for (; x < width; ++x) {
    out[x] = static_cast<T>(dy2 * h_top[x] + dy1 * h_bot[x]);
  }
}

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

          // Per-thread scratch buffers for horizontally-interpolated rows.
          // Two buffers; swapped when y2_base advances to avoid a redundant recompute.
          std::vector<float> h_buf0(output_width), h_buf1(output_width);
          float* h_top = h_buf0.data();
          float* h_bot = h_buf1.data();
          // -1 is a safe sentinel: input_width_mul_y* values are always >= 0.
          int32_t cached_y1_base = -1;
          int32_t cached_y2_base = -1;

          for (int32_t y = 0; y < output_height; ++y) {
            T* const out_row = Ydata + y * output_width;

            // Whole-row y extrapolation: skip separable computation entirely.
            if (use_extrapolation &&
                (p.y_original[y] < 0 || p.y_original[y] > static_cast<float>(input_height - 1))) {
              std::fill(out_row, out_row + output_width, static_cast<T>(extrapolation_value));
              continue;
            }

            const int32_t y1_base = p.input_width_mul_y1[y];
            const int32_t y2_base = p.input_width_mul_y2[y];
            const float dy1_y = p.dy1[y];
            const float dy2_y = p.dy2[y];

            // Row-pair caching mirrors the reference bilinearScaling swap logic:
            // when the new y1_base equals the previous y2_base, swap pointers
            // so the already-computed h_bot becomes the new h_top for free.
            if (y1_base == cached_y2_base && y2_base != cached_y2_base) {
              std::swap(h_top, h_bot);
              cached_y1_base = cached_y2_base;
              UpsampleBilinearHorizontalRow(Xdata + y2_base, p.in_x1, p.in_x2,
                                            p.dx1, p.dx2, h_bot, output_width);
              cached_y2_base = y2_base;
            } else {
              if (y1_base != cached_y1_base) {
                UpsampleBilinearHorizontalRow(Xdata + y1_base, p.in_x1, p.in_x2,
                                              p.dx1, p.dx2, h_top, output_width);
                cached_y1_base = y1_base;
              }
              if (y2_base != cached_y2_base) {
                UpsampleBilinearHorizontalRow(Xdata + y2_base, p.in_x1, p.in_x2,
                                              p.dx1, p.dx2, h_bot, output_width);
                cached_y2_base = y2_base;
              }
            }

            if (use_extrapolation) {
              // Per-pixel x boundary check; scalar since branchy.
              for (int32_t x = 0; x < output_width; ++x) {
                if (p.x_original[x] < 0 || p.x_original[x] > static_cast<float>(input_width - 1)) {
                  out_row[x] = static_cast<T>(extrapolation_value);
                } else {
                  out_row[x] = static_cast<T>(dy2_y * h_top[x] + dy1_y * h_bot[x]);
                }
              }
            } else {
              // NEON-accelerated vertical lerp (float32x4 for T=float, scalar otherwise).
              UpsampleBilinearVerticalRow(h_top, h_bot, out_row,
                                          output_width, dy1_y, dy2_y);
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

// 4-term bilinear lerp kernel for uint8 channels.
// Computes: Y[c] = (w11*X11[c] + w21*X21[c] + w12*X12[c] + w22*X22[c]) >> 20
// where w11+w21+w12+w22 == 2^20 and all weights >= 0.
// All values and weights are non-negative so >> 20 is identical to / (1<<20).
// NEON path processes 8 channels/iteration on ARM using the separable approximation
// (at most 1 ULP from the 4-term result); scalar path is bit-exact with original.
inline void BilinearLerpChannels(
    const uint8_t* ORT_RESTRICT top1,
    const uint8_t* ORT_RESTRICT top2,
    const uint8_t* ORT_RESTRICT bot1,
    const uint8_t* ORT_RESTRICT bot2,
    uint8_t* ORT_RESTRICT dst,
    int32_t num_channels,
    int32_t dx1_s10,  // weight toward x2, in [0, 1024]
    int32_t dy1_s10   // weight toward y2, in [0, 1024]
) {
  const int32_t dx2_s10 = 1024 - dx1_s10;
  const int32_t dy2_s10 = 1024 - dy1_s10;
  const int32_t w11 = dx2_s10 * dy2_s10;
  const int32_t w21 = dx1_s10 * dy2_s10;
  const int32_t w12 = dx2_s10 * dy1_s10;
  const int32_t w22 = dx1_s10 * dy1_s10;
  int32_t c = 0;
#if defined(ORT_UPSAMPLE_USE_NEON)
  // Separable approximation: htop = t1 + ((dx1*(t2-t1)) >> 10), then vertical.
  // At most 1 ULP from 4-term; NEON vmull_s16 path, 8 channels/iter.
  const int16x8_t vdx1 = vdupq_n_s16(static_cast<int16_t>(dx1_s10));
  const int16x8_t vdy1 = vdupq_n_s16(static_cast<int16_t>(dy1_s10));
  for (; c <= num_channels - 8; c += 8) {
    const int16x8_t vt1 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(top1 + c)));
    const int16x8_t vt2 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(top2 + c)));
    const int16x8_t vb1 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(bot1 + c)));
    const int16x8_t vb2 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(bot2 + c)));
    const int16x8_t dtop = vsubq_s16(vt2, vt1);
    const int16x8_t dbot = vsubq_s16(vb2, vb1);
    const int16x8_t htop = vaddq_s16(vt1, vcombine_s16(
        vshrn_n_s32(vmull_s16(vget_low_s16(dtop),  vget_low_s16(vdx1)),  10),
        vshrn_n_s32(vmull_s16(vget_high_s16(dtop), vget_high_s16(vdx1)), 10)));
    const int16x8_t hbot = vaddq_s16(vb1, vcombine_s16(
        vshrn_n_s32(vmull_s16(vget_low_s16(dbot),  vget_low_s16(vdx1)),  10),
        vshrn_n_s32(vmull_s16(vget_high_s16(dbot), vget_high_s16(vdx1)), 10)));
    const int16x8_t dvert = vsubq_s16(hbot, htop);
    const int16x8_t vout = vaddq_s16(htop, vcombine_s16(
        vshrn_n_s32(vmull_s16(vget_low_s16(dvert),  vget_low_s16(vdy1)),  10),
        vshrn_n_s32(vmull_s16(vget_high_s16(dvert), vget_high_s16(vdy1)), 10)));
    vst1_u8(dst + c, vqmovun_s16(vout));
  }
#endif
  // Scalar: 4-term formula, bit-exact with original.
  // uint8 inputs + non-negative weights → sum is non-negative → >> 20 == / (1<<20).
  for (; c < num_channels; ++c) {
    dst[c] = static_cast<uint8_t>(
        (w11 * static_cast<int32_t>(top1[c]) + w21 * static_cast<int32_t>(top2[c]) +
         w12 * static_cast<int32_t>(bot1[c]) + w22 * static_cast<int32_t>(bot2[c])) >> 20);
  }
}

// Specialisation for non-uint8 T: falls back to scaled-integer path (no SIMD).
template <typename T>
inline void BilinearLerpChannelsGeneric(
    const T* ORT_RESTRICT top1, const T* ORT_RESTRICT top2,
    const T* ORT_RESTRICT bot1, const T* ORT_RESTRICT bot2,
    T* ORT_RESTRICT dst,
    int32_t num_channels,
    int32_t dx1_s10, int32_t dy1_s10
) {
  // Reuse dx2/dy2 implicitly: dx2 = 1024-dx1, dy2 = 1024-dy1
  const int32_t dx2_s10 = 1024 - dx1_s10;
  const int32_t dy2_s10 = 1024 - dy1_s10;
  // Combine into 20-bit weights (scale 1024*1024 = 2^20)
  const int32_t w11 = dx2_s10 * dy2_s10;
  const int32_t w21 = dx1_s10 * dy2_s10;
  const int32_t w12 = dx2_s10 * dy1_s10;
  const int32_t w22 = dx1_s10 * dy1_s10;
  for (int32_t c = 0; c < num_channels; ++c) {
    if constexpr (std::is_integral<T>::value) {
      // Use division for signed types: >> rounds toward -inf but / rounds toward zero.
      // For uint8_t the result is always non-negative so both are equivalent,
      // but int8_t can produce negative intermediates.
      dst[c] = static_cast<T>((w11 * static_cast<int32_t>(top1[c]) +
                                w21 * static_cast<int32_t>(top2[c]) +
                                w12 * static_cast<int32_t>(bot1[c]) +
                                w22 * static_cast<int32_t>(bot2[c])) / (1 << 20));
    } else {
      constexpr float inv_scale = 1.0f / static_cast<float>(1 << 20);
      dst[c] = static_cast<T>((static_cast<float>(w11) * static_cast<float>(top1[c]) +
                                static_cast<float>(w21) * static_cast<float>(top2[c]) +
                                static_cast<float>(w12) * static_cast<float>(bot1[c]) +
                                static_cast<float>(w22) * static_cast<float>(bot2[c])) * inv_scale);
    }
  }
}

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

  // dx/dy _scale_10 store weights in [0, 1024] (2^10).  dx1+dx2=1024, dy1+dy2=1024.
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

            if constexpr (UseExtrapolation) {
              if ((p.y_original[y] < 0 || p.y_original[y] > static_cast<float>(input_height - 1)) ||
                  (p.x_original[x] < 0 || p.x_original[x] > static_cast<float>(input_width - 1))) {
                for (int32_t c = 0; c < num_channels; ++c) {
                  Ydata[output_offset + c] = static_cast<T>(extrapolation_value);
                }
                continue;
              }
            }

            const int32_t X11_offset = (p.input_width_mul_y1[y] + p.in_x1[x]) * num_channels;
            const int32_t X21_offset = (p.input_width_mul_y1[y] + p.in_x2[x]) * num_channels;
            const int32_t X12_offset = (p.input_width_mul_y2[y] + p.in_x1[x]) * num_channels;
            const int32_t X22_offset = (p.input_width_mul_y2[y] + p.in_x2[x]) * num_channels;

            if constexpr (std::is_same<T, uint8_t>::value) {
              // 4-term scalar / NEON separable path (uint8_t only).
              BilinearLerpChannels(
                  reinterpret_cast<const uint8_t*>(Xdata + X11_offset),
                  reinterpret_cast<const uint8_t*>(Xdata + X21_offset),
                  reinterpret_cast<const uint8_t*>(Xdata + X12_offset),
                  reinterpret_cast<const uint8_t*>(Xdata + X22_offset),
                  reinterpret_cast<uint8_t*>(Ydata + output_offset),
                  num_channels,
                  p.dx1_scale_10[x],
                  p.dy1_scale_10[y]);
            } else {
              BilinearLerpChannelsGeneric(
                  Xdata + X11_offset, Xdata + X21_offset,
                  Xdata + X12_offset, Xdata + X22_offset,
                  Ydata + output_offset,
                  num_channels,
                  p.dx1_scale_10[x], p.dy1_scale_10[y]);
            }
          }
        });
  }
}

}  // namespace onnxruntime
#if defined(_MSC_VER) && !defined(__clang__)
#pragma warning(pop)
#endif