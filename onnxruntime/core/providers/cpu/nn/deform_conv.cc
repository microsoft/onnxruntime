// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//
// CPU implementation of DeformConv (deformable convolution 2D).
//
// High-level pipeline (one batch item at a time for peak memory):
//   (1) Build a bilinear sampling plan from offsets (parallel): for each (offset_group, kernel tap, output pixel),
//       store 4 neighbor indices + 4 weights in AoSoA blocks (see kPlanAoSoALanes).
//   (2) Fill im2col matrix (parallel over channels x kernel taps): each row is one (channel, i, j) slice,
//       reusing the plan row shared by all channels in the same offset group.
//   (3) Grouped GEMM: Y_g = W_g * Col_g per group (highly optimized in math::Gemm / BLAS).
//   (4) Optional bias: add B[m] to each output channel map (vectorized per row).
//
// Biggest win vs a naive loop: reusing the sampling plan across C/offset_group channels (plan built once per
// offset row, not per channel) plus AoSoA layout so the gather/interpolate inner loop can SIMD-unroll 8-wide.

#include "deform_conv.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/util/math_cpuonly.h"
#include "core/util/force_inline.h"
#include "core/util/math.h"

#if defined(__GNUC__) && !defined(__wasm__)
#define ORT_CPU_RESTRICT __restrict__
#elif defined(_MSC_VER)
#define ORT_CPU_RESTRICT __restrict
#else
#define ORT_CPU_RESTRICT
#endif

// Hint the inner lane loop for SIMD / vectorization (OpenMP simd, Clang loop, or GCC ivdep); empty otherwise.
#if defined(_OPENMP)
#define ORT_CPU_SIMD_INNER_LOOP _Pragma("omp simd")
#elif defined(__clang__)
#define ORT_CPU_SIMD_INNER_LOOP _Pragma("clang loop vectorize(enable)")
#elif defined(__GNUC__)
#define ORT_CPU_SIMD_INNER_LOOP _Pragma("GCC ivdep")
#else
#define ORT_CPU_SIMD_INNER_LOOP
#endif

namespace onnxruntime {

namespace {

// AoSoA "lane" count: each BilinearSamplePlanBlock holds 8 output pixels' worth of idx/weights per corner.
// For T=float, 8 matches one 256-bit AVX2 vector of floats; auto-vectorizers often turn the lane loop into
// SIMD. For T=double, SIMD is typically 4-wide; the 8-lane layout still unrolls the scalar work and keeps
// the same indexing (pidx / 8, pidx % 8) in PlanStoreSample — changing it requires revisiting all offsets.
constexpr int64_t kPlanAoSoALanes = 8;

// Overflow-safe size_t multiply: returns a * b only if the product fits in size_t.
// Guard: for a > 0, a * b <= max  <=>  b <= max / a (integer division; avoids computing a*b first).
// If a == 0 the product is 0 and is always representable, so the check is skipped.
// Used when deriving batch/group strides from tensor shapes so pointer arithmetic like
// base + n * stride cannot wrap size_t on valid ONNX shapes.
ORT_FORCEINLINE size_t CheckedMulSizeT(size_t a, size_t b, size_t max_size_t, const char* err) {
  ORT_ENFORCE(a == 0 || b <= max_size_t / a, err);
  return a * b;
}

// Verifies that batch indexing over n items with byte stride `stride` stays within addressable size_t range.
// For n > 0, the largest offset used when stepping batch index 0..n-1 is (n-1)*stride plus element spans;
// requiring n * stride <= max_size_t is a conservative upper bound that n * stride itself does not overflow
// (same inequality as CheckedMulSizeT with arguments n and stride).
ORT_FORCEINLINE void CheckedBatchSpan(size_t n, size_t stride, size_t max_size_t, const char* err) {
  ORT_ENFORCE(n == 0 || stride <= max_size_t / n, err);
}

struct CpuDeformConvStrides {
  size_t x_batch_stride = 0;
  size_t y_batch_stride = 0;
  size_t w_group_stride = 0;
  size_t col_group_stride = 0;
  size_t y_group_stride = 0;
  size_t offset_batch_stride = 0;
  size_t mask_batch_stride = 0;
};

struct CpuDeformConvExecutionDims {
  int64_t plan_rows = 0;
  int64_t padded_spatial_count = 0;
  size_t block_count = 0;
  int64_t im2col_rows = 0;
  int64_t col_buffer_size = 0;
};

inline CpuDeformConvExecutionDims ComputeCpuDeformConvExecutionDims(const DeformConvParams& params,
                                                                    const DeformConvCommonDims& common_dims,
                                                                    int64_t ptrdiff_max,
                                                                    size_t max_size_t) {
  const int64_t int64_max = std::numeric_limits<int64_t>::max();

  ORT_ENFORCE(params.offset_group <= int64_max / common_dims.kernel_size, "plan_rows overflows int64.");
  const int64_t plan_rows = params.offset_group * common_dims.kernel_size;

  ORT_ENFORCE(plan_rows > 0 && common_dims.output_image_size > 0, "Invalid plan dimensions.");
  ORT_ENFORCE(common_dims.output_image_size <= int64_max - (kPlanAoSoALanes - 1),
              "output_image_size is too large and will overflow.");
  // Round output_image_size up to a multiple of kPlanAoSoALanes so each plan "row" occupies an integer number
  // of BilinearSamplePlanBlocks. Storage is row-major: row r starts at offset r * padded_spatial_count in
  // "logical pixels"; block index = that offset / kPlanAoSoALanes. Only indices [0, output_image_size) are
  // read when filling im2col; the padded tail slots in the last block are never read (FillColRow uses output_size).
  // [IMPORTANT] Plan buffer is not zero-filled; tail lanes in the last block stay uninitialized (FillColRow uses tail_count only).
  const int64_t padded_spatial_count = (common_dims.output_image_size + kPlanAoSoALanes - 1) /
                                       kPlanAoSoALanes * kPlanAoSoALanes;
  const size_t blocks_per_row = static_cast<size_t>(padded_spatial_count) / kPlanAoSoALanes;
  ORT_ENFORCE(blocks_per_row <= (max_size_t / static_cast<size_t>(plan_rows)),
              "Sampling plan size overflows size_t.");
  const size_t block_count = static_cast<size_t>(plan_rows) * blocks_per_row;

  ORT_ENFORCE(plan_rows <= ptrdiff_max, "plan_rows exceeds ptrdiff_t range.");
  ORT_ENFORCE(common_dims.output_image_size == 0 || plan_rows <= int64_max / common_dims.output_image_size,
              "Flattened bilinear plan task count overflows int64.");
  const int64_t flattened_plan_tasks = plan_rows * common_dims.output_image_size;
  ORT_ENFORCE(flattened_plan_tasks <= ptrdiff_max,
              "Flattened bilinear plan tasks exceed ptrdiff_t range (needed for thread pool parallelization).");
  ORT_ENFORCE(params.C <= int64_max / common_dims.kernel_size, "im2col row count overflows int64.");
  const int64_t im2col_rows = params.C * common_dims.kernel_size;
  ORT_ENFORCE(im2col_rows <= ptrdiff_max, "im2col row count exceeds ptrdiff_t range.");
  ORT_ENFORCE(im2col_rows <= int64_max / common_dims.output_image_size, "col_buffer_size overflows int64.");
  const int64_t col_buffer_size = im2col_rows * common_dims.output_image_size;

  return CpuDeformConvExecutionDims{
      plan_rows,
      padded_spatial_count,
      block_count,
      im2col_rows,
      col_buffer_size};
}

inline CpuDeformConvStrides ComputeCpuDeformConvStrides(const DeformConvParams& params,
                                                        const DeformConvCommonDims& common_dims,
                                                        size_t max_size_t) {
  const size_t c_size = static_cast<size_t>(params.C);
  const size_t m_size = static_cast<size_t>(params.M);
  const size_t n_size = static_cast<size_t>(params.N);
  const size_t group_size = static_cast<size_t>(params.group);
  const size_t offset_group_size = static_cast<size_t>(params.offset_group);
  const size_t input_image_size_sz = static_cast<size_t>(common_dims.input_image_size);
  const size_t output_image_size_sz = static_cast<size_t>(common_dims.output_image_size);
  const size_t kernel_size_sz = static_cast<size_t>(common_dims.kernel_size);
  const size_t kernel_dim_sz = static_cast<size_t>(common_dims.kernel_dim);
  const size_t m_per_group_sz = m_size / group_size;

  // Flat strides (elements between batch items or group slices). Computed once per Compute(), not per pixel.
  // Hot paths then use pointer += stride instead of repeated rank-4/5 index math — typically saves several
  // multiplies/adds per inner iteration in exchange for this O(1) setup cost.
  CpuDeformConvStrides strides;
  strides.x_batch_stride = CheckedMulSizeT(c_size, input_image_size_sz, max_size_t, "X batch stride overflows size_t.");
  strides.y_batch_stride = CheckedMulSizeT(m_size, output_image_size_sz, max_size_t, "Y batch stride overflows size_t.");
  strides.w_group_stride = CheckedMulSizeT(m_per_group_sz, kernel_dim_sz, max_size_t, "weight group stride overflows size_t.");
  strides.col_group_stride = CheckedMulSizeT(kernel_dim_sz, output_image_size_sz, max_size_t, "col group stride overflows size_t.");
  strides.y_group_stride = CheckedMulSizeT(m_per_group_sz, output_image_size_sz, max_size_t, "Y group stride overflows size_t.");

  const size_t offset_rows = CheckedMulSizeT(
      CheckedMulSizeT(offset_group_size, kernel_size_sz, max_size_t, "offset rows overflows size_t."),
      static_cast<size_t>(2), max_size_t, "offset rows overflows size_t.");
  strides.offset_batch_stride = CheckedMulSizeT(offset_rows, output_image_size_sz, max_size_t, "offset batch stride overflows size_t.");

  const size_t mask_rows = CheckedMulSizeT(offset_group_size, kernel_size_sz, max_size_t, "mask rows overflows size_t.");
  strides.mask_batch_stride = CheckedMulSizeT(mask_rows, output_image_size_sz, max_size_t, "mask batch stride overflows size_t.");

  CheckedBatchSpan(n_size, strides.x_batch_stride, max_size_t, "X batch indexing overflows size_t.");
  CheckedBatchSpan(n_size, strides.y_batch_stride, max_size_t, "Y batch indexing overflows size_t.");
  CheckedBatchSpan(n_size, strides.offset_batch_stride, max_size_t, "offset batch indexing overflows size_t.");
  if (params.use_mask) {
    CheckedBatchSpan(n_size, strides.mask_batch_stride, max_size_t, "mask batch indexing overflows size_t.");
  }

  return strides;
}

namespace sampling_plan_internal {

// One AoSoA "macro-cell": 8 output pixels x 4 bilinear corners. See kPlanAoSoALanes and PlanStoreSample.
// [IMPORTANT] Last-block tail lanes may be uninitialized; keep in sync with padded_spatial_count / FillColRow tail_count.
template <typename T>
struct alignas(64) BilinearSamplePlanBlock {
  int32_t idx[4][kPlanAoSoALanes];
  T w[4][kPlanAoSoALanes];
};

template <typename T>
struct DeformableIm2colContext {
  const T* ORT_CPU_RESTRICT data_im = nullptr;
  const T* ORT_CPU_RESTRICT data_offset = nullptr;
  const T* ORT_CPU_RESTRICT data_mask = nullptr;
  int height = 0;
  int width = 0;
  int64_t kernel_h = 0;
  int64_t kernel_w = 0;
  int64_t stride_h = 0;
  int64_t stride_w = 0;
  int64_t channels = 0;
  int64_t offset_groups = 0;
  int64_t height_col = 0;
  int64_t width_col = 0;
  int64_t padded_spatial_count = 0;
  const size_t* ORT_CPU_RESTRICT kernel_offset_base_delta = nullptr;
  const T* ORT_CPU_RESTRICT kernel_base_h = nullptr;
  const T* ORT_CPU_RESTRICT kernel_base_w = nullptr;
  BilinearSamplePlanBlock<T>* ORT_CPU_RESTRICT sampling_plan_blocks = nullptr;
  T* ORT_CPU_RESTRICT data_col = nullptr;
  concurrency::ThreadPool* thread_pool = nullptr;
};

template <typename T>
ORT_FORCEINLINE void PlanStoreSample(BilinearSamplePlanBlock<T>* ORT_CPU_RESTRICT blocks, int64_t pidx,
                                     int32_t idx00, int32_t idx01, int32_t idx10, int32_t idx11,
                                     T w00, T w01, T w10, T w11) {
  // Scatter one output pixel into lane `pidx % 8` across the four corners. AoSoA vs AoS: here `w[k][0..7]`
  // are contiguous in memory for corner k, so the gather loop can load 8 weights per corner with vector
  // loads; with AoS (one pixel's 4 corners packed together), the same 8 pixels would be strided and harder
  // to SIMD. alignas(64) on the block type aligns starts to cache lines (struct may still span multiple lines).
  const int64_t block = pidx / kPlanAoSoALanes;
  const int64_t lane = pidx % kPlanAoSoALanes;
  auto& dst = blocks[block];
  dst.idx[0][lane] = idx00;
  dst.idx[1][lane] = idx01;
  dst.idx[2][lane] = idx10;
  dst.idx[3][lane] = idx11;
  dst.w[0][lane] = w00;
  dst.w[1][lane] = w01;
  dst.w[2][lane] = w10;
  dst.w[3][lane] = w11;
}

// Matches std::floor for in-range finite x. Call only after coords pass the inverted bounds check below
// (NaN makes each comparison false, so the && fails and ! rejects without a separate isfinite branch).
// Performance trick: std::floor can be slow due to handling edge cases (NaN, Inf, negative zero).
// This custom implementation uses a simple cast to int and a boolean subtraction, which compiles
// to fast, branchless instructions on most architectures.
template <typename T>
ORT_FORCEINLINE int DeformConvFastFloor(T x) {
  // Assumes x is in int range after prior bounds filtering; T→int truncates toward zero.
  const int i = static_cast<int>(x);
  return i - static_cast<int>(i > x);
}

template <typename T>
ORT_FORCEINLINE void BilinearPlanOneSample(
    const T* ORT_CPU_RESTRICT ptr_offset_h,
    const T* ORT_CPU_RESTRICT ptr_offset_w,
    int height,
    int width,
    int64_t h_col,
    int64_t w_col,
    int64_t local_idx,
    int64_t stride_h,
    int64_t stride_w,
    T base_h,
    T base_w,
    BilinearSamplePlanBlock<T>* ORT_CPU_RESTRICT plan_blocks) {
  // Deformable sampling point in input space (fractional): col (h_col,w_col) -> image (h_im, w_im).
  const T h_im = static_cast<T>(h_col * stride_h) + base_h + ptr_offset_h[local_idx];
  const T w_im = static_cast<T>(w_col * stride_w) + base_w + ptr_offset_w[local_idx];

  // In-bounds test on open rectangle (-1, H) x (-1, W) (same as strict && on comparisons). Bitwise & evaluates
  // all four preds (no short-circuit); NaN makes each compare false → treated as out-of-bounds without isnan().
  // One branch remains on `in_bounds == 0` to skip bilinear work when fully outside; inner fast/slow path is separate.
  const T neg1 = static_cast<T>(-1);
  const T h_max = static_cast<T>(height);
  const T w_max = static_cast<T>(width);
  const unsigned in_bounds = static_cast<unsigned>(
      (h_im > neg1) & (h_im < h_max) & (w_im > neg1) & (w_im < w_max));
  if (in_bounds == 0u) {
    PlanStoreSample(plan_blocks, local_idx, 0, 0, 0, 0,
                    static_cast<T>(0), static_cast<T>(0), static_cast<T>(0), static_cast<T>(0));
    return;
  }

  const int h_low = DeformConvFastFloor(h_im);
  const int w_low = DeformConvFastFloor(w_im);
  const T h_floor = static_cast<T>(h_low);
  const T w_floor = static_cast<T>(w_low);
  const int h_high = h_low + 1;
  const int w_high = w_low + 1;

  const T lh = h_im - h_floor;
  const T lw = w_im - w_floor;
  const T hh = static_cast<T>(1) - lh;
  const T hw = static_cast<T>(1) - lw;

  // Bilinear interpolation weights calculation:
  // w00 (top-left)     = (1 - dy) * (1 - dx)
  // w01 (top-right)    = (1 - dy) * dx
  // w10 (bottom-left)  = dy * (1 - dx)
  // w11 (bottom-right) = dy * dx
  T plan_w00 = hh * hw;
  T plan_w01 = hh * lw;
  T plan_w10 = lh * hw;
  T plan_w11 = lh * lw;

  // Safe under DeformConvValidateAndParse precondition: (H + 1) * W <= int_max.
  // With h_high <= H and w_high <= W, these linearized int indices stay in range.
  // Near borders h_low/w_low can be -1, but lower bounds also remain representable in int32.
  const int base_low = h_low * width;
  const int base_high = h_high * width;
  int32_t idx00 = base_low + w_low;
  int32_t idx01 = base_low + w_high;
  int32_t idx10 = base_high + w_low;
  int32_t idx11 = base_high + w_high;

  // Fast path: If the entire 2x2 interpolation window is strictly inside the image boundaries,
  // we can safely store the indices and weights without any further bounds checking.
  // This branch is taken for the vast majority of pixels, significantly speeding up the plan generation.
  if (static_cast<unsigned>(h_low) < static_cast<unsigned>(height - 1) &&
      static_cast<unsigned>(w_low) < static_cast<unsigned>(width - 1)) {
    PlanStoreSample(
        plan_blocks, local_idx,
        idx00, idx01, idx10, idx11,
        plan_w00, plan_w01, plan_w10, plan_w11);
  } else {
    // Slow path (Edge cases): The interpolation window overlaps with the image boundary.
    // We must check each of the 4 corners individually. If a corner is out of bounds,
    // its corresponding weight and index are forced to 0. This ensures that out-of-bounds
    // reads fetch a safe value (at index 0) which is then multiplied by a 0.0 weight,
    // effectively contributing 0 to the final interpolated result (zero-padding semantics).
    const bool v00 = (h_low >= 0 && w_low >= 0);
    const bool v01 = (h_low >= 0 && w_high < width);
    const bool v10 = (h_high < height && w_low >= 0);
    const bool v11 = (h_high < height && w_high < width);

    plan_w00 = v00 ? plan_w00 : static_cast<T>(0);
    plan_w01 = v01 ? plan_w01 : static_cast<T>(0);
    plan_w10 = v10 ? plan_w10 : static_cast<T>(0);
    plan_w11 = v11 ? plan_w11 : static_cast<T>(0);

    idx00 = v00 ? idx00 : 0;
    idx01 = v01 ? idx01 : 0;
    idx10 = v10 ? idx10 : 0;
    idx11 = v11 ? idx11 : 0;

    PlanStoreSample(
        plan_blocks, local_idx,
        idx00, idx01, idx10, idx11,
        plan_w00, plan_w01, plan_w10, plan_w11);
  }
}

template <typename T>
void BuildAllBilinearSamplingPlansImpl(
    const T* ORT_CPU_RESTRICT data_offset,
    int height,
    int width,
    int64_t stride_h,
    int64_t stride_w,
    int64_t offset_groups,
    size_t output_size,
    int64_t padded_spatial_count,
    int64_t width_col,
    int64_t kernel_size,
    const size_t* ORT_CPU_RESTRICT kernel_offset_base_delta,
    const T* ORT_CPU_RESTRICT kernel_base_h,
    const T* ORT_CPU_RESTRICT kernel_base_w,
    BilinearSamplePlanBlock<T>* ORT_CPU_RESTRICT sampling_plan_blocks,
    concurrency::ThreadPool* thread_pool) {
  const int64_t plan_rows = offset_groups * kernel_size;
  ORT_ENFORCE(kernel_offset_base_delta != nullptr, "kernel_offset_base_delta must not be null.");
  ORT_ENFORCE(kernel_base_h != nullptr, "kernel_base_h must not be null.");
  ORT_ENFORCE(kernel_base_w != nullptr, "kernel_base_w must not be null.");
  const size_t kernel_size_sz = static_cast<size_t>(kernel_size);
  const size_t offset_group_stride = static_cast<size_t>(2) * kernel_size_sz;
  const int64_t output_size_i64 = static_cast<int64_t>(output_size);

  // Plan is built once per (offset_group, kernel tap) row and reused for every input channel in that group:
  // work factor ~ O(offset_group * kH * kW * out_h * out_w) instead of O(C * kH * kW * ...) for bilinear setup.
  // Flatten (row, output pixel) to one task range so TryParallelFor can split fine-grained work even when
  // offset_group * kernel_size is small (parallelizing only the outer dimension would under-use threads).
  const int64_t total_plan_tasks = plan_rows * output_size_i64;
  // Unit cost is a dimensionless heuristic for ORT's thread pool splitter, not CPU cycles.
  // We keep plan-build chunking slightly finer than before so offset_group==1 cases can expose
  // enough parallel tasks early instead of leaving work concentrated in the later fill stage.
  constexpr double kCostPerBilinearSample = 8.0;
  concurrency::ThreadPool::TryParallelFor(
      thread_pool,
      static_cast<std::ptrdiff_t>(total_plan_tasks),
      kCostPerBilinearSample,
      [&](ptrdiff_t begin, ptrdiff_t end) {
        const int64_t end_task = static_cast<int64_t>(end);
        int64_t task = static_cast<int64_t>(begin);
        int64_t row = task / output_size_i64;
        int64_t local_idx = task % output_size_i64;
        int64_t offset_grp = row / kernel_size;
        int64_t kernel_idx = row % kernel_size;
        size_t offset_grp_base = static_cast<size_t>(offset_grp) * offset_group_stride;

        while (task < end_task) {
          const size_t kernel_idx_sz = static_cast<size_t>(kernel_idx);
          const size_t offset_base = offset_grp_base + kernel_offset_base_delta[kernel_idx_sz];
          const T* ORT_CPU_RESTRICT ptr_offset_h = data_offset + offset_base * output_size;
          const T* ORT_CPU_RESTRICT ptr_offset_w = data_offset + (offset_base + 1) * output_size;
          const size_t plan_row_base = static_cast<size_t>(row) * static_cast<size_t>(padded_spatial_count);
          BilinearSamplePlanBlock<T>* ORT_CPU_RESTRICT row_plan = sampling_plan_blocks + (plan_row_base / kPlanAoSoALanes);

          // Output pixel index: local_idx = h_col * width_col + w_col (row-major flatten of [0, out_h) x [0, out_w)).
          const int64_t h_col = local_idx / width_col;
          const int64_t w_col = local_idx % width_col;
          BilinearPlanOneSample(ptr_offset_h, ptr_offset_w, height, width, h_col, w_col, local_idx, stride_h,
                                stride_w, kernel_base_h[kernel_idx_sz], kernel_base_w[kernel_idx_sz], row_plan);

          ++task;
          if (++local_idx == output_size_i64) {
            local_idx = 0;
            ++row;
            if (++kernel_idx == kernel_size) {
              kernel_idx = 0;
              ++offset_grp;
              offset_grp_base += offset_group_stride;
            }
          }
        }
      });
}

template <typename T, bool UseMask>
void FillColRowFromSamplingPlanImpl(
    const T* ORT_CPU_RESTRICT im_ptr,
    const BilinearSamplePlanBlock<T>* ORT_CPU_RESTRICT plan_blocks,
    int64_t spatial_count,
    size_t mask_row_base,
    const T* ORT_CPU_RESTRICT ptr_mask,
    T* ORT_CPU_RESTRICT col_ptr) {
  // val = sum_{c in corners} w_c * im[idx_c]; optionally val *= mask[local_idx] (DeformConv v2).
  // UseMask is a template parameter so the no-mask build has zero mask branches/loads in this loop (better SIMD).
  const int64_t block_count = spatial_count / kPlanAoSoALanes;
  const int64_t tail_count = spatial_count % kPlanAoSoALanes;

  int64_t local_idx = 0;
  for (int64_t b = 0; b < block_count; ++b) {
    const auto& block = plan_blocks[b];
    // Inner lane loop: 8 pixels; for float, compilers often SIMD this (commonly a few× faster than scalar, ISA/optimizer dependent).
    ORT_CPU_SIMD_INNER_LOOP for (int lane = 0; lane < kPlanAoSoALanes; ++lane) {
      T val = block.w[0][lane] * im_ptr[block.idx[0][lane]] +
              block.w[1][lane] * im_ptr[block.idx[1][lane]] +
              block.w[2][lane] * im_ptr[block.idx[2][lane]] +
              block.w[3][lane] * im_ptr[block.idx[3][lane]];
      if constexpr (UseMask) {
        val *= ptr_mask[mask_row_base + local_idx];
      }
      col_ptr[local_idx] = val;
      ++local_idx;
    }
  }

  // [IMPORTANT] Last partial block: only lanes [0, tail_count) are valid; do not SIMD-load all 8 without init/zero.
  if (tail_count > 0) {
    const auto& block = plan_blocks[block_count];
    ORT_CPU_SIMD_INNER_LOOP for (int lane = 0; lane < tail_count; ++lane) {
      T val = block.w[0][lane] * im_ptr[block.idx[0][lane]] +
              block.w[1][lane] * im_ptr[block.idx[1][lane]] +
              block.w[2][lane] * im_ptr[block.idx[2][lane]] +
              block.w[3][lane] * im_ptr[block.idx[3][lane]];
      if constexpr (UseMask) {
        val *= ptr_mask[mask_row_base + local_idx];
      }
      col_ptr[local_idx] = val;
      ++local_idx;
    }
  }
}

template <typename T, bool UseMask>
void DeformableIm2colPlanned(const DeformableIm2colContext<T>& ctx) {
  ORT_ENFORCE(ctx.sampling_plan_blocks != nullptr, "sampling_plan_blocks must not be null.");
  ORT_ENFORCE(ctx.data_col != nullptr, "data_col must not be null.");
  // Single-image im2col: col buffer is [C*kH*kW, out_h*out_w] per batch index n only → peak memory O(C*kH*kW*HWout)
  // instead of O(N*...) when N>1. UseMask is compile-time so mask loads/branches are absent when false.
  const int64_t channel_per_offset_group = ctx.channels / ctx.offset_groups;
  const int64_t kernel_size = ctx.kernel_h * ctx.kernel_w;
  const int64_t output_size = ctx.height_col * ctx.width_col;
  const size_t output_size_sz = static_cast<size_t>(output_size);

  BuildAllBilinearSamplingPlansImpl(
      ctx.data_offset, ctx.height, ctx.width,
      ctx.stride_h, ctx.stride_w,
      ctx.offset_groups,
      output_size_sz, ctx.padded_spatial_count, ctx.width_col, kernel_size,
      ctx.kernel_offset_base_delta, ctx.kernel_base_h, ctx.kernel_base_w,
      ctx.sampling_plan_blocks, ctx.thread_pool);

  // Heuristic cost per im2col row (one channel x one kernel tap): ~one full pass over output pixels with gathers.
  // Slightly higher when UseMask (extra multiply per pixel). Same note as kCostPerBilinearSample: for scheduling only.
  // For small offset_group (especially 1), each sampling-plan row is reused by many channels, so this stage
  // dominates; reduce cost to encourage finer split and better load balance at high C.
  const double base_cost = static_cast<double>(output_size) * (UseMask ? 12.0 : 10.0);
  const double offset_group_adjust = (ctx.offset_groups == 1) ? 0.5 : 1.0;
  const double parallel_cost = base_cost * offset_group_adjust;
  concurrency::ThreadPool::TryParallelFor(
      ctx.thread_pool,
      static_cast<std::ptrdiff_t>(ctx.channels * kernel_size),
      parallel_cost,
      [&](ptrdiff_t begin, ptrdiff_t end) {
        int64_t c_im = begin / kernel_size;
        int64_t rem = begin % kernel_size;
        int64_t i = rem / ctx.kernel_w;
        int64_t j = rem % ctx.kernel_w;

        for (ptrdiff_t idx = begin; idx < end; ++idx) {
          const int64_t offset_grp = c_im / channel_per_offset_group;

          // Pointer arithmetic and index calculation for the current im2col row.
          // `col_ptr`: Points to the start of the current row in the output `col_buffer`.
          // Shape of col_buffer is [C * kH * kW, out_h * out_w].
          // Row-major flatten over (channel, kernel_y, kernel_x): idx = c_im * (kH*kW) + i * kW + j.
          T* ORT_CPU_RESTRICT col_ptr = ctx.data_col + static_cast<int64_t>(idx) * output_size;

          // `im_ptr`: Points to the start of the current channel `c_im` in the input image.
          // Shape of input image is [C, H, W].
          const T* ORT_CPU_RESTRICT im_ptr = ctx.data_im + c_im * static_cast<int64_t>(ctx.height) * ctx.width;

          // `row`: Identifies which pre-computed sampling plan to use.
          // The sampling plan is shared across channels that belong to the same `offset_grp`.
          // Formula: plan_row_index = offset_grp * (kH * kW) + (i * kW + j)
          const int64_t row = offset_grp * kernel_size + i * ctx.kernel_w + j;

          // `row_plan`: Points to the start of the AoSoA blocks for this specific `row`.
          // Since each block holds `kPlanAoSoALanes` elements, we divide the padded base index by it.
          const size_t plan_row_base = static_cast<size_t>(row) * static_cast<size_t>(ctx.padded_spatial_count);
          const BilinearSamplePlanBlock<T>* ORT_CPU_RESTRICT row_plan = ctx.sampling_plan_blocks + (plan_row_base / kPlanAoSoALanes);

          const T* ORT_CPU_RESTRICT ptr_mask = nullptr;
          size_t mask_row_base = 0;
          if constexpr (UseMask) {
            // If DeformConv v2 (with modulation mask), fetch the mask pointer.
            // Shape of mask is [offset_group * kH * kW, out_h * out_w].
            // The `row` index perfectly matches the mask's row index.
            ptr_mask = ctx.data_mask;
            mask_row_base = static_cast<size_t>(row) * output_size_sz;
          }

          // Execute the gather and interpolation for this specific (channel, kernel_y, kernel_x) combination
          // across all spatial output pixels [0, out_h * out_w).
          FillColRowFromSamplingPlanImpl<T, UseMask>(
              im_ptr, row_plan, output_size, mask_row_base, ptr_mask, col_ptr);

          if (++j == ctx.kernel_w) {
            j = 0;
            if (++i == ctx.kernel_h) {
              i = 0;
              ++c_im;
            }
          }
        }
      });
}

}  // namespace sampling_plan_internal

}  // namespace

template <typename T>
ORT_FORCEINLINE void DeformConvCpuAddBiasToRow(T* ORT_CPU_RESTRICT row, const T* ORT_CPU_RESTRICT bias_data,
                                               int64_t channel, ptrdiff_t spatial_len) {
  // row[s] += bias[channel] for all s; Eigen maps to SIMD on large spatial_len (often several x vs scalar loop).
  EigenVectorArrayMap<T>(row, spatial_len) += bias_data[channel];
}

template <typename T>
void DeformConvCpuAddBias(T* ORT_CPU_RESTRICT y_data, const T* ORT_CPU_RESTRICT bias_data, int64_t batch_n,
                          int64_t num_output_channels, int64_t output_image_size, size_t output_image_size_elements,
                          size_t y_batch_stride, concurrency::ThreadPool* thread_pool) {
  const int64_t int64_max = std::numeric_limits<int64_t>::max();
  const int64_t ptrdiff_max = static_cast<int64_t>(std::numeric_limits<std::ptrdiff_t>::max());
  const ptrdiff_t spatial_len = static_cast<ptrdiff_t>(output_image_size);
  const int64_t M = num_output_channels;

  // N==1: parallelize over M channels only. Avoids the N>1 path's initial k/M and k%M per thread chunk and keeps
  // the hot loop free of division (integer div is ~tens of cycles; negligible vs spatial SIMD work for large HW,
  // but this path is the common inference case and is simpler for the pool to split).
  // Y[0, m, :] += B[m] elementwise over spatial indices.
  if (batch_n == 1) {
    const double cost_per_channel_slice = static_cast<double>(output_image_size);
    concurrency::ThreadPool::TryParallelFor(
        thread_pool, static_cast<std::ptrdiff_t>(M), cost_per_channel_slice,
        [&](ptrdiff_t first, ptrdiff_t last) {
          for (ptrdiff_t m = first; m < last; ++m) {
            const size_t m_sz = static_cast<size_t>(m);
            T* ORT_CPU_RESTRICT y_row = y_data + m_sz * output_image_size_elements;
            DeformConvCpuAddBiasToRow<T>(y_row, bias_data, static_cast<int64_t>(m), spatial_len);
          }
        });
    return;
  }

  ORT_ENFORCE(batch_n <= int64_max / M, "N*M overflows int64 for bias parallelization.");

  // N>1: flatten (n, m) to k = n * M + m so TryParallelFor sees enough tasks; update (n,m) by increment/wrap
  // inside the loop to avoid div/mod per iteration (see loop body).
  const int64_t total_tasks = batch_n * M;
  ORT_ENFORCE(total_tasks <= ptrdiff_max, "N*M exceeds ptrdiff_t range for bias parallelization.");
  const double cost_per_task = static_cast<double>(output_image_size);

  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(total_tasks), cost_per_task,
      [&](ptrdiff_t first, ptrdiff_t last) {
        // Initialize (n,m) from `first` only once per [first,last); advance by m++ with wrap (no per-iter div).
        int64_t n = static_cast<int64_t>(first) / M;
        int64_t m = static_cast<int64_t>(first) % M;
        for (ptrdiff_t k = first; k < last; ++k) {
          const size_t n_sz = static_cast<size_t>(n);
          const size_t m_sz = static_cast<size_t>(m);

          // Pointer arithmetic formula: Y_row_ptr = y_data + (n * y_batch_stride) + (m * output_image_size)
          // Mathematical operation: Y[n, m, spatial_idx] += B[m] for all spatial_idx in [0, output_image_size).
          T* ORT_CPU_RESTRICT y_row = y_data + n_sz * y_batch_stride + m_sz * output_image_size_elements;
          DeformConvCpuAddBiasToRow<T>(y_row, bias_data, m, spatial_len);

          // For subsequent tasks, we simply increment `m` and wrap around to increment `n`.
          // This completely eliminates division and modulo operations inside the hot loop.
          if (++m == M) {
            m = 0;
            ++n;
          }
        }
      });
}

template <typename T>
Status DeformConv<T>::Compute(OpKernelContext* context) const {
  const auto* X = context->Input<Tensor>(0);
  const auto* W = context->Input<Tensor>(1);
  const auto* offset = context->Input<Tensor>(2);
  const auto* B = context->Input<Tensor>(3);     // optional
  const auto* mask = context->Input<Tensor>(4);  // optional

  DeformConvParams params;
  ORT_RETURN_IF_ERROR(DeformConvValidateAndParse(
      attrs_,
      X->Shape(),
      W->Shape(),
      offset->Shape(),
      B ? &B->Shape() : nullptr,
      mask ? &mask->Shape() : nullptr,
      params));

  const int64_t N = params.N;
  const int64_t C = params.C;
  const int64_t H = params.H;
  const int64_t W_in = params.W_in;
  const int64_t M = params.M;
  const int64_t kH = params.kH;
  const int64_t kW = params.kW;
  const int64_t stride_h = params.stride_h;
  const int64_t stride_w = params.stride_w;
  const int64_t group = params.group;
  const int64_t offset_group = params.offset_group;
  const int64_t out_h = params.out_h;
  const int64_t out_w = params.out_w;
  const bool use_mask = params.use_mask;

  // --- Phase 1: Pre-computation and Memory Allocation ---
  // 1.0) Output Y [N, M, out_h, out_w]; early exit if empty.
  const TensorShape Y_shape({N, M, out_h, out_w});
  Tensor* Y = context->Output(0, Y_shape);
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  // 1.1) Shared (CPU/CUDA) runtime bounds + derived dimensions.
  const int64_t ptrdiff_max = static_cast<int64_t>(std::numeric_limits<std::ptrdiff_t>::max());

  DeformConvCommonDims common_dims;
  ORT_RETURN_IF_ERROR(DeformConvValidateAndComputeCommonDims(params, common_dims));
  const int64_t output_image_size = common_dims.output_image_size;
  const int64_t kernel_dim = common_dims.kernel_dim;  // K dimension for GEMM: C/group * kH * kW
  const size_t max_size_t = std::numeric_limits<size_t>::max();
  const CpuDeformConvExecutionDims exec_dims = ComputeCpuDeformConvExecutionDims(params, common_dims, ptrdiff_max, max_size_t);
  const int64_t padded_spatial_count = exec_dims.padded_spatial_count;
  const size_t block_count = exec_dims.block_count;

  // Compute base sampling points and offset deltas on the fly using InlinedVector.
  // This avoids heap allocations (std::vector) while completely eliminating the need for
  // shared_mutex, atomic reference counting, and mutable state in the OpKernel.
  // The computation cost (a few dozen cycles) is vastly lower than lock/atomic overhead.
  const size_t kernel_size_sz = static_cast<size_t>(common_dims.kernel_size);
  // 49 is enough to inline up to 7x7 kernels without heap allocation.
  onnxruntime::InlinedVector<size_t, 49> offset_base_delta(kernel_size_sz);
  onnxruntime::InlinedVector<T, 49> base_h(kernel_size_sz);
  onnxruntime::InlinedVector<T, 49> base_w(kernel_size_sz);
  for (int64_t kernel_idx = 0; kernel_idx < common_dims.kernel_size; ++kernel_idx) {
    const int64_t i = kernel_idx / params.kW;
    const int64_t j = kernel_idx % params.kW;
    const size_t kernel_idx_sz = static_cast<size_t>(kernel_idx);
    // Offset tensor layout per ONNX DeformConv: for each offset_group and kernel tap, two maps (dy, dx)
    // of shape [out_h, out_w]. Flat row-major offset index for tap (i,j) is 2 * kernel_idx within the group.
    offset_base_delta[kernel_idx_sz] = static_cast<size_t>(2) * kernel_idx_sz;
    // Base sampling point in input space (before adding deform offsets): standard conv unwarped grid.
    base_h[kernel_idx_sz] = static_cast<T>(-params.pad_h + i * params.dilation_h);
    base_w[kernel_idx_sz] = static_cast<T>(-params.pad_w + j * params.dilation_w);
  }

  // Col buffer: shape [C*kH*kW, out_h*out_w]. Allocate per-image (process one image at a time)
  // to reduce peak memory when N is large; im2col is implemented per-image anyway.
  const int64_t col_buffer_size = exec_dims.col_buffer_size;

  // 1.2) Flat strides (element counts) for batch/group pointer bumping — see ComputeCpuDeformConvStrides body.
  // x_batch_stride = C * H * W; y_batch_stride = M * out_h * out_w; w_group_stride = (M/group) * kernel_dim;
  // col_group_stride = kernel_dim * out_h * out_w; y_group_stride = (M/group) * out_h * out_w;
  // offset_batch_stride = (2 * offset_group * kH * kW) * out_h * out_w (dy and dx maps per tap).
  // mask_batch_stride = (offset_group * kH * kW) * out_h * out_w (one modulation weight per tap, no factor 2).
  const CpuDeformConvStrides strides = ComputeCpuDeformConvStrides(params, common_dims, max_size_t);
  const size_t output_image_size_sz = static_cast<size_t>(output_image_size);
  const size_t x_batch_stride = strides.x_batch_stride;
  const size_t y_batch_stride = strides.y_batch_stride;
  const size_t w_group_stride = strides.w_group_stride;
  const size_t col_group_stride = strides.col_group_stride;
  const size_t y_group_stride = strides.y_group_stride;
  const size_t offset_batch_stride = strides.offset_batch_stride;
  const size_t mask_batch_stride = strides.mask_batch_stride;

  // 1.3) GEMM call-site bounds (checked once outside group loop).
  ORT_ENFORCE((M / group) <= ptrdiff_max, "GEMM M dimension exceeds ptrdiff_t range.");
  ORT_ENFORCE(output_image_size <= ptrdiff_max, "GEMM N dimension exceeds ptrdiff_t range.");
  ORT_ENFORCE(kernel_dim <= ptrdiff_max, "GEMM K dimension exceeds ptrdiff_t range.");

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
  auto col_buffer = IAllocator::MakeUniquePtr<T>(alloc, SafeInt<size_t>(col_buffer_size));

  ORT_ENFORCE(static_cast<uint64_t>(H) * static_cast<uint64_t>(W_in) <= static_cast<uint64_t>(std::numeric_limits<int>::max()),
              "DeformConv requires H*W to fit in int for sampling indices.");

  auto plan_blocks = IAllocator::MakeUniquePtr<sampling_plan_internal::BilinearSamplePlanBlock<T>>(alloc, SafeInt<size_t>(block_count));

  // Aliasing contract for this optimized path:
  // - input tensors may alias each other (read-only is fine),
  // - output Y must not overlap any input tensor (DeformConv is not an in-place kernel).
  const T* ORT_CPU_RESTRICT Xdata = X->Data<T>();
  const T* ORT_CPU_RESTRICT Wdata = W->Data<T>();
  const T* ORT_CPU_RESTRICT offset_data = offset->Data<T>();
  const T* ORT_CPU_RESTRICT mask_data = use_mask ? mask->Data<T>() : nullptr;
  T* ORT_CPU_RESTRICT Ydata = Y->MutableData<T>();
  const T* ORT_CPU_RESTRICT Bdata = (B != nullptr) ? B->Data<T>() : nullptr;

  // --- Phase 2: Core Computation (Im2Col + GEMM) ---
  // Process each image in the batch sequentially to save peak memory.
  concurrency::ThreadPool* thread_pool = context->GetOperatorThreadPool();
  for (int64_t n = 0; n < N; ++n) {
    const size_t n_idx = static_cast<size_t>(n);

    // 2.1) Deformable Im2Col for image n.
    // Gather deformed samples into col buffer for GEMM.
    const T* ORT_CPU_RESTRICT X_curr = Xdata + n_idx * x_batch_stride;
    const T* ORT_CPU_RESTRICT offset_curr = offset_data + n_idx * offset_batch_stride;
    const T* ORT_CPU_RESTRICT mask_curr = use_mask ? (mask_data + n_idx * mask_batch_stride) : nullptr;
    T* ORT_CPU_RESTRICT col_buffer_ptr = col_buffer.get();

    sampling_plan_internal::DeformableIm2colContext<T> im2col_ctx{
        X_curr, offset_curr, mask_curr,
        static_cast<int>(H), static_cast<int>(W_in),
        kH, kW, stride_h, stride_w, C, offset_group, out_h, out_w,
        padded_spatial_count,
        offset_base_delta.data(),
        base_h.data(),
        base_w.data(),
        plan_blocks.get(),
        col_buffer_ptr,
        thread_pool};
    // use_mask is runtime, but the hot gather loop is compiled twice (UseMask true/false) so the false
    // build has no mask load/multiply/branch per pixel — see FillColRowFromSamplingPlanImpl.
    if (use_mask) {
      sampling_plan_internal::DeformableIm2colPlanned<T, true>(im2col_ctx);
    } else {
      sampling_plan_internal::DeformableIm2colPlanned<T, false>(im2col_ctx);
    }

    // 2.2) GEMM for each group. Y = W * Col (per group).
    // The deformable convolution is cast as a Matrix Multiplication (GEMM).
    // For each group, the weight matrix W has shape [M/group, C/group * kH * kW]
    // and the gathered column matrix Col has shape [C/group * kH * kW, out_h * out_w].
    // The result Y_g is [M/group, out_h * out_w].
    for (int64_t g = 0; g < group; ++g) {
      const size_t g_idx = static_cast<size_t>(g);
      // Weight for group g: shape [M/group, C/group, kH, kW], row-major.
      const T* ORT_CPU_RESTRICT weight_g = Wdata + g_idx * w_group_stride;

      // Col rows for group g: layout [C*kH*kW, out_h*out_w], group g spans rows [g*kernel_dim, (g+1)*kernel_dim).
      const T* ORT_CPU_RESTRICT col_g = col_buffer_ptr + g_idx * col_group_stride;

      // Output slice for group g: [n, g*M/group:(g+1)*M/group, out_h, out_w].
      T* ORT_CPU_RESTRICT Y_g = Ydata + n_idx * y_batch_stride + g_idx * y_group_stride;

      // GEMM: C = alpha * A * B + beta * C with alpha=1, beta=0  =>  Y_g = W_g * Col_g.
      // Dimensions: A is (M_g, K), B is (K, N_out), C is (M_g, N_out), where M_g=M/group, K=kernel_dim, N_out=output_image_size.
      math::Gemm<T>(
          CblasNoTrans,
          CblasNoTrans,
          static_cast<ptrdiff_t>(M / group),          // M
          static_cast<ptrdiff_t>(output_image_size),  // N
          static_cast<ptrdiff_t>(kernel_dim),         // K
          static_cast<T>(1),                          // alpha
          weight_g,                                   // A
          col_g,                                      // B
          static_cast<T>(0),                          // beta
          Y_g,                                        // C
          thread_pool,
          nullptr);  // mlas_backend_kernel_selector_config
    }
  }

  // --- Phase 3: Post-processing ---
  // 3.1) Add bias if provided (broadcast over spatial dimensions).
  if (Bdata != nullptr) {
    DeformConvCpuAddBias<T>(Ydata, Bdata, N, M, output_image_size, output_image_size_sz, y_batch_stride, thread_pool);
  }

  return Status::OK();
}

// Explicit instantiation in this .cc keeps DeformConv<float/double> definitions out of other TUs that only
// include deform_conv.h — one copy of Compute() per T in the library, faster builds and predictable link size.
template class DeformConv<float>;
template class DeformConv<double>;

#define REGISTER_DEFORMCONV_KERNEL_TYPED(T)                       \
  ONNX_CPU_OPERATOR_VERSIONED_TYPED_KERNEL(                       \
      DeformConv, 19, 21, T,                                      \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DeformConv<T>);                                             \
  ONNX_CPU_OPERATOR_TYPED_KERNEL(                                 \
      DeformConv, 22, T,                                          \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      DeformConv<T>)

REGISTER_DEFORMCONV_KERNEL_TYPED(float)
REGISTER_DEFORMCONV_KERNEL_TYPED(double)

#undef REGISTER_DEFORMCONV_KERNEL_TYPED

}  // namespace onnxruntime
