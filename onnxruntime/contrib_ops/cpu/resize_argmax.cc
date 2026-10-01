// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#include "core/providers/cpu/tensor/upsample.h"

namespace onnxruntime {
namespace contrib {
namespace {

GetOriginalCoordinateFunc GetCoordinates(const std::string& mode) {
  ResizeCoordinateTransformationMode coordinates = HALF_PIXEL;
  if (mode == "align_corners")
    coordinates = ALIGN_CORNERS;
  else if (mode == "asymmetric")
    coordinates = ASYMMETRIC;
  else if (mode == "pytorch_half_pixel")
    coordinates = PYTORCH_HALF_PIXEL;
  else
    ORT_ENFORCE(mode == "half_pixel", "Unsupported coordinate transformation mode: ", mode);
  return UpsampleBase::GetOriginalCoordinateFromResizedCoordinate(coordinates);
}

struct ResizeArgMaxPlan {
  static constexpr int64_t tile_size = 256;
  int64_t channels, input_height, input_width, height, width, tiles_per_row;
  const BilinearParams& p;
  bool identity;

  static bool CheckSource(const float* source, const float* reference,
                          int64_t row0, int64_t row1, int64_t begin,
                          int64_t end) {
    for (int64_t iy : {row0, row1}) {
      for (int64_t ix = begin; ix < end; ++ix) {
        const float value = source[iy + ix];
        if (!std::isfinite(value) ||
            (reference && !(value <= reference[iy + ix])))
          return false;
      }
    }
    return true;
  }

  template <bool SelectLast>
  void RunTile(const float* input, int64_t* output, int64_t tile) const {
    const int64_t row = tile / tiles_per_row;
    const int64_t n = row / height;
    const int64_t oy = row % height;
    const int64_t start = tile % tiles_per_row * tile_size;
    const int64_t count = std::min(tile_size, width - start);
    const int64_t input_plane = input_height * input_width;
    const int64_t output_plane = height * width;
    const int64_t offset = oy * width + start;
    const int64_t row0 = p.input_width_mul_y1[oy];
    const int64_t row1 = p.input_width_mul_y2[oy];
    const int64_t begin = p.in_x1[start], end = p.in_x2[start + count - 1] + 1;
    // These input sizes keep the source indices exact in FP32. All weights
    // are then in [0, 1], so finite source values give finite products.
    bool prune = !identity && input_height <= (1 << 24) &&
                 input_width <= (1 << 24) && 2 * (end - begin) < count;
    int64_t budget = 4 * count;
    const float* reference = nullptr;
    float best[tile_size], w00[tile_size], w01[tile_size];
    float w10[tile_size], w11[tile_size];
    int32_t indices[tile_size] = {};
    for (int64_t j = 0; j < count; ++j) {
      const int64_t ox = start + j;
      w00[j] = p.dx2[ox] * p.dy2[oy];
      w01[j] = p.dx1[ox] * p.dy2[oy];
      w10[j] = p.dx2[ox] * p.dy1[oy];
      w11[j] = p.dx1[ox] * p.dy1[oy];
    }
    for (int64_t step = 0; step < channels; ++step) {
      const int64_t c =
          SelectLast && step > 0 ? channels - step : step;
      const float* source = input + (n * channels + c) * input_plane;
      // Limit unsuccessful checks to one tile's four source reads. Each skipped
      // class pays for more checks with the interpolation values it saves.
      prune &= budget >= 2 * (end - begin);
      if (prune && (!SelectLast || step > 0)) {
        budget -= 2 * (end - begin);
        const bool replace =
            !reference || source[row0 + begin] > reference[row0 + begin];
        // The reference class wins every tie. Nonnegative interpolation
        // weights preserve this order, including rounded ties.
        if (CheckSource(source, replace ? nullptr : reference, row0, row1,
                        begin, end)) {
          if (replace)
            reference = source;
          else {
            budget += count;
            continue;
          }
        }
      }
      float values[tile_size];
      if (identity) {
        // Resize copies the input when all dimensions are unchanged.
        std::copy_n(source + offset, count, values);
      } else {
        for (int64_t j = 0; j < count; ++j) {
          const int64_t ox = start + j;
          const int32_t x0 = p.in_x1[ox], x1 = p.in_x2[ox];
          // Keep the product and sum order from UpsampleBilinear.
          values[j] = w00[j] * source[row0 + x0] + w01[j] * source[row0 + x1] +
                      w10[j] * source[row1 + x0] + w11[j] * source[row1 + x1];
        }
      }
      if (c == 0) {
        std::copy_n(values, count, best);
        continue;
      }
      for (int64_t j = 0; j < count; ++j) {
        // Channel zero must initialize the result, including a possible NaN.
        const bool take = values[j] > best[j] ||
                          (SelectLast && values[j] == best[j] && indices[j] == 0);
        uint32_t next_bits, best_bits;
        std::memcpy(&next_bits, values + j, sizeof(next_bits));
        std::memcpy(&best_bits, best + j, sizeof(best_bits));
        const uint32_t mask = 0u - static_cast<uint32_t>(take);
        best_bits = (next_bits & mask) | (best_bits & ~mask);
        std::memcpy(best + j, &best_bits, sizeof(best_bits));
        indices[j] = (static_cast<uint32_t>(c) & mask) |
                     (static_cast<uint32_t>(indices[j]) & ~mask);
      }
    }
    std::copy_n(indices, count, output + n * output_plane + offset);
  }
};

}  // namespace

class ResizeArgMax final : public OpKernel {
 public:
  explicit ResizeArgMax(const OpKernelInfo& info)
      : OpKernel(info),
        coordinates_(GetCoordinates(info.GetAttrOrDefault<std::string>("coordinate_transformation_mode", "half_pixel"))),
        keepdims_(info.GetAttrOrDefault<int64_t>("keepdims", 1)),
        select_last_(info.GetAttrOrDefault<int64_t>("select_last_index", 0)) {
    ORT_ENFORCE((keepdims_ == 0 || keepdims_ == 1) && (select_last_ == 0 || select_last_ == 1),
                "keepdims and select_last_index must be 0 or 1");
  }

  Status Compute(OpKernelContext* context) const override {
    const auto* input = context->Input<Tensor>(0);
    const auto& shape = input->Shape();
    ORT_RETURN_IF_NOT(shape.NumDimensions() == 4, "ResizeArgMax requires NCHW input");
    constexpr int64_t limit = std::numeric_limits<int32_t>::max();
    for (auto dim : shape.GetDims()) {
      ORT_RETURN_IF_NOT(dim > 0 && dim <= limit, "Input dimensions must be positive and fit in int32");
    }
    ORT_RETURN_IF_NOT(shape[2] <= limit / 2 && shape[3] <= limit / 2 && shape[2] * shape[3] <= limit,
                      "Input plane is too large");
    const auto* scales = context->Input<Tensor>(1);
    const auto* sizes = context->Input<Tensor>(2);
    const bool has_scales = scales && scales->Shape().Size() != 0;
    const bool has_sizes = sizes && sizes->Shape().Size() != 0;
    ORT_RETURN_IF_NOT(has_scales != has_sizes, "Specify either scales or sizes");
    TensorShapeVector dims(4);
    float scale[4];
    if (has_scales) {
      ORT_RETURN_IF_NOT(scales->Shape().NumDimensions() == 1 && scales->Shape().Size() == 4,
                        "scales must have four values");
      for (int i = 0; i < 4; ++i) {
        scale[i] = scales->Data<float>()[i];
        const double length = std::floor(scale[i] * static_cast<float>(shape[i]));
        ORT_RETURN_IF_NOT(std::isfinite(scale[i]) && scale[i] > 0 && length >= 1 && length <= limit,
                          "Invalid output dimension");
        dims[i] = static_cast<int64_t>(length);
      }
      ORT_RETURN_IF_NOT(scale[0] == 1 && scale[1] == 1, "Batch and channel scales must be one");
    } else {
      ORT_RETURN_IF_NOT(sizes->Shape().NumDimensions() == 1 && sizes->Shape().Size() == 4,
                        "sizes must have four values");
      for (int i = 0; i < 4; ++i) {
        dims[i] = sizes->Data<int64_t>()[i];
        ORT_RETURN_IF_NOT(dims[i] > 0 && dims[i] <= limit, "Invalid output dimension");
        scale[i] = static_cast<float>(dims[i]) / static_cast<float>(shape[i]);
      }
    }
    ORT_RETURN_IF_NOT(dims[0] == shape[0] && dims[1] == shape[1], "Batch and channel sizes must not change");
    ORT_RETURN_IF_NOT(dims[2] + dims[3] <= limit / 2, "Output axes are too large");
    const int64_t height = dims[2], width = dims[3];
    if (keepdims_) {
      dims[1] = 1;
    } else {
      dims.erase(dims.begin() + 1);
    }
    auto* output = context->Output(0, dims);
    if (shape[1] == 1) {
      std::fill_n(output->MutableData<int64_t>(), output->Shape().Size(), int64_t{0});
      return Status::OK();
    }
    AllocatorPtr allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));
    const float roi[] = {0, 0, 0, 0, 1, 1, 1, 1};
    const auto p = SetupUpsampleBilinear(static_cast<int32_t>(shape[2]), static_cast<int32_t>(shape[3]),
                                         static_cast<int32_t>(height), static_cast<int32_t>(width),
                                         scale[2], scale[3], roi, allocator, coordinates_, true);
    const ResizeArgMaxPlan plan{shape[1], shape[2], shape[3], height, width,
                                (width + ResizeArgMaxPlan::tile_size - 1) / ResizeArgMaxPlan::tile_size,
                                p, shape[2] == height && shape[3] == width};
    concurrency::ThreadPool::TrySimpleParallelFor(
        context->GetOperatorThreadPool(), shape[0] * height * plan.tiles_per_row,
        [&](std::ptrdiff_t tile) {
          if (select_last_) {
            plan.RunTile<true>(input->Data<float>(), output->MutableData<int64_t>(), tile);
          } else {
            plan.RunTile<false>(input->Data<float>(), output->MutableData<int64_t>(), tile);
          }
        });
    return Status::OK();
  }

 private:
  GetOriginalCoordinateFunc coordinates_;
  int64_t keepdims_, select_last_;
};

ONNX_CPU_OPERATOR_MS_KERNEL(
    ResizeArgMax, 1,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    ResizeArgMax);

}  // namespace contrib
}  // namespace onnxruntime
