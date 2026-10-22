// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <limits>
#include <type_traits>

#include <gsl/span>

#include "core/common/common.h"
#include "core/common/float16.h"
#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/framework/tensor_shape.h"
#include "core/util/shape_checker.h"

namespace onnxruntime {
namespace contrib {

enum class MoEParallelType {
  None = 0,
  EP = 1,
  TP = 2,
  EPAndTP = 3,
};

struct MoEParameters {
  MoEParameters() = default;

  explicit MoEParameters(int64_t tensor_shards)
      : tensor_shards(tensor_shards) {}

  int64_t num_rows{0};
  int64_t num_experts{0};
  int64_t local_num_experts{0};
  int64_t hidden_size{0};
  int64_t inter_size{0};

  MoEParallelType parallel_type{MoEParallelType::None};
  int64_t tensor_shards{1};
};
namespace moe_helper {

struct MoEWeightBits {
  int64_t fc1;
  int64_t fc2;
  int64_t fc3;
};

inline int64_t PackedByteCount(int64_t logical_element_count, int64_t bits) {
  const int64_t bit_count = SafeInt<int64_t>(logical_element_count) * bits;
  ORT_ENFORCE(bit_count % 8 == 0, "Packed weights must be byte-aligned.");
  return bit_count / 8;
}

inline int64_t PackedByteCountWithPadding(int64_t logical_element_count, int64_t bits) {
  ORT_ENFORCE(logical_element_count >= 0, "Logical element count must be nonnegative.");
  ORT_ENFORCE(bits >= 1 && bits <= 8, "Packed code bits must be between 1 and 8.");
  return SafeInt<int64_t>(logical_element_count / 8) * bits +
         ((logical_element_count % 8) * bits + 7) / 8;
}

template <typename ScaleT>
Status ValidateInt3ProjectionContent(gsl::span<const uint8_t> weights,
                                     gsl::span<const ScaleT> scales,
                                     gsl::span<const uint8_t> zero_points,
                                     bool has_zero_points,
                                     gsl::span<const int64_t> logical_shape,
                                     int64_t block_size) {
  static_assert(std::is_same_v<ScaleT, MLFloat16> || std::is_same_v<ScaleT, BFloat16>);
  ORT_RETURN_IF_NOT(logical_shape.size() == 3, "INT3 logical projection shape must have rank 3 [E,N,K].");
  ORT_RETURN_IF_NOT(block_size == 32 || block_size == 64 || block_size == 128,
                    "INT3 block_size must be 32, 64, or 128.");
  for (int64_t dimension : logical_shape) {
    ORT_RETURN_IF_NOT(dimension > 0, "INT3 logical dimensions must be positive.");
  }
  const int64_t experts = logical_shape[0];
  const int64_t outputs = logical_shape[1];
  const int64_t logical_count = logical_shape[2];
  ORT_RETURN_IF_NOT(logical_count % block_size == 0,
                    "INT3 K must be block aligned.");
  const int64_t blocks = logical_count / block_size;
  const int64_t weight_bytes = PackedByteCountWithPadding(logical_count, 3);
  const int64_t zp_bytes = PackedByteCountWithPadding(blocks, 3);
  const uint64_t max_count = std::numeric_limits<int64_t>::max();
  const uint64_t max_size = std::numeric_limits<size_t>::max();
  ORT_RETURN_IF_NOT(static_cast<uint64_t>(experts) <= max_count / static_cast<uint64_t>(outputs),
                    "INT3 row count overflow.");
  const uint64_t rows = static_cast<uint64_t>(experts) * static_cast<uint64_t>(outputs);
  for (int64_t stride : {weight_bytes, blocks, zp_bytes}) {
    const uint64_t unsigned_stride = static_cast<uint64_t>(stride);
    ORT_RETURN_IF_NOT(rows <= max_count / unsigned_stride && rows <= max_size / unsigned_stride,
                      "INT3 buffer count overflow.");
  }
  const size_t row_count = static_cast<size_t>(rows);
  const size_t weight_stride = static_cast<size_t>(weight_bytes);
  const size_t scale_stride = static_cast<size_t>(blocks);
  const size_t zp_stride = static_cast<size_t>(zp_bytes);
  ORT_RETURN_IF_NOT(weights.size() == row_count * weight_stride, "INT3 weight byte count mismatch.");
  ORT_RETURN_IF_NOT(scales.size() == row_count * scale_stride, "INT3 scale count mismatch.");
  ORT_RETURN_IF_NOT(has_zero_points ? zero_points.size() == row_count * zp_stride : zero_points.empty(),
                    "INT3 zero-point presence or byte count mismatch.");
  for (const auto& scale : scales) {
    const float value = scale.ToFloat();
    ORT_RETURN_IF_NOT(std::isfinite(value) && value >= 0, "INT3 scales must be finite and nonnegative.");
  }
  if (has_zero_points) {
    for (size_t byte = 0; byte < zp_stride; ++byte) {
      uint8_t expected = 0;
      for (size_t bit = 0; bit < 8; ++bit) {
        const size_t position = byte * 8 + bit;
        if (position / 3 < scale_stride && position % 3 == 2) {
          expected |= static_cast<uint8_t>(1u << bit);
        }
      }
      for (size_t row = 0; row < row_count; ++row) {
        ORT_RETURN_IF_NOT(zero_points[row * zp_stride + byte] == expected,
                          "INT3 zero points must all be 4 with zero padding.");
      }
    }
  }
  return Status::OK();
}

inline MoEWeightBits WeightBitsFromPackSize(int64_t pack_size) {
  ORT_ENFORCE(pack_size > 0 && 8 % pack_size == 0,
              "pack_size must be a positive divisor of 8, got ", pack_size, ".");
  const int64_t bits = 8 / pack_size;
  return MoEWeightBits{bits, bits, bits};
}

template <typename Tensor>
Status CheckFp8BlockScale(const Tensor* scale, int64_t num_experts, int64_t rows, int64_t cols,
                          int64_t block_size, const char* name) {
  ORT_RETURN_IF_NOT(block_size > 0, "FP8 block scaling requires a positive block_size.");
  ORT_RETURN_IF_NOT(scale != nullptr, "QMoE block-scaled FP8 requires ", name, ".");
  ORT_RETURN_IF_NOT(scale->template IsDataType<float>() || scale->template IsDataType<MLFloat16>() ||
                        scale->template IsDataType<BFloat16>(),
                    name, " must be a float32, float16, or bfloat16 tensor for block-scaled FP8.");
  const auto expected_shape = make_shape(num_experts,
                                         rows / block_size + (rows % block_size != 0),
                                         cols / block_size + (cols % block_size != 0));
  ORT_RETURN_IF_NOT(scale->Shape() == expected_shape,
                    name, " is expected to have shape ", expected_shape, ", got ", scale->Shape(), ".");
  return Status::OK();
}

// Helper to check shape dimensions
#define ASSERT_SHAPE_DIMENSION(shape_ptr, dim, name)                             \
  if (shape_ptr != nullptr) {                                                    \
    if (shape_ptr->NumDimensions() != dim) {                                     \
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input '", name,     \
                             "' is expected to have ", dim, " dimensions, got ", \
                             shape_ptr->NumDimensions());                        \
    }                                                                            \
  }

#define ASSERT_SHAPE_3D(shape_ptr, name) ASSERT_SHAPE_DIMENSION(shape_ptr, 3, name)

#define CHECK_SHAPE(shape_ptr, name, ...)                                    \
  if (shape_ptr != nullptr) {                                                \
    const TensorShape& expected_shape = make_shape(__VA_ARGS__);             \
    if (*shape_ptr != expected_shape) {                                      \
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input '", name, \
                             "' is expected to have shape ", expected_shape, \
                             ", got ", *shape_ptr);                          \
    }                                                                        \
  }

template <typename Tensor>
Status CheckInputs(MoEParameters& parameters,
                   const Tensor* input,                           // required
                   const Tensor* router_probs,                    // required
                   const TensorShape* fc1_experts_weights_shape,  // required
                   const Tensor* fc1_experts_bias,                // optional
                   const Tensor* fc1_experts_scales,              // required for qMoE; NULL for MOE
                   const Tensor* fc1_zero_points,                 // optional, for qMoE
                   const TensorShape* fc2_experts_weights_shape,  // required
                   const Tensor* fc2_experts_bias,                // optional
                   const Tensor* fc2_experts_scales,              // required for qMoE; NULL for MOE
                   const Tensor* fc2_zero_points,                 // optional, for qMoE
                   const TensorShape* fc3_experts_weights_shape,  // optional
                   const Tensor* fc3_experts_bias,                // optional
                   const Tensor* fc3_experts_scales,              // required for qMoE; NULL for MOE
                   const Tensor* fc3_zero_points,                 // optional, for qMoE
                   const MoEWeightBits& weight_bits,
                   const bool is_fused_swiglu,
                   const int64_t block_size = 0) {  // block size for block-wise quantization
  ORT_RETURN_IF(weight_bits.fc1 <= 0 || weight_bits.fc1 > 8 ||
                    weight_bits.fc2 <= 0 || weight_bits.fc2 > 8 ||
                    weight_bits.fc3 <= 0 || weight_bits.fc3 > 8,
                "FC weight bits must be between 1 and 8, got FC1=", weight_bits.fc1,
                ", FC2=", weight_bits.fc2, ", FC3=", weight_bits.fc3, ".");

  // Required inputs
  if (input == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'input' is required.");
  }
  ASSERT_TENSOR_2D_OR_3D(input);

  if (router_probs == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'router_probs' is required.");
  }
  ASSERT_TENSOR_2D(router_probs);

  if (fc1_experts_weights_shape == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'fc1_experts_weights' is required.");
  }
  ASSERT_SHAPE_3D(fc1_experts_weights_shape, "fc1_experts_weights");

  if (fc2_experts_weights_shape == nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input 'fc2_experts_weights' is required.");
  }
  ASSERT_SHAPE_3D(fc2_experts_weights_shape, "fc2_experts_weights");

  const auto& input_dims = input->Shape().GetDims();
  const auto& router_probs_dims = router_probs->Shape().GetDims();

  int64_t num_rows = input_dims.size() == 2 ? input_dims[0] : input_dims[0] * input_dims[1];
  int64_t hidden_size = input_dims[input_dims.size() - 1];
  int64_t num_experts = router_probs_dims[1];

  const bool has_uniform_weight_bits = weight_bits.fc1 == weight_bits.fc2 && weight_bits.fc2 == weight_bits.fc3;
  const bool has_legacy_pack_size = has_uniform_weight_bits && 8 % weight_bits.fc1 == 0;
  const int64_t legacy_pack_size = has_legacy_pack_size ? 8 / weight_bits.fc1 : 0;
  ORT_RETURN_IF(has_legacy_pack_size && hidden_size % legacy_pack_size != 0,
                "hidden_size (", hidden_size, ") must be divisible by pack_size (", legacy_pack_size, ").");

  int64_t local_num_experts = fc1_experts_weights_shape->GetDims()[0];

  const int64_t inter_size_numerator = SafeInt<int64_t>(fc2_experts_weights_shape->GetDims()[1]) *
                                       fc2_experts_weights_shape->GetDims()[2] * 8;
  const int64_t inter_size_denominator = SafeInt<int64_t>(hidden_size) * weight_bits.fc2;
  ORT_RETURN_IF(inter_size_numerator % inter_size_denominator != 0,
                "Unable to infer inter_size from fc2_experts_weights shape ",
                *fc2_experts_weights_shape, " and hidden_size ", hidden_size, ".");
  int64_t inter_size = inter_size_numerator / inter_size_denominator;
  ORT_RETURN_IF(has_legacy_pack_size && inter_size % legacy_pack_size != 0,
                "inter_size (", inter_size, ") must be divisible by pack_size (", legacy_pack_size, ").");
  bool legacy_shape = false;
  const auto& fc2_experts_weights_dims = fc2_experts_weights_shape->GetDims();
  const auto& fc1_experts_weights_dims = fc1_experts_weights_shape->GetDims();
  legacy_shape = (hidden_size != inter_size && fc2_experts_weights_dims[1] == inter_size) ||
                 (hidden_size == inter_size && is_fused_swiglu && fc1_experts_weights_dims[1] == hidden_size);

  // Fused swiglu doubles the output dimension of FC1 since it fused two GEMMs into one.
  const int64_t fc1_inter_size = is_fused_swiglu ? (inter_size + inter_size) : inter_size;

  if (legacy_shape) {
    // legacy shape does not match column major memory layout. This is for backward compatibility.
    ORT_RETURN_IF((SafeInt<int64_t>(fc1_inter_size) * weight_bits.fc1) % 8 != 0 ||
                      (SafeInt<int64_t>(hidden_size) * weight_bits.fc2) % 8 != 0 ||
                      (SafeInt<int64_t>(inter_size) * weight_bits.fc3) % 8 != 0,
                  "Expert dimensions and FC weight bits must produce byte-aligned weights for the legacy layout.");
    CHECK_SHAPE(fc1_experts_weights_shape, "fc1_experts_weights", num_experts, hidden_size,
                PackedByteCount(fc1_inter_size, weight_bits.fc1));
    CHECK_SHAPE(fc2_experts_weights_shape, "fc2_experts_weights", num_experts, inter_size,
                PackedByteCount(hidden_size, weight_bits.fc2));
    CHECK_SHAPE(fc3_experts_weights_shape, "fc3_experts_weights", num_experts, hidden_size,
                PackedByteCount(inter_size, weight_bits.fc3));
  } else {
    ORT_RETURN_IF((SafeInt<int64_t>(hidden_size) * weight_bits.fc1) % 8 != 0 ||
                      (SafeInt<int64_t>(inter_size) * weight_bits.fc2) % 8 != 0 ||
                      (SafeInt<int64_t>(hidden_size) * weight_bits.fc3) % 8 != 0,
                  "Expert dimensions and FC weight bits must produce byte-aligned weights.");
    CHECK_SHAPE(fc1_experts_weights_shape, "fc1_experts_weights", num_experts, fc1_inter_size,
                PackedByteCount(hidden_size, weight_bits.fc1));
    CHECK_SHAPE(fc2_experts_weights_shape, "fc2_experts_weights", num_experts, hidden_size,
                PackedByteCount(inter_size, weight_bits.fc2));
    CHECK_SHAPE(fc3_experts_weights_shape, "fc3_experts_weights", num_experts, inter_size,
                PackedByteCount(hidden_size, weight_bits.fc3));
  }

  CHECK_TENSOR_SHAPE(router_probs, num_rows, num_experts);

  CHECK_TENSOR_SHAPE(fc1_experts_bias, num_experts, fc1_inter_size);
  CHECK_TENSOR_SHAPE(fc2_experts_bias, num_experts, hidden_size);
  CHECK_TENSOR_SHAPE(fc3_experts_bias, num_experts, inter_size);

  // Validate scale tensors: Handle both row-wise and block-wise quantization flexibly
  // First, detect the actual quantization method from the tensor shapes
  bool is_row_wise_quantization = true;
  if (fc1_experts_scales != nullptr) {
    const auto& fc1_scales_dims = fc1_experts_scales->Shape().GetDims();
    if (fc1_scales_dims.size() == 3) {
      is_row_wise_quantization = false;
    }
  }

  if (block_size > 0 && !is_row_wise_quantization) {
    // Block-wise quantization: 3D scale tensors
    // For block-wise quantization, we calculate the number of blocks using ceiling division
    // to handle cases where the dimension is not perfectly divisible by block_size
    const int64_t fc1_blocks_per_row = (hidden_size + block_size - 1) / block_size;
    const int64_t fc2_blocks_per_row = (inter_size + block_size - 1) / block_size;
    const int64_t fc3_blocks_per_row = (hidden_size + block_size - 1) / block_size;

    CHECK_TENSOR_SHAPE(fc1_experts_scales, num_experts, fc1_inter_size, fc1_blocks_per_row);
    CHECK_TENSOR_SHAPE(fc2_experts_scales, num_experts, hidden_size, fc2_blocks_per_row);
    CHECK_TENSOR_SHAPE(fc3_experts_scales, num_experts, inter_size, fc3_blocks_per_row);

    // Validate zero-point tensors (block-wise)
    const int64_t fc1_zp_blocks = PackedByteCountWithPadding(fc1_blocks_per_row, weight_bits.fc1);
    const int64_t fc2_zp_blocks = PackedByteCountWithPadding(fc2_blocks_per_row, weight_bits.fc2);
    const int64_t fc3_zp_blocks = PackedByteCountWithPadding(fc3_blocks_per_row, weight_bits.fc3);

    CHECK_TENSOR_SHAPE(fc1_zero_points, num_experts, fc1_inter_size, fc1_zp_blocks);
    CHECK_TENSOR_SHAPE(fc2_zero_points, num_experts, hidden_size, fc2_zp_blocks);
    CHECK_TENSOR_SHAPE(fc3_zero_points, num_experts, inter_size, fc3_zp_blocks);
  } else {
    // Row-wise quantization: 2D scale tensors or 3D with last dimension = 1
    // Handle both {num_experts, features} and {num_experts, features, 1} shapes
    if (fc1_experts_scales != nullptr) {
      const auto& fc1_scales_dims = fc1_experts_scales->Shape().GetDims();
      if (fc1_scales_dims.size() == 2) {
        CHECK_TENSOR_SHAPE(fc1_experts_scales, num_experts, fc1_inter_size);
        CHECK_TENSOR_SHAPE(fc1_zero_points, num_experts,
                           PackedByteCountWithPadding(fc1_inter_size, weight_bits.fc1));
      } else if (fc1_scales_dims.size() == 3) {
        CHECK_TENSOR_SHAPE(fc1_experts_scales, num_experts, fc1_inter_size, 1);
        CHECK_TENSOR_SHAPE(fc1_zero_points, num_experts,
                           PackedByteCountWithPadding(fc1_inter_size, weight_bits.fc1));
      } else {
        ORT_THROW("fc1_experts_scales must be 2D or 3D tensor");
      }
    }

    if (fc2_experts_scales != nullptr) {
      const auto& fc2_scales_dims = fc2_experts_scales->Shape().GetDims();
      if (fc2_scales_dims.size() == 2) {
        CHECK_TENSOR_SHAPE(fc2_experts_scales, num_experts, hidden_size);
        CHECK_TENSOR_SHAPE(fc2_zero_points, num_experts,
                           PackedByteCountWithPadding(hidden_size, weight_bits.fc2));
      } else if (fc2_scales_dims.size() == 3) {
        CHECK_TENSOR_SHAPE(fc2_experts_scales, num_experts, hidden_size, 1);
        CHECK_TENSOR_SHAPE(fc2_zero_points, num_experts,
                           PackedByteCountWithPadding(hidden_size, weight_bits.fc2));
      } else {
        ORT_THROW("fc2_experts_scales must be 2D or 3D tensor");
      }
    }

    if (fc3_experts_scales != nullptr) {
      const auto& fc3_scales_dims = fc3_experts_scales->Shape().GetDims();
      if (fc3_scales_dims.size() == 2) {
        CHECK_TENSOR_SHAPE(fc3_experts_scales, num_experts, inter_size);
        CHECK_TENSOR_SHAPE(fc3_zero_points, num_experts,
                           PackedByteCountWithPadding(inter_size, weight_bits.fc3));
      } else if (fc3_scales_dims.size() == 3) {
        CHECK_TENSOR_SHAPE(fc3_experts_scales, num_experts, inter_size, 1);
        CHECK_TENSOR_SHAPE(fc3_zero_points, num_experts,
                           PackedByteCountWithPadding(inter_size, weight_bits.fc3));
      } else {
        ORT_THROW("fc3_experts_scales must be 2D or 3D tensor");
      }
    }
  }

  if (fc3_experts_weights_shape == nullptr) {
    // If fc3 weights are not provided, ensure no other fc3 parameters are provided
    ORT_ENFORCE(fc3_experts_bias == nullptr && fc3_experts_scales == nullptr && fc3_zero_points == nullptr);
  } else {
    // If fc3 weights are provided, ensure scales logic is consistent
    ORT_ENFORCE(fc1_experts_scales == nullptr || fc3_experts_scales != nullptr);  // MOE no scale, or qMOE need scales
  }

  parameters.num_rows = num_rows;
  parameters.num_experts = num_experts;
  parameters.local_num_experts = local_num_experts;
  parameters.hidden_size = hidden_size;
  parameters.inter_size = inter_size;
  if (num_experts == local_num_experts) {
    if (parameters.tensor_shards == 1) {
      parameters.parallel_type = MoEParallelType::None;
    } else {
      parameters.parallel_type = MoEParallelType::TP;
    }
  } else if (num_experts > local_num_experts) {
    if (parameters.tensor_shards == 1) {
      parameters.parallel_type = MoEParallelType::EP;
    } else {
      parameters.parallel_type = MoEParallelType::EPAndTP;
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "num_experts must be greater than or equal to local_num_experts, got ", num_experts,
                           " and ", local_num_experts);
  }

  return Status::OK();
}

template <typename Tensor>
Status CheckInputs(MoEParameters& parameters,
                   const Tensor* input, const Tensor* router_probs,
                   const TensorShape* fc1_experts_weights_shape, const Tensor* fc1_experts_bias,
                   const Tensor* fc1_experts_scales, const Tensor* fc1_zero_points,
                   const TensorShape* fc2_experts_weights_shape, const Tensor* fc2_experts_bias,
                   const Tensor* fc2_experts_scales, const Tensor* fc2_zero_points,
                   const TensorShape* fc3_experts_weights_shape, const Tensor* fc3_experts_bias,
                   const Tensor* fc3_experts_scales, const Tensor* fc3_zero_points,
                   const int64_t pack_size, const bool is_fused_swiglu, const int64_t block_size = 0) {
  return CheckInputs(parameters, input, router_probs,
                     fc1_experts_weights_shape, fc1_experts_bias, fc1_experts_scales, fc1_zero_points,
                     fc2_experts_weights_shape, fc2_experts_bias, fc2_experts_scales, fc2_zero_points,
                     fc3_experts_weights_shape, fc3_experts_bias, fc3_experts_scales, fc3_zero_points,
                     WeightBitsFromPackSize(pack_size), is_fused_swiglu, block_size);
}

template <typename Tensor>
Status CheckInputs(MoEParameters& parameters,
                   const Tensor* input,                // required
                   const Tensor* router_probs,         // required
                   const Tensor* fc1_experts_weights,  // required
                   const Tensor* fc1_experts_bias,     // optional
                   const Tensor* fc1_experts_scales,   // required for qMoE; NULL for MOE
                   const Tensor* fc1_zero_points,      // optional, for qMoE
                   const Tensor* fc2_experts_weights,  // required
                   const Tensor* fc2_experts_bias,     // optional
                   const Tensor* fc2_experts_scales,   // required for qMoE; NULL for MOE
                   const Tensor* fc2_zero_points,      // optional, for qMoE
                   const Tensor* fc3_experts_weights,  // optional
                   const Tensor* fc3_experts_bias,     // optional
                   const Tensor* fc3_experts_scales,   // required for qMoE; NULL for MOE
                   const Tensor* fc3_zero_points,      // optional, for qMoE
                   const MoEWeightBits& weight_bits,
                   const bool is_fused_swiglu,
                   const int64_t block_size = 0) {  // block size for block-wise quantization

  const TensorShape* fc1_shape = (fc1_experts_weights != nullptr) ? &fc1_experts_weights->Shape() : nullptr;
  const TensorShape* fc2_shape = (fc2_experts_weights != nullptr) ? &fc2_experts_weights->Shape() : nullptr;
  const TensorShape* fc3_shape = (fc3_experts_weights != nullptr) ? &fc3_experts_weights->Shape() : nullptr;

  return CheckInputs(parameters, input, router_probs, fc1_shape, fc1_experts_bias, fc1_experts_scales, fc1_zero_points,
                     fc2_shape, fc2_experts_bias, fc2_experts_scales, fc2_zero_points,
                     fc3_shape, fc3_experts_bias, fc3_experts_scales, fc3_zero_points,
                     weight_bits, is_fused_swiglu, block_size);
}

template <typename Tensor>
Status CheckInputs(MoEParameters& parameters,
                   const Tensor* input, const Tensor* router_probs,
                   const Tensor* fc1_experts_weights, const Tensor* fc1_experts_bias,
                   const Tensor* fc1_experts_scales, const Tensor* fc1_zero_points,
                   const Tensor* fc2_experts_weights, const Tensor* fc2_experts_bias,
                   const Tensor* fc2_experts_scales, const Tensor* fc2_zero_points,
                   const Tensor* fc3_experts_weights, const Tensor* fc3_experts_bias,
                   const Tensor* fc3_experts_scales, const Tensor* fc3_zero_points,
                   const int64_t pack_size, const bool is_fused_swiglu, const int64_t block_size = 0) {
  return CheckInputs(parameters, input, router_probs,
                     fc1_experts_weights, fc1_experts_bias, fc1_experts_scales, fc1_zero_points,
                     fc2_experts_weights, fc2_experts_bias, fc2_experts_scales, fc2_zero_points,
                     fc3_experts_weights, fc3_experts_bias, fc3_experts_scales, fc3_zero_points,
                     WeightBitsFromPackSize(pack_size), is_fused_swiglu, block_size);
}

}  // namespace moe_helper
}  // namespace contrib
}  // namespace onnxruntime
