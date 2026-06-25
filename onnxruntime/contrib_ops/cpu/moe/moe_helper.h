// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
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
                   const int64_t pack_size,                       // number of weights packed together (like 2 for uint4 packed to uint8)
                   const bool is_fused_swiglu,
                   const int64_t block_size = 0) {  // block size for block-wise quantization
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

  int64_t local_num_experts = fc1_experts_weights_shape->GetDims()[0];

  int64_t inter_size = (fc2_experts_weights_shape->GetDims()[1] *
                        fc2_experts_weights_shape->GetDims()[2] * pack_size) /
                       hidden_size;

  bool legacy_shape = false;
  const auto& fc2_experts_weights_dims = fc2_experts_weights_shape->GetDims();
  const auto& fc1_experts_weights_dims = fc1_experts_weights_shape->GetDims();
  legacy_shape = (hidden_size != inter_size && fc2_experts_weights_dims[1] == inter_size) ||
                 (hidden_size == inter_size && is_fused_swiglu && fc1_experts_weights_dims[1] == hidden_size);

  // Fused swiglu doubles the output dimension of FC1 since it fused two GEMMs into one.
  const int64_t fc1_inter_size = is_fused_swiglu ? (inter_size + inter_size) : inter_size;
  const int64_t zp_pack_size = pack_size;  // Zero points packing (1 for 8-bit, 2 for 4-bit)

  if (legacy_shape) {
    // legacy shape does not match column major memory layout. This is for backward compatibility.
    CHECK_SHAPE(fc1_experts_weights_shape, "fc1_experts_weights", num_experts, hidden_size, fc1_inter_size / pack_size);
    CHECK_SHAPE(fc2_experts_weights_shape, "fc2_experts_weights", num_experts, inter_size, hidden_size / pack_size);
    CHECK_SHAPE(fc3_experts_weights_shape, "fc3_experts_weights", num_experts, hidden_size, inter_size / pack_size);
  } else {
    CHECK_SHAPE(fc1_experts_weights_shape, "fc1_experts_weights", num_experts, fc1_inter_size, hidden_size / pack_size);
    CHECK_SHAPE(fc2_experts_weights_shape, "fc2_experts_weights", num_experts, hidden_size, inter_size / pack_size);
    CHECK_SHAPE(fc3_experts_weights_shape, "fc3_experts_weights", num_experts, inter_size, hidden_size / pack_size);
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
    if (fc1_scales_dims.size() == 3 && fc1_scales_dims[2] > 1) {
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
    const int64_t fc1_zp_blocks = (fc1_blocks_per_row + zp_pack_size - 1) / zp_pack_size;
    const int64_t fc2_zp_blocks = (fc2_blocks_per_row + zp_pack_size - 1) / zp_pack_size;
    const int64_t fc3_zp_blocks = (fc3_blocks_per_row + zp_pack_size - 1) / zp_pack_size;

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
        CHECK_TENSOR_SHAPE(fc1_zero_points, num_experts, (fc1_inter_size + zp_pack_size - 1) / zp_pack_size);
      } else if (fc1_scales_dims.size() == 3) {
        CHECK_TENSOR_SHAPE(fc1_experts_scales, num_experts, fc1_inter_size, 1);
        CHECK_TENSOR_SHAPE(fc1_zero_points, num_experts, (fc1_inter_size + zp_pack_size - 1) / zp_pack_size);
      } else {
        ORT_THROW("fc1_experts_scales must be 2D or 3D tensor");
      }
    }

    if (fc2_experts_scales != nullptr) {
      const auto& fc2_scales_dims = fc2_experts_scales->Shape().GetDims();
      if (fc2_scales_dims.size() == 2) {
        CHECK_TENSOR_SHAPE(fc2_experts_scales, num_experts, hidden_size);
        CHECK_TENSOR_SHAPE(fc2_zero_points, num_experts, (hidden_size + zp_pack_size - 1) / zp_pack_size);
      } else if (fc2_scales_dims.size() == 3) {
        CHECK_TENSOR_SHAPE(fc2_experts_scales, num_experts, hidden_size, 1);
        CHECK_TENSOR_SHAPE(fc2_zero_points, num_experts, (hidden_size + zp_pack_size - 1) / zp_pack_size);
      } else {
        ORT_THROW("fc2_experts_scales must be 2D or 3D tensor");
      }
    }

    if (fc3_experts_scales != nullptr) {
      const auto& fc3_scales_dims = fc3_experts_scales->Shape().GetDims();
      if (fc3_scales_dims.size() == 2) {
        CHECK_TENSOR_SHAPE(fc3_experts_scales, num_experts, inter_size);
        CHECK_TENSOR_SHAPE(fc3_zero_points, num_experts, (inter_size + zp_pack_size - 1) / zp_pack_size);
      } else if (fc3_scales_dims.size() == 3) {
        CHECK_TENSOR_SHAPE(fc3_experts_scales, num_experts, inter_size, 1);
        CHECK_TENSOR_SHAPE(fc3_zero_points, num_experts, (inter_size + zp_pack_size - 1) / zp_pack_size);
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
                   const int64_t pack_size,            // number of weights packed together (like 2 for uint4 packed to uint8)
                   const bool is_fused_swiglu,
                   const int64_t block_size = 0) {  // block size for block-wise quantization

  const TensorShape* fc1_shape = (fc1_experts_weights != nullptr) ? &fc1_experts_weights->Shape() : nullptr;
  const TensorShape* fc2_shape = (fc2_experts_weights != nullptr) ? &fc2_experts_weights->Shape() : nullptr;
  const TensorShape* fc3_shape = (fc3_experts_weights != nullptr) ? &fc3_experts_weights->Shape() : nullptr;

  return CheckInputs(parameters, input, router_probs, fc1_shape, fc1_experts_bias, fc1_experts_scales, fc1_zero_points,
                     fc2_shape, fc2_experts_bias, fc2_experts_scales, fc2_zero_points,
                     fc3_shape, fc3_experts_bias, fc3_experts_scales, fc3_zero_points,
                     pack_size, is_fused_swiglu, block_size);
}

}  // namespace moe_helper
}  // namespace contrib
}  // namespace onnxruntime
