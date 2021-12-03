#pragma once

#include "cstdint"

/**
 * @brief Sometime the usage is needed for perform layout optimization for
 * better inference performance. So that we need to annotate the node_arg in
 * graph and Tensors at runtime with is usage
 *
 */
enum class TensorUsage : uint8_t {
  Generic = 0,
  // Weight = 1,
  ConvWeight = 2,
  DepthwiseConvWeight = 3,
  // LstmWeight = 4, // TODO
  // LstmBias = 5, // TODO
};
