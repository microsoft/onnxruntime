// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <type_traits>
#include <unordered_map>

namespace onnxruntime {
namespace training {

enum class OptimizerOutputKey : int {
  GradientAccumulation,
  GradientAllIsFinite,
  GlobalGradientNorm,
  DeltaAllIsFinite,
};

struct OptimizerOutputKeyHash {
  size_t operator()(OptimizerOutputKey key) const {
    using underlying_type = std::underlying_type<OptimizerOutputKey>::type;
    return std::hash<underlying_type>{}(static_cast<underlying_type>(key));
  }
};

template <typename T>
using OptimizerOutputKeyMap =
    std::unordered_map<OptimizerOutputKey, T, OptimizerOutputKeyHash>;

}  // namespace training
}  // namespace onnxruntime
