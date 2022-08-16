// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/status.h"

namespace onnxruntime {
namespace nnapi {

class Shaper {
 public:
  using Shape = std::vector<uint32_t>;

  void AddShape(const std::string& name, const Shape& shape);
  inline const Shape& operator[](const std::string& key) const {
    return shape_map_.at(key);
  }
  // If the shape of certain input is dynamic
  // Use the following 2 functions to update the particular shape
  // and calculate the new output shape
  // Only perform this when the NNAPI model is finalized!
  common::Status UpdateShape(const std::string& name, const Shape& new_shape);
  common::Status UpdateDynamicDimensions();

  void Clear();

 private:
  std::unordered_map<std::string, Shape> shape_map_;
  std::vector<std::function<common::Status(Shaper&)>> shape_ops_;
};

}  // namespace nnapi
}  // namespace onnxruntime
