// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include <iostream>

#include "gsl/gsl"

#include "core/common/status.h"
#include "core/common/safeint.h"
#include "helper.h"

namespace onnxruntime {
namespace nnapi {

class Shaper {
 public:
  using Shape = std::vector<uint32_t>;

  Shaper(const GraphViewer& graph_viewer);

  void AddShape(const std::string& name, const Shape& shape);

  inline const Shape& operator[](const std::string& key) {
    if (shape_map_.find(key) != shape_map_.end()) {
      return shape_map_.at(key);
    }
    const auto& shape = GetShapeInfoFromNodeArg(*graph_viewer_, key);
    shape_map_.insert(std::make_pair(key, shape));
    return shape_map_.at(key);
  }

  // If the shape of certain input is dynamic
  // Use the following 2 functions to update the particular shape
  // and calculate the new output shape
  // Only perform this when the NNAPI model is finalized!
  common::Status UpdateShape(const std::string& name, const Shape& new_shape);
  common::Status UpdateDynamicDimensions();

 private:
  std::unordered_map<std::string, Shape> shape_map_;
  std::vector<std::function<common::Status(Shaper&)>> shape_ops_;

  gsl::not_null<const GraphViewer*> graph_viewer_;
};

}  // namespace nnapi
}  // namespace onnxruntime
