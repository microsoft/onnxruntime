// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/common/inlined_containers.h"
#include "core/common/status.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"

namespace onnxruntime {
namespace nnapi {

class Shaper {
 public:
  using Shape = InlinedVector<uint32_t>;

  Shaper(const GraphViewer& graph_viewer);

  void AddShape(const std::string& name, const Shape& shape) {
    shape_map_[name] = shape;
  }

  inline Shape operator[](const std::string& key) const {
    auto it = shape_map_.find(key);
    if (it != shape_map_.end()) {
      return it->second;
    }
    const auto shape = GetShapeInfoFromNodeArg(*graph_viewer_, key);
    return shape;
  }

  // If the shape of certain input is dynamic
  // Use the following 2 functions to update the particular shape
  // and calculate the new output shape
  // Only perform this when the NNAPI model is finalized!
  // TODO: Commented out these update shape methods for now due to the lack of dynamic shape support
  // in NNAPI EP. Can be added back and enhanced for future use if more support is available.
  /*
  common::Status UpdateShape(const std::string& name, const Shape& new_shape);
  common::Status UpdateDynamicDimensions();
  */

 private:
  std::unordered_map<std::string, Shape> shape_map_;
  /*
  std::vector<std::function<common::Status(Shaper&)>> shape_ops_;
   */

  const GraphViewer* graph_viewer_;
};

}  // namespace nnapi
}  // namespace onnxruntime
