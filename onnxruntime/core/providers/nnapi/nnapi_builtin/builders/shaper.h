// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <string>
#include <unordered_map>

#include "core/common/inlined_containers.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"

namespace onnxruntime {
namespace nnapi {

class Shaper {
 public:
  using Shape = InlinedVector<uint32_t>;

  Shaper(const GraphViewer& graph_viewer) : graph_viewer_(&graph_viewer) {}

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

  // Note: Original code to update shapes are removed for now due to lack of dynamic shape support in NNAPI EP.
  // Can be added back and enhanced in the future if more support is available.

 private:
  std::unordered_map<std::string, Shape> shape_map_;
  const GraphViewer* graph_viewer_;
};

}  // namespace nnapi
}  // namespace onnxruntime
