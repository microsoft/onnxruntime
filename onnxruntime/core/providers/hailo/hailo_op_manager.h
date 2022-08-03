/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#pragma once
#include <map>
#include <string>
#include <memory>
#include "hailo_node_capability.h"

namespace onnxruntime {
class HailoOpManager {
 public:
    HailoOpManager();

    /**
    * This will check if the ORT node is Supported by the Hailo execution provider
    *
    * Several things will be checked from the node
    * - Is the OpType is regestered with the Hailo execution provider?
    * - Are the tensor dimensions Supported by the Hailo execution provider
    * - Are operator attributes Supported by the Hailo execution provider
    *
    * @param node the node that is being checked
    * 
    * @return true if the node is Supported by the Hailo execution provider
    *         false is returned otherwise.
    */
    bool IsNodeSupported(const Node& node, const GraphViewer& graph_viewer) const;

 private:
    std::map<std::string, std::unique_ptr<HailoNodeCapability>> hailo_ops_map;
};
}  // namespace onnxruntime