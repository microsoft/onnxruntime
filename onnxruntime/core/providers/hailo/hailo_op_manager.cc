/**
 * Copyright (c) 2022 Hailo Technologies Ltd. All rights reserved.
 * Distributed under the MIT license (https://opensource.org/licenses/MIT)
 **/

#include "hailo_op_manager.h"
#include <iostream>

namespace onnxruntime {
HailoOpManager::HailoOpManager() {
    hailo_ops_map.emplace(std::make_pair("HailoOp", std::unique_ptr<HailoNodeCapability>(new HailoDefaultNodeCapability())));
}

bool HailoOpManager::IsNodeSupported(const Node& node, const GraphViewer& graph_viewer) const {
    auto it = hailo_ops_map.find(node.OpType());
    if (it == hailo_ops_map.end()) {
        return false;
    }
    return it->second->Supported(node, graph_viewer);
}

}  // namespace onnxruntime
