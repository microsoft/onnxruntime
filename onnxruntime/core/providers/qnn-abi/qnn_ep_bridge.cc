// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include "qnn_ep_bridge.h"

// QNN-ABI: Shared library implementation with full IsOpSupported validation
// This is designed exclusively for shared library builds and avoids
// internal header conflicts by using only OpBuilder existence checking

#include "builder/op_builder_factory.h"
#include "core/session/onnxruntime_c_api.h"

#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <vector>

namespace onnxruntime {
namespace qnn {

std::vector<const OrtNode*> QnnEpBridge::ValidateNodesWithIsOpSupported(
    const OrtGraph* graph,
    const OrtApi& ort_api,
    const OrtLogger* logger,
    const OrtEpGraphSupportInfo* graph_support_info) {

    std::cout << "DEBUG: Bridge ValidateNodesWithIsOpSupported called!" << std::endl;
    // QNN-ABI: Shared library implementation with enhanced OpBuilder validation
    std::vector<const OrtNode*> supported_nodes;
    (void)logger; // Suppress unused parameter warning

    try {
        // Get nodes using OrtApi
        OrtArrayOfConstObjects* nodes_array = nullptr;
        size_t num_nodes = 0;


        std::cout << "DEBUG: Attempting to get nodes from graph..." << std::endl;
        if (ort_api.Graph_GetNodes(graph, &nodes_array) != nullptr) {
            return supported_nodes;
        }

        std::cout << "DEBUG: Successfully retrieved nodes array from graph." << std::endl;
        if (ort_api.ArrayOfConstObjects_GetSize(nodes_array, &num_nodes) != nullptr) {
            if (nodes_array) ort_api.ReleaseArrayOfConstObjects(nodes_array);
            return supported_nodes;
        }
        std::cout << "DEBUG: Number of nodes in graph: " << num_nodes << std::endl;

        if (num_nodes == 0) {
            if (nodes_array) ort_api.ReleaseArrayOfConstObjects(nodes_array);
            return supported_nodes;
        }

        // Get nodes data for iteration
        const void* const* nodes_data = nullptr;
        if (ort_api.ArrayOfConstObjects_GetData(nodes_array, &nodes_data) != nullptr) {
            if (nodes_array) ort_api.ReleaseArrayOfConstObjects(nodes_array);
            return supported_nodes;
        }

        std::cout << "DEBUG: Successfully retrieved nodes data." << std::endl;
        // Validate each node using OpBuilder existence check + basic attribute validation
        for (size_t i = 0; i < num_nodes; ++i) {
            const OrtNode* ort_node = static_cast<const OrtNode*>(nodes_data[i]);
            const char* op_type = nullptr;
            if (ort_api.Node_GetOperatorType(ort_node, &op_type) != nullptr || op_type == nullptr) {
                continue;
            }

            if (!ort_node || !op_type) {
                continue;
            }


            std::string op_type_str(op_type);
            const auto* op_builder = GetOpBuilder(op_type_str);

            if (op_builder == nullptr) {
                continue;
            }

            std::cout << "DEBUG: Found OpBuilder for operator: " << op_type_str << std::endl;

            // Enhanced validation: OpBuilder existence + basic node validation
            if (op_builder->IsOpSupportedForABI(ort_node, ort_api, graph, logger, graph_support_info)) {
                supported_nodes.push_back(ort_node);
            }
        }

        // Clean up
        if (nodes_array) ort_api.ReleaseArrayOfConstObjects(nodes_array);

    } catch (...) {
        supported_nodes.clear();
    }

    return supported_nodes;
}

// QNN-ABI: Enhanced validation using OpBuilder and OrtApi
// bool QnnEpBridge::ValidateNodeUsingOpBuilder(
//     const OrtNode* ort_node,
//     const char* op_type,
//     const OrtApi& ort_api,
//     const OrtGraph* graph,
//     const OrtLogger* logger,
//     const OrtEpGraphSupportInfo* graph_support_info) {



// }

} // namespace qnn
} // namespace onnxruntime
