// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include "core/session/onnxruntime_c_api.h"
#include <vector>
#include <string>

namespace onnxruntime {
namespace qnn {

// QNN-ABI: Bridge interface for shared library builds
// Uses enhanced OpBuilder validation with OrtApi-only types
class QnnEpBridge {
public:

    // Main function to validate nodes using enhanced OpBuilder validation
    // Returns vector of supported OrtNode pointers
    static std::vector<const OrtNode*> ValidateNodesWithIsOpSupported(
        const OrtGraph* graph,
        const OrtApi& ort_api,
        const OrtLogger* logger,
        const OrtEpGraphSupportInfo* graph_support_info);

// private:
//     // Enhanced validation using OpBuilder + OrtApi
//     static bool ValidateNodeUsingOpBuilder(
//         const OrtNode* ort_node,
//         const char* op_type,
//         const OrtApi& ort_api,
//         const OrtGraph* graph,
//         const OrtLogger* logger);
};

} // namespace qnn
} // namespace onnxruntime
