// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/contexts.h"

namespace onnxruntime {
namespace openvino_ep {

using sw = SharedContext::SharedWeights;

// Creates a new model without the DQ/Q operators in the src graph as per pre-defined rulesets
Status CreateModelWithStrippedQDQNodes(const GraphViewer& src_graph,
                                       const logging::Logger& logger,
                                       bool enable_ovep_weight_sharing,
                                       bool enable_ovep_qdq_optimizer,
                                       /*out*/ std::unique_ptr<onnxruntime::Model>& model,
                                       /*out*/ sw& shared_weights);

bool dumpMetaDataMapToBinary(const sw::Metadata::Map& shared_weights, const std::string& filename);
}  // namespace openvino_ep
}  // namespace onnxruntime
