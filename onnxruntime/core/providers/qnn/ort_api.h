// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#define BUILD_QNN_EP_STATIC 0

#if BUILD_QNN_EP_STATIC
#ifdef _WIN32
#include <winmeta.h>
#include "core/platform/tracing.h"
#include "core/platform/windows/logging/etw_sink.h"
#endif

// Includes when building QNN EP statically
#include "onnx/defs/data_type_utils.h"
#include "core/common/common.h"
#include "core/common/status.h"
#include "core/common/safeint.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/capture.h"
#include "core/common/path_string.h"
#include "core/platform/env.h"
#include "core/framework/data_types.h"
#include "core/framework/float16.h"
#include "core/framework/run_options.h"
#include "core/framework/execution_provider.h"
#include "core/framework/model_metadef_id_generator.h"
#include "core/framework/compute_capability.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/node_unit.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/utils.h"
#include "core/graph/constants.h"
#include "core/graph/basic_types.h"
#include "core/graph/model.h"
#include "core/graph/graph_viewer.h"
#include "core/optimizer/qdq_transformer/selectors_actions/qdq_selectors.h"
#include "core/optimizer/qdq_transformer/selectors_actions/shared/utils.h"
#include "core/providers/common.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/onnxruntime_cxx_api.h"
#else
// Includes when building QNN EP as a shared library
#include "core/providers/shared_library/provider_api.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"
#endif

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/onnxruntime_run_options_config_keys.h"

#include <memory>

namespace onnxruntime {
std::unique_ptr<ONNX_NAMESPACE::TypeProto> TypeProto__Create();

#if BUILD_QNN_EP_STATIC
using Node_EdgeEnd = Node::EdgeEnd;
#endif

std::unique_ptr<Node_EdgeEnd> Node_EdgeEnd__Create(const Node& node, int src_arg_index, int dst_arg_index);
std::unique_ptr<NodeUnit> NodeUnit__Create(gsl::span<const Node* const> dq_nodes,
                                           const Node& target_node,
                                           gsl::span<const Node* const> q_nodes,
                                           NodeUnit::Type unit_type,
                                           gsl::span<const NodeUnitIODef> inputs,
                                           gsl::span<const NodeUnitIODef> outputs,
                                           size_t input_edge_count,
                                           gsl::span<const Node_EdgeEnd* const> output_edges);

namespace logging {
std::unique_ptr<Capture> Capture__Create(const Logger& logger, logging::Severity severity, const char* category,
                                         logging::DataType dataType, const CodeLocation& location);
}  // namespace logging
}  // namespace onnxruntime
