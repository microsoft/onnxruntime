// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#pragma once

#include "core/providers/shared_library/provider_api.h"
#define ORT_API_MANUAL_INIT
#include "core/session/onnxruntime_cxx_api.h"

#include "core/framework/allocatormgr.h"
// #include "core/framework/arena_extend_strategy.h"
// #include "core/framework/execution_provider.h"
// #include "core/platform/ort_mutex.h"
// #include "migraphx_execution_provider_info.h"

namespace onnxruntime {

bool IsGraphInput(const GraphViewer& graph, const std::string& name);

bool IsGraphInitializer(const GraphViewer& graph, const std::string& name, bool check_outer_scope = true);

const Node* GetInputNode(const Node& node, int arg_index);

std::size_t getNodeInputNum(const Node& node);

bool isInputNode(const Node* node, const std::string& name);

bool canEvalShapeGeneral(const GraphViewer& graph, const Node* node, std::vector<NodeIndex>& input_nodes);

bool canEvalNodeArgument(const GraphViewer& graph, const Node* node, std::vector<std::size_t> indices, std::vector<NodeIndex>& input_nodes);

}
