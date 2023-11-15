// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_graph_configs_helper.h"

#include "HTP/QnnHtpGraph.h"

namespace onnxruntime {
namespace qnn {

const QnnGraph_Config_t** QnnGraphConfigsBuilder::GetQnnGraphConfigs() {
  if (graph_config_ptrs_.empty()) {
    return nullptr;
  }

  if (!IsNullTerminated()) {
    graph_config_ptrs_.push_back(nullptr);
  }

  return graph_config_ptrs_.data();
}

QnnHtpGraph_CustomConfig_t& QnnGraphConfigsBuilder::PushHtpGraphCustomConfig() {
  htp_custom_graph_configs_.push_back(QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT);
  return htp_custom_graph_configs_.back();
}

QnnGraph_Config_t& QnnGraphConfigsBuilder::PushGraphConfig() {
  graph_configs_.push_back(QNN_GRAPH_CONFIG_INIT);
  QnnGraph_Config_t& config = graph_configs_.back();

  // Add pointer to this new graph config to the list of graph config pointers.
  if (IsNullTerminated()) {
    graph_config_ptrs_.back() = &config;  // Replace last nullptr entry.
  } else {
    graph_config_ptrs_.push_back(&config);
  }

  return config;
}

}  // namespace qnn
}  // namespace onnxruntime
