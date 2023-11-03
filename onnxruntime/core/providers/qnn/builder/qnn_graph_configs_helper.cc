// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/qnn_graph_configs_helper.h"

#include "HTP/QnnHtpGraph.h"

namespace onnxruntime {
namespace qnn {

const QnnGraph_Config_t** QnnGraphConfigsHolder::GetQnnGraphConfigs() {
  return graph_config_ptrs_.empty() ? nullptr : graph_config_ptrs_.data();
}

QnnHtpGraph_CustomConfig_t& QnnGraphConfigsHolder::PushHtpGraphCustomConfig() {
  htp_custom_graph_configs_.push_back(QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT);
  return htp_custom_graph_configs_.back();
}

QnnGraph_Config_t& QnnGraphConfigsHolder::PushGraphConfig(bool is_last) {
  graph_configs_.push_back(QNN_GRAPH_CONFIG_INIT);
  QnnGraph_Config_t& config = graph_configs_.back();

  graph_config_ptrs_.push_back(&config);
  if (is_last) {
    graph_config_ptrs_.push_back(nullptr);
  }
  return config;
}

}  // namespace qnn
}  // namespace onnxruntime
