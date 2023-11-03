// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/common/inlined_containers_fwd.h>

#include "HTP/QnnHtpGraph.h"

namespace onnxruntime {
namespace qnn {

class QnnGraphConfigsHolder {
 public:
  const QnnGraph_Config_t** GetQnnGraphConfigs();
  QnnHtpGraph_CustomConfig_t& PushHtpGraphCustomConfig();
  QnnGraph_Config_t& PushGraphConfig(bool is_last);

 private:
  InlinedVector<QnnHtpGraph_CustomConfig_t> htp_custom_graph_configs_;
  InlinedVector<QnnGraph_Config_t> graph_configs_;
  InlinedVector<const QnnGraph_Config_t*> graph_config_ptrs_;
};

}  // namespace qnn
}  // namespace onnxruntime
