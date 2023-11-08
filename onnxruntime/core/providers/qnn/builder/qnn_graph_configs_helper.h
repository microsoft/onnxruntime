// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/common/inlined_containers_fwd.h>

#include "HTP/QnnHtpGraph.h"

namespace onnxruntime {
namespace qnn {

/**
 * Helper class for building a null-terminated list of QNN Graph configurations.
 * A QNN configuration consists of multiple objects with references to each other. This
 * class ensures that all configuration objects have the same lifetime, so that they remain valid
 * across the call to graphCreate().
 */
class QnnGraphConfigsBuilder {
 public:
  /**
   * Returns a pointer to the beginning of a null-terminated array of QNN Graph configurations.
   * This result is passed QNN's graphCreate() API.
   *
   * \return Pointer to null-terminated QnnGraph_Config_t* array.
   */
  const QnnGraph_Config_t** GetQnnGraphConfigs();

  /**
   * Creates and returns a reference to a new HTP graph configuration object. The object is initialized to
   * the QNN recommended default value. The caller is meant to override fields in this object.
   *
   * \return A reference to a default QnnHtpGraph_CustomConfig_t object.
   */
  QnnHtpGraph_CustomConfig_t& PushHtpGraphCustomConfig();

  /**
   * Creates and returns a reference to a new graph configuration object. The object is initialized to
   * the QNN recommended default value. The caller is meant to override fields in this object.
   *
   * \return A reference to a default QnnGraph_Config_t object.
   */
  QnnGraph_Config_t& PushGraphConfig();

 private:
  bool IsNullTerminated() const {
    return !graph_config_ptrs_.empty() && graph_config_ptrs_.back() == nullptr;
  }

  InlinedVector<QnnHtpGraph_CustomConfig_t> htp_custom_graph_configs_;
  InlinedVector<QnnGraph_Config_t> graph_configs_;
  InlinedVector<const QnnGraph_Config_t*> graph_config_ptrs_;
};

}  // namespace qnn
}  // namespace onnxruntime
