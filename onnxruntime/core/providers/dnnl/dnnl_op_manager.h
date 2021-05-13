// Copyright(C) 2021 Intel Corporation
// Licensed under the MIT License

#pragma once
#include "dnnl_node_capability.h"
#include <map>
#include <string>
#include <memory>

namespace onnxruntime {
class DnnlOpManager {
 public:
  DnnlOpManager();

  /**
  * This will check if the ORT node is Supported by the DNNL execution provider
  *
  * Several things will be checked from the node
  * - Is the OpType is regestered with the DNNL execution provider?
  * - Are the tensor dimensions Supported by the DNNL execution provider
  * - Are operator attributes Supported by the DNNL execution provider
  *
  * @param node the node that is being checked
  * 
  * @return true if the node is Supported by the DNNL execution provider
  *         false is returned otherwise.
  */
  bool IsNodeSupported(const Node* node) const;

  /**
  * Find out if the OpType is one of the OpTypes Supported by the DNNL execution provider
  *
  * This only looks at the OpType it does not look at other factors that may mean
  * the operator is not Supported.
  *
  * @param opType the name of the operator i.e. "Add" or "Conv" etc.
  *
  * @return true is the OpType is one of those Supported by the DNNL execution provider
  *         false is returned otherwise.
  */
  bool IsOpTypeAvalible(const std::string& opType) const;

 private:
  std::map<std::string, std::unique_ptr<DnnlNodeCapability>> dnnl_ops_map_;
};
}  // namespace onnxruntime