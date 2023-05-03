// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "QnnTypes.h"
#include "core/session/onnxruntime_cxx_api.h"

#include "nlohmann/json.hpp"

#include <functional>
#include <numeric>
#include <vector>
#include <string>
#include <unordered_set>

namespace onnxruntime {
namespace qnn {
class QnnOpConfigWrapper;

namespace utils {
size_t GetElementSizeByType(const Qnn_DataType_t& data_type);

size_t GetElementSizeByType(ONNXTensorElementDataType elem_type);

// TODO: make these work with Wrappers?
std::ostream& operator<<(std::ostream& out, const Qnn_Param_t& qnn_param);
std::ostream& operator<<(std::ostream& out, const Qnn_Tensor_t& tensor);
std::ostream& operator<<(std::ostream& out, const QnnOpConfigWrapper& op_conf_wrapper);

class QnnJSONGraph {
 public:
  QnnJSONGraph();

  void AddOp(const QnnOpConfigWrapper& op_conf_wrapper);
  const nlohmann::json& Finalize();

 private:
  void AddOpTensors(gsl::span<const Qnn_Tensor_t> tensors);

  nlohmann::json json_;
  std::unordered_set<std::string> seen_tensors_; // Keep track of tensors already added to json object.
  std::unordered_set<std::string> seen_op_types_;
};

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
