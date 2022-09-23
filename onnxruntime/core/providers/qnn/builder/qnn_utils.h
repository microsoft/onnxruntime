// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "QnnTypes.h"

#include <functional>
#include <numeric>
#include <vector>
#include <string>

namespace onnxruntime {
namespace qnn {
namespace utils {
uint32_t GetTensorIdFromName(const std::string& name);

size_t GetElementSizeByType(const Qnn_DataType_t& data_type);

int64_t GetDataSize(const std::vector<uint32_t>& dims);

std::ostream& operator<<(std::ostream& out, const Qnn_Param_t& qnn_param);
std::ostream& operator<<(std::ostream& out, const Qnn_Tensor_t& tensor);
std::ostream& operator<<(std::ostream& out, const Qnn_OpConfig_t& op_definition);

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
