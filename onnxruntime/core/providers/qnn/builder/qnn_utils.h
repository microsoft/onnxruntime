// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "QnnTypes.h"
#include "core/session/onnxruntime_cxx_api.h"

#include <functional>
#include <numeric>
#include <vector>
#include <string>

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

}  // namespace utils
}  // namespace qnn
}  // namespace onnxruntime
