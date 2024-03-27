// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "QnnTypes.h"  // From QNN_SDK/include/QNN
#include "core/common/status.h"
#include "core/framework/node_unit.h"

namespace onnxruntime {
namespace qnn {

class QnnQuantParamsManager {
 public:
  QnnQuantParamsManager() = default;

  Status GetQnnQuantParams(const onnxruntime::NodeUnitIODef& node_unit_io,
                           /*out*/ Qnn_QuantizeParams_t& qnn_quant_params);

 private:
  // Maps a tensor name to its QNN quantization parameters.
  std::unordered_map<std::string, Qnn_QuantizeParams_t> quant_params_;

  // For per-channel quantization, this buffer stores the actual scale and zp values
  // to which the Qnn_QuantizeParams_t structure points.
  std::vector<Qnn_ScaleOffset_t> scale_offset_data_;  // For axisScaleOffset
};
}  // namespace qnn
}  // namespace onnxruntime
