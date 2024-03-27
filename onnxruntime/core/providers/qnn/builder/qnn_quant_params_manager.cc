// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "qnn_quant_params_manager.h"
#include <cassert>
#include "core/framework/node_unit.h"

namespace onnxruntime {
namespace qnn {

Status QnnQuantParamsManager::GetQnnQuantParams(const onnxruntime::NodeUnitIODef& node_unit_io,
                                                /*out*/ Qnn_QuantizeParams_t& qnn_quant_params) {
  if (!node_unit_io.quant_param.has_value()) {
    qnn_quant_params.encodingDefinition = QNN_DEFINITION_UNDEFINED;
    qnn_quant_params.quantizationEncoding = QNN_QUANTIZATION_ENCODING_UNDEFINED;
    return Status::OK();
  }
  const auto& ort_quant_params = node_unit_io.quant_param.value();
  const auto* scale_shape = ort_quant_params.scale.Shape();  // TODO: May not need to check shape at all, just presence of axis
  assert(scale_shape != nullptr);

  auto scale_rank = scale_shape->dim_size();
  const bool is_per_tensor = scale_rank == 0;
  assert(is_per_tensor == !ort_quant_params.axis.has_value());

  // TODO: Finish

  return Status::OK();
}
}  // namespace qnn
}  // namespace onnxruntime
