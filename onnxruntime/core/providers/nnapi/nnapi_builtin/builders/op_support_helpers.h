// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <vector>

#include "core/common/common.h"
#include "core/providers/nnapi/nnapi_builtin/builders/helper.h"
#include "core/providers/nnapi/nnapi_builtin/builders/op_support_checker.h"
#include "core/providers/shared/node_unit/node_unit.h"

namespace onnxruntime::nnapi::op_support_helpers {

inline bool IsNodeLayoutNHWC(const NodeUnit& node_unit) {
  return node_unit.Domain() == kMSInternalNHWCDomain;
}

bool IsQuantizationScaleSupported(const InitializedTensorSet& initializers,
                                  const NodeUnitIODef& io_def,
                                  const OpSupportCheckParams& params,
                                  const std::string& op_type,
                                  bool is_quant_matmul,
                                  bool is_conv_matmul_u8s8_weight);

bool IsQuantizationZeroPointSupported(const InitializedTensorSet& initializers,
                                      const NodeUnitIODef& io_def,
                                      const std::string& op_type,
                                      const Path& model_path,
                                      bool is_quant_matmul,
                                      bool is_conv_matmul_u8s8_weight);

// Check if the given quantized input(s) or output(s) is supported
bool IsQuantizedIOSupported(const InitializedTensorSet& initializers, const NodeUnit& node_unit,
                            const std::vector<size_t>& indices, const OpSupportCheckParams& params, ArgType arg_type);

// Some Quantized NNAPI operations have required output scale and zero point
// e.g. Softmax (uint8) requires output scale be 1.f/256 and zp be 0
// This helper function checks if the given io_def has required scale and zp
bool HasRequiredScaleAndZeroPoint(const InitializedTensorSet& initializers,
                                  const std::string& op_desc,
                                  const NodeUnitIODef& io_def,
                                  const Path& path,
                                  float required_scale, int32_t required_zp);

}  // namespace onnxruntime::nnapi::op_support_helpers
