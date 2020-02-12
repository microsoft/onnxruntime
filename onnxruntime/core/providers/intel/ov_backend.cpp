// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include <map>
#include <string>
#include <memory>
#include <sstream>
#include <fstream>

#include <inference_engine.hpp>
#include <ngraph/frontend/onnx_import/onnx.hpp>

// FIXME: These should not be needed after v1 ops
// are fully integrated into onnx importer
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/opset1_upgrade.hpp>
#include <ngraph/pass/convert_fp32_to_fp16.hpp>

// FIXME: Remove before production
#include <ngraph/serializer.hpp>

#include "core/session/onnxruntime_cxx_api.h"
#include "core/graph/graph.h"
#include "core/common/logging/logging.h"

#include "ov_backend.h"

namespace onnxruntime {
namespace intel_ep {




}  // namespace intel_ep
}  // namespace onnxruntime