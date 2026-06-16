// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
class GraphViewer;

namespace openvino_ep {

namespace qdq_scales_fix {
Status Transform(const GraphViewer& src_graph,
                 const logging::Logger& logger,
                 /*out*/ std::unique_ptr<onnxruntime::Model>& model);
}
}  // namespace openvino_ep
}  // namespace onnxruntime
