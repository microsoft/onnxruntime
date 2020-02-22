// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/graph/onnx_protobuf.h"

#include "core/session/inference_session.h"
#include "core/framework/op_kernel.h"
#include "core/framework/session_state.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/op.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "gtest/gtest.h"
#include "test_utils.h"
using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
using namespace logging;

namespace test {

class XPUExecutionProvider : public IExecutionProvider {
 public:
  XPUExecutionProvider() : IExecutionProvider{onnxruntime::kCpuExecutionProvider} {}
};

}  // namespace test
}  // namespace onnxruntime
