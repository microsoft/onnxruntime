// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/framework/execution_frame.h"
#include "core/framework/session_state.h"
#include "core/graph/op.h"
#include "core/common/logging/logging.h"
using namespace ::onnxruntime::common;
namespace onnxruntime {

OpKernelContext::OpKernelContext(const OpKernel* kernel,
                                 const logging::Logger& logger)
    : kernel_(kernel),
      logger_(&logger) {

  ORT_ENFORCE(kernel != nullptr, "OpKernel was null");
}

}  // namespace onnxruntime
