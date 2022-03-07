// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/model.h"
#include "core/framework/tensorprotoutils.h"

#include "tvm_runner.h"
#include "tvm_utils.h"
#include "tvm_compiler.h"
#include "tvm_api.h"


using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace tvm {

TVMRunner::TVMRunner(const TvmEPOptions& options,
                     const std::shared_ptr<TvmModule>& mod,
                     const InputsInfoMap& inputs_info,
                     const std::vector<DLTensor>& output_tensors) {
    runner_ = getTVMRunnerImpl(options.executor, mod, inputs_info, options.output_shapes, output_tensors);
}

common::Status TVMRunner::operator()(FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
  return runner_->run(api, context);
}

}   // namespace tvm
}   // namespace onnxruntime
