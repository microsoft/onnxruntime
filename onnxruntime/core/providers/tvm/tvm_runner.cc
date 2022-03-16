// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/model.h"
#include "core/framework/tensorprotoutils.h"

#include "tvm_runner.h"


using namespace ONNX_NAMESPACE;
namespace onnxruntime {
namespace tvm {

TVMRunner::TVMRunner(const TvmEPOptions& options,
                     const std::shared_ptr<TvmModule>& mod,
                     const InputsInfoMap& inputs_info,
                     const std::vector<DLTensor>& output_tensors) {
  runner_ = getTVMRunnerImpl(mod, options, inputs_info, output_tensors);
}

common::Status TVMRunner::operator()(FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context) {
  return runner_->run(api, context);
}

}   // namespace tvm
}   // namespace onnxruntime
