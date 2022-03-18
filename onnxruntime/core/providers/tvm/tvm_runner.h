// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TVM_RUNNER_H
#define TVM_RUNNER_H

#include <vector>
#include <memory>

#include "tvm_runner_impl.h"


namespace onnxruntime {
namespace tvm {

class TVMRunner {
public:
  TVMRunner() = delete;
  virtual ~TVMRunner() = default;

  TVMRunner(const TvmEPOptions& options,
            const std::shared_ptr<TvmModule>& mod,
            const InputsInfoMap& inputs_info,
            const std::vector<DLTensor>& output_tensor);

  common::Status operator()(FunctionState state, const OrtCustomOpApi* api, OrtKernelContext* context);

private:
  std::shared_ptr<RunnerImpl> runner_;
};

}   // namespace tvm
}   // namespace onnxruntime

#endif  // TVM_TVM_RUNNER_H
