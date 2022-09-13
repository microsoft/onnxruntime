// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/common/path.h"
#include "core/framework/framework_common.h"
#include "core/platform/env.h"
#include "core/platform/path_lib.h"
#include "orttraining/training_api/include/on_device_utils.h"
#include "core/providers/cpu/cpu_execution_provider.h"

namespace onnxruntime {
namespace training {
namespace api {

Status GetParametersDifference(const OrtValue output_params, const OrtValue old_output_params, OrtValue& output) {
  // allocate memory for output.
  std::unique_ptr<IExecutionProvider> cpu_execution_provider = std::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false));

  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), {output_params.Get<Tensor>().Shape().Size()},
                       cpu_execution_provider->GetAllocator(0, OrtMemTypeDefault),
                       output);

  // calculate the differences between parameters
  const float* params = output_params.Get<Tensor>().Data<float>();
  const float* old_params = old_output_params.Get<Tensor>().Data<float>();
  float* output_params_data = output.GetMutable<Tensor>()->MutableData<float>();
  for (int64_t i = 0; i < output_params.Get<Tensor>().Shape().Size(); ++i) {
    output_params_data[i] = params[i] - old_params[i];
    if (i == 0) {
      std::cout << "params = " << params[i] << std::endl;
      std::cout << "old_params = " << old_params[i] << std::endl;
      std::cout << "diff = " << output_params_data[i] << std::endl;
    }
  }

  return Status::OK();
}

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
