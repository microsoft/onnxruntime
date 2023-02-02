// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "python/onnxruntime_pybind_exceptions.h"
#include "python/onnxruntime_pybind_mlvalue.h"
#include "python/onnxruntime_pybind_state_common.h"

#include "core/platform/env.h"
#include <unordered_map>
#include <cstdlib>

namespace onnxruntime {
namespace python {
namespace py = pybind11;

using namespace onnxruntime::logging;

using ExecutionProviderMap = std::unordered_map<std::string, std::shared_ptr<IExecutionProvider>>;
using ExecutionProviderLibInfoMap = std::unordered_map<std::string, std::pair<std::string, ProviderOptions>>;

class ORTTrainingPythonEnv {
 public:
  ORTTrainingPythonEnv();

  Environment& GetORTEnv();

  std::shared_ptr<IExecutionProvider> GetExecutionProviderInstance(const std::string& provider_type,
                                                                   size_t hash);

  void AddExecutionProvider(const std::string& provider_type,
                            size_t hash,
                            std::unique_ptr<IExecutionProvider> execution_provider);

  void RegisterExtExecutionProviderInfo(const std::string& provider_type,
                                        const std::string& provider_lib_path,
                                        const ProviderOptions& default_options);

  const std::vector<std::string>& GetAvailableTrainingExecutionProviderTypes();

  ExecutionProviderLibInfoMap ext_execution_provider_info_map_;

  void ClearExecutionProviderInstances();

 private:
  std::string GetExecutionProviderMapKey(const std::string& provider_type,
                                         size_t hash);

  std::unique_ptr<Environment> ort_env_;
  ExecutionProviderMap execution_provider_instances_map_;
  std::vector<std::string> available_training_eps_;
};

}  // namespace python
}  // namespace onnxruntime
