#pragma once
#include "core/graph/model.h"
#include "core/providers/cpu/cpu_execution_provider.h"
namespace onnxruntime {
namespace training {
namespace api {

// static std::unique_ptr<Environment> env;
const std::vector<std::string> GRAD_SUFFIX{"_grad.accumulation.buffer", "_grad", "_grad.accumulation.out"};
const std::string MOMENT_1_SUFFIX{".exp_avg"};
const std::string MOMENT_2_SUFFIX{".exp_avg_sq"};
// TODO: don't hard code the state names, should get the state names according to the optimizer types.
const std::vector<std::string> MOMENT_STATE_NAMES{"momentum0", "momentum1"};

void GetGraphInputOutputNames(const Graph& graph,
                              std::vector<std::string>& input_names,
                              std::vector<std::string>& output_names);
bool GetParamNameFromSuffix(const std::string& name, const std::string& suffix, std::string& param_name);

bool GetParamNameFromGradient(const std::string& grad_name, std::string& param_name);

Status OrtValueLike(const SessionState& sess_state, const OrtValue& input_val, OrtValue& output_val);

}  // namespace api
}  // namespace training
}  // namespace onnxruntime
