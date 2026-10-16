// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/session_loop.h"

#include <algorithm>

#include "core/common/safeint.h"
#include "core/framework/data_transfer.h"
#include "core/framework/run_options.h"
#include "core/framework/tensor.h"
#include "core/graph/graph.h"
#include "core/session/IOBinding.h"
#include "core/session/inference_session.h"
#include "core/session/stateful_execution_utils.h"

namespace onnxruntime {
using namespace stateful_execution;

namespace {
Status CheckScalar(const NodeArg& arg, int element_type) {
  ORT_RETURN_IF_ERROR(CheckTensorType(arg));
  const auto& type = arg.TypeAsProto()->tensor_type();
  ORT_RETURN_IF_NOT(type.elem_type() == element_type &&
                        (!type.has_shape() || type.shape().dim_size() == 0),
                    "SessionLoop condition requires a scalar of the configured type: ", arg.Name());
  return Status::OK();
}

}  // namespace

SessionLoop::SessionLoop(InferenceSession& session, SessionLoopOptions options)
    : session_(session), options_(std::move(options)), fetch_names_(options_.output_names) {
  auto add_output = [&](const std::string& name) {
    if (FindOutput(fetch_names_, name) == fetch_names_.size()) {
      fetch_names_.push_back(name);
    }
  };
  for (const auto& binding : options_.feedback) {
    add_output(binding.output_name);
  }
  if (!options_.generated_token_output.empty()) {
    add_output(options_.generated_token_output);
  }
  for (const auto& condition : options_.exit_conditions) {
    if (condition.kind == LoopExitKind::BooleanOutput) {
      add_output(condition.output_name);
    }
  }
}

Status SessionLoop::Create(InferenceSession& session, SessionLoopOptions options,
                           std::shared_ptr<const SessionLoop>& out) {
  std::unique_ptr<IOBinding> binding;
  ORT_RETURN_IF_ERROR(session.NewIOBinding(&binding));
  ORT_RETURN_IF(options.exit_conditions.empty(), "SessionLoop requires at least one explicit exit condition");
  ORT_RETURN_IF(options.output_names.empty(), "SessionLoop requires at least one selected output");
  const auto& inputs = *session.GetModelInputs().second;
  const auto& outputs = *session.GetModelOutputs().second;
  for (const auto* input : inputs) {
    ORT_RETURN_IF_ERROR(CheckTensorType(*input));
  }
  InlinedHashSet<std::string> destinations;
  for (const auto& name : options.finite_input_names) {
    ORT_RETURN_IF_NOT(FindArgument(inputs, name) && destinations.insert(name).second,
                      "Invalid or duplicate finite input: ", name);
  }
  ORT_RETURN_IF(!options.nonempty_input.empty() &&
                    std::find(options.finite_input_names.begin(), options.finite_input_names.end(),
                              options.nonempty_input) == options.finite_input_names.end(),
                "Nonempty input must name a finite input");
  for (const auto& binding : options.feedback) {
    const auto* input = FindArgument(inputs, binding.input_name);
    const auto* output = FindArgument(outputs, binding.output_name);
    ORT_RETURN_IF_NOT(input && output && destinations.insert(binding.input_name).second,
                      "Invalid or conflicting feedback binding: ", binding.input_name);
    ORT_RETURN_IF_ERROR(CheckCompatibleTypes(*output, *input));
  }
  InlinedHashSet<std::string> selected;
  for (const auto& name : options.output_names) {
    const auto* output = FindArgument(outputs, name);
    ORT_RETURN_IF_NOT(output && selected.insert(name).second, "Invalid or duplicate selected output: ", name);
    ORT_RETURN_IF_ERROR(CheckTensorType(*output));
  }
  if (!options.generated_token_output.empty()) {
    const auto* output = FindArgument(outputs, options.generated_token_output);
    ORT_RETURN_IF_NOT(output, "Unknown generated-token output");
    ORT_RETURN_IF_ERROR(CheckScalar(*output, ONNX_NAMESPACE::TensorProto_DataType_INT64));
  }
  InlinedHashSet<LoopExitKind> kinds;
  for (const auto& condition : options.exit_conditions) {
    ORT_RETURN_IF_NOT(kinds.insert(condition.kind).second, "Duplicate SessionLoop exit condition");
    switch (condition.kind) {
      case LoopExitKind::EndOfInputs:
        ORT_RETURN_IF(options.finite_input_names.empty(), "EndOfInputs requires a finite input source");
        break;
      case LoopExitKind::MaxIterations:
      case LoopExitKind::MaxGeneratedTokens:
        ORT_RETURN_IF(condition.limit < 0, "SessionLoop limits must be nonnegative");
        if (condition.kind == LoopExitKind::MaxIterations) {
          break;
        }
        [[fallthrough]];
      case LoopExitKind::Eos:
        ORT_RETURN_IF(options.generated_token_output.empty(), "Token exits require a generated-token output");
        if (condition.kind == LoopExitKind::Eos) {
          ORT_RETURN_IF(condition.eos_token_ids.empty() ||
                            std::any_of(condition.eos_token_ids.begin(), condition.eos_token_ids.end(),
                                        [](int64_t id) { return id < 0; }),
                        "EOS requires nonnegative token IDs");
        }
        break;
      case LoopExitKind::BooleanOutput: {
        const auto* output = FindArgument(outputs, condition.output_name);
        ORT_RETURN_IF_NOT(output, "Unknown boolean exit output");
        ORT_RETURN_IF_ERROR(CheckScalar(*output, ONNX_NAMESPACE::TensorProto_DataType_BOOL));
        break;
      }
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unknown SessionLoop exit kind");
    }
  }
  out = std::shared_ptr<const SessionLoop>(new SessionLoop(session, std::move(options)));
  return Status::OK();
}

Status SessionLoop::ValidateInputs(const SessionLoopInputs& inputs) const {
  const auto& arguments = *session_.GetModelInputs().second;
  InlinedHashSet<std::string> finite(options_.finite_input_names.begin(), options_.finite_input_names.end());
  ORT_RETURN_IF(inputs.inputs.size() + finite.size() != arguments.size(), "SessionLoop input count mismatch");
  for (const auto& [name, value] : inputs.inputs) {
    const auto* argument = FindArgument(arguments, name);
    ORT_RETURN_IF_NOT(argument && !finite.contains(name), "Unknown or conflicting invariant input: ", name);
    ORT_RETURN_IF_ERROR(CheckInput(*argument, value));
  }
  ORT_RETURN_IF(finite.empty() && !inputs.batches.empty(), "Batches require finite input names");
  ORT_RETURN_IF(!options_.nonempty_input.empty() && inputs.batches.empty(), "Prefill input must not be empty");
  for (const auto& batch : inputs.batches) {
    ORT_RETURN_IF(batch.size() != finite.size(), "SessionLoop finite batch input count mismatch");
    for (const auto& [name, value] : batch) {
      ORT_RETURN_IF_NOT(finite.contains(name), "Unknown finite input: ", name);
      ORT_RETURN_IF_ERROR(CheckInput(*FindArgument(arguments, name), value));
      ORT_RETURN_IF(name == options_.nonempty_input && value.Get<Tensor>().Shape().Size() == 0,
                    "Prefill input tensor must not be empty");
    }
  }
  return Status::OK();
}

Status SessionLoop::CreateExecution(SessionLoopInputs inputs, std::unique_ptr<SessionLoopExecution>& out) const {
  ORT_RETURN_IF_ERROR(ValidateInputs(inputs));
  out = std::unique_ptr<SessionLoopExecution>(new SessionLoopExecution(shared_from_this(), std::move(inputs)));
  return Status::OK();
}

SessionLoopExecution::SessionLoopExecution(std::shared_ptr<const SessionLoop> loop, SessionLoopInputs inputs)
    : loop_(std::move(loop)), initial_inputs_(std::move(inputs)) {}

void SessionLoopExecution::EvaluateExits(const InlinedVector<OrtValue>& values) {
  progress_.exit_reasons.clear();
  const auto& options = loop_->options_;
  for (const auto& condition : options.exit_conditions) {
    bool stop = false;
    switch (condition.kind) {
      case LoopExitKind::EndOfInputs:
        stop = static_cast<uint64_t>(progress_.iterations) >= initial_inputs_.batches.size();
        break;
      case LoopExitKind::MaxIterations:
        stop = progress_.iterations >= condition.limit;
        break;
      case LoopExitKind::MaxGeneratedTokens:
        stop = progress_.generated_tokens >= condition.limit;
        break;
      case LoopExitKind::Eos:
        if (!values.empty()) {
          const int64_t token = *values[FindOutput(loop_->fetch_names_, options.generated_token_output)]
                                     .Get<Tensor>()
                                     .Data<int64_t>();
          stop = std::find(condition.eos_token_ids.begin(), condition.eos_token_ids.end(), token) !=
                 condition.eos_token_ids.end();
        }
        break;
      case LoopExitKind::BooleanOutput:
        stop = !values.empty() &&
               *values[FindOutput(loop_->fetch_names_, condition.output_name)].Get<Tensor>().Data<bool>();
        break;
    }
    if (stop) {
      progress_.exit_reasons.push_back(condition.kind);
    }
  }
  std::sort(progress_.exit_reasons.begin(), progress_.exit_reasons.end());
  progress_.completed = !progress_.exit_reasons.empty();
}

Status SessionLoopExecution::Step(SessionLoopStepResult& out) {
  ORT_RETURN_IF(requires_reset_, "SessionLoop execution requires Reset after a failure or cancellation");
  if (cancelled_.load()) {
    requires_reset_ = true;
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "SessionLoop execution cancelled; call Reset before reuse");
  }
  if (!progress_.completed) {
    EvaluateExits({});
  }
  if (progress_.completed) {
    out = progress_;
    return Status::OK();
  }
  ORT_RETURN_IF(!loop_->options_.finite_input_names.empty() &&
                    static_cast<uint64_t>(progress_.iterations) >= initial_inputs_.batches.size(),
                "SessionLoop finite inputs exhausted without EndOfInputs");

  // Any exception or error after this point conservatively invalidates mutable state.
  requires_reset_ = true;
  std::unique_ptr<IOBinding> binding;
  ORT_RETURN_IF_ERROR(loop_->session_.NewIOBinding(&binding));
  for (const auto& [name, value] : initial_inputs_.inputs) {
    ORT_RETURN_IF_ERROR(binding->BindInput(name, value));
  }
  if (!loop_->options_.finite_input_names.empty()) {
    for (const auto& [name, value] : initial_inputs_.batches[static_cast<size_t>(progress_.iterations)]) {
      ORT_RETURN_IF_ERROR(binding->BindInput(name, value));
    }
  }
  for (size_t i = 0; i < feedback_.size(); ++i) {
    ORT_RETURN_IF_ERROR(binding->BindInput(loop_->options_.feedback[i].input_name, feedback_[i]));
  }
  for (const auto& name : loop_->fetch_names_) {
    ORT_RETURN_IF_ERROR(binding->BindOutput(name));
  }
  ORT_RETURN_IF_ERROR(binding->SynchronizeInputs());
  RunOptions run_options;
  ORT_RETURN_IF_ERROR(loop_->session_.Run(run_options, *binding));
  ORT_RETURN_IF_ERROR(binding->SynchronizeOutputs());
  ORT_RETURN_IF(cancelled_.load(), "SessionLoop execution cancelled during a step; call Reset before reuse");

  InlinedVector<OrtValue> snapshots;
  snapshots.reserve(binding->GetOutputs().size());
  const auto allocator = binding->GetCPUAllocator(kCpuExecutionProvider);
  CPUDataTransfer transfer;
  for (const auto& value : binding->GetOutputs()) {
    const Tensor& source = value.Get<Tensor>();
    OrtValue snapshot;
    Tensor::InitOrtValue(source.DataType(), source.Shape(), allocator, snapshot);
    if (source.SizeInBytes() != 0) {
      ORT_RETURN_IF_ERROR(transfer.CopyTensor(source, *snapshot.GetMutable<Tensor>()));
    }
    snapshots.push_back(std::move(snapshot));
  }
  for (const auto& condition : loop_->options_.exit_conditions) {
    if (condition.kind == LoopExitKind::BooleanOutput) {
      ORT_RETURN_IF(snapshots[FindOutput(loop_->fetch_names_, condition.output_name)]
                            .Get<Tensor>()
                            .Shape()
                            .NumDimensions() != 0,
                    "Boolean exit output must be scalar");
    }
  }
  if (!loop_->options_.generated_token_output.empty()) {
    const auto& token =
        snapshots[FindOutput(loop_->fetch_names_, loop_->options_.generated_token_output)].Get<Tensor>();
    ORT_RETURN_IF(token.Shape().NumDimensions() != 0 || *token.Data<int64_t>() < 0,
                  "Generated-token output must be a nonnegative scalar");
    progress_.generated_tokens = SafeInt<int64_t>(progress_.generated_tokens) + 1;
  }
  progress_.iterations = SafeInt<int64_t>(progress_.iterations) + 1;
  EvaluateExits(snapshots);
  feedback_.clear();
  for (const auto& item : loop_->options_.feedback) {
    feedback_.push_back(snapshots[FindOutput(loop_->fetch_names_, item.output_name)]);
  }
  SessionLoopStepResult result = progress_;
  result.executed = true;
  for (const auto& name : loop_->options_.output_names) {
    result.outputs.push_back(snapshots[FindOutput(loop_->fetch_names_, name)]);
  }
  requires_reset_ = false;
  out = std::move(result);
  return Status::OK();
}

Status SessionLoopExecution::Run(SessionLoopStepResult& out) {
  SessionLoopStepResult result;
  do {
    ORT_RETURN_IF_ERROR(Step(result));
  } while (!result.completed);
  out = std::move(result);
  return Status::OK();
}

void SessionLoopExecution::Reset() {
  feedback_.clear();
  progress_ = {};
  cancelled_.store(false);
  requires_reset_ = false;
}

}  // namespace onnxruntime
