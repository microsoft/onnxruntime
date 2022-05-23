// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>
#include <onnxruntime_cxx_api.h>

#include "cxxopts.hpp"
#include "core/util/math.h"
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include "core/session/environment.h"
#include "core/session/inference_session.h"
#include "core/providers/cpu/cpu_provider_factory_creator.h"
#include "orttraining/core/framework/tensorboard/event_writer.h"
#include "orttraining/training_api/include/utils.h"
#include "orttraining/training_api/include/interfaces.h"

#include "orttraining/test/training_api/synthetic_data_loader.h"

using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace onnxruntime::training;
using namespace onnxruntime::training::tensorboard;
using namespace onnxruntime::training::api;
using namespace std;

#ifdef USE_CUDA
namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(const OrtCUDAProviderOptions* provider_options);
std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name);

}  // namespace onnxruntime
#endif

struct TestRunnerParameters {
  PathString model_training_graph_path;
  std::optional<PathString> model_evaluation_graph_path;
  PathString optimizer_training_graph_path;
  // path to checkpoint to load
  PathString checkpoint_to_load_path;
  std::string model_name;

  PathString train_data_dir;
  PathString test_data_dir;
  PathString output_dir;  // Output of training, e.g., trained model files.

  size_t train_batch_size;
  size_t num_train_epochs;
  size_t eval_batch_size;
  size_t eval_interval;
  size_t checkpoint_interval;
  int gradient_accumulation_steps = 1;

  // Allocator to use for allocating inputs from the dataset (optional).
  AllocatorPtr input_allocator;
};

struct OrtTestRunnerParameters {
  logging::Severity log_severity{logging::Severity::kWARNING};
  int vlog_level{-1};
};

Status ParseArguments(int argc, char* argv[], TestRunnerParameters& params, OrtTestRunnerParameters& ort_params) {
  cxxopts::Options options("Training API Test", "Main Program to test training C++ APIs.");
  // clang-format off
  options
    .add_options()
      ("model_training_graph_path", "The path to the training model to load. ",
        cxxopts::value<std::string>()->default_value(""))
      ("model_evaluation_graph_path", "The path to the evaluation model to load. ",
        cxxopts::value<std::string>()->default_value(""))
      ("optimizer_training_graph_path", "The path to the optimizer graph to load. ",
        cxxopts::value<std::string>()->default_value(""))
      ("checkpoint_to_load_path", "The path to the checkpoint to load if provided.",
        cxxopts::value<std::string>()->default_value(""))
      ("model_name",
       "The name of the model.",
        cxxopts::value<std::string>()->default_value("model_test"))

      ("train_data_dir", "Input ONNX example files (can be a glob or comma separated).",
        cxxopts::value<std::string>()->default_value("bert_data/128/books_wiki_en_corpus/train"))
      ("test_data_dir", "Input ONNX example files (can be a glob or comma separated).",
        cxxopts::value<std::string>()->default_value("bert_data/128/books_wiki_en_corpus/test"))
      ("output_dir", "The output directory where the trained model files will be written.",
        cxxopts::value<std::string>()->default_value(""))

      ("train_batch_size", "Total batch size for training.", cxxopts::value<int>())
      ("eval_batch_size", "Total batch size for eval.", cxxopts::value<int>())
      ("num_train_epochs", "Total number of training epochs to perform.", cxxopts::value<int>()->default_value("100"))
      ("eval_interval", "Number of training steps before doing evaluation.", cxxopts::value<int>()->default_value("1000"))
      ("checkpoint_interval", "Number of training steps before saving checkpoint.", cxxopts::value<int>()->default_value("1000"))
      ("gradient_accumulation_steps", "The number of gradient accumulation steps before performing a backward/update pass.",
        cxxopts::value<int>()->default_value("1"));

  options
    .add_options("ORT configuration")
      ("ort_log_severity", "ORT minimum logging severity (see onnxruntime::logging::Severity values)",
        cxxopts::value<int>()->default_value("2"/*logging::Severity::kWARNING*/))
      ("ort_vlog_level", "ORT maximum VLOG level (verbose debug logging)",
        cxxopts::value<int>()->default_value("-1"));
  // clang-format on

  try {
    auto flags = options.parse(argc, argv);

    params.model_training_graph_path = ToPathString(flags["model_training_graph_path"].as<std::string>());
    std::string eval_path = flags["model_evaluation_graph_path"].as<std::string>();
    if (eval_path.empty()) {
      params.model_evaluation_graph_path = std::nullopt;
    } else {
      params.model_evaluation_graph_path = ToPathString(eval_path);
    }

    params.optimizer_training_graph_path = ToPathString(flags["optimizer_training_graph_path"].as<std::string>());
    params.checkpoint_to_load_path = ToPathString(flags["checkpoint_to_load_path"].as<std::string>());
    params.model_name = flags["model_name"].as<std::string>();

    params.train_batch_size = flags["train_batch_size"].as<int>();
    if (flags.count("eval_batch_size")) {
      params.eval_batch_size = flags["eval_batch_size"].as<int>();
    } else {
      params.eval_batch_size = params.train_batch_size;
    }
    params.num_train_epochs = flags["num_train_epochs"].as<int>();
    params.eval_interval = flags["eval_interval"].as<int>();
    params.checkpoint_interval = flags["checkpoint_interval"].as<int>();

    params.gradient_accumulation_steps = flags["gradient_accumulation_steps"].as<int>();
    if (params.gradient_accumulation_steps < 1) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT,
                    "Invalid gradient_accumulation_steps parameter: should be >= 1");
    }

    params.train_data_dir = ToPathString(flags["train_data_dir"].as<std::string>());
    params.test_data_dir = ToPathString(flags["test_data_dir"].as<std::string>());
    params.output_dir = ToPathString(flags["output_dir"].as<std::string>());
    if (params.output_dir.empty()) {
      printf("No output directory specified. Trained model files will not be saved.\n");
    }

    ort_params.log_severity = static_cast<logging::Severity>(flags["ort_log_severity"].as<int>());
    ORT_RETURN_IF_NOT(
        logging::Severity::kVERBOSE <= ort_params.log_severity && ort_params.log_severity <= logging::Severity::kFATAL,
        "Log severity must be in the range [", static_cast<int>(logging::Severity::kVERBOSE),
        ", ", static_cast<int>(logging::Severity::kFATAL), "].");
    ort_params.vlog_level = flags["ort_vlog_level"].as<int>();
  } catch (const exception& e) {
    const std::string msg = "Failed to parse the command line arguments";
    cerr << msg << ": " << e.what() << "\n"
         << options.help() << "\n";
    return Status(ONNXRUNTIME, INVALID_ARGUMENT, msg);
  }

  return Status::OK();
}

Status RunTraining(const TestRunnerParameters& params) {
  std::string tensorboard_file = params.output_dir + "/tb.event";
  std::shared_ptr<EventWriter> tensorboard = std::make_shared<EventWriter>(tensorboard_file);

  CheckpointState state;
  ORT_ENFORCE(LoadCheckpoint(params.checkpoint_to_load_path, state).IsOK());

#ifdef USE_CUDA
  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  const auto& api = Ort::GetApi();
  ORT_ENFORCE(api.CreateCUDAProviderOptions(&cuda_options) == nullptr);

  // MUST set execution provider before model/optimizer creation.
  SetExecutionProvider(cuda_options);
#endif

  Module module(params.model_training_graph_path,
                state.module_checkpoint_state.named_parameters,
                params.model_evaluation_graph_path);

  bool do_eval = params.model_evaluation_graph_path.has_value();

  Optimizer optimizer(params.optimizer_training_graph_path,
                      state.module_checkpoint_state.named_parameters);

  size_t sample_count_per_epoch = 4;
  ::test::training_api::SyntheticDataLoader data_loader(sample_count_per_epoch, params.train_batch_size);

  int64_t total_step_count = static_cast<int64_t>(params.num_train_epochs * data_loader.NumOfBatches());
  int64_t warmup_step_count = total_step_count / 3;
  LinearLRScheduler scheduler = LinearLRScheduler(optimizer, warmup_step_count, total_step_count);
  std::string tag("train");

  const size_t stabilized_perf_start_step = 0;
  double stabilized_total_end_to_end_time{0};
  auto end_to_end_start = std::chrono::high_resolution_clock::now();

  for (size_t epoch = 0, batch_idx = 0; epoch < params.num_train_epochs; ++epoch) {
    // for (auto it = data_loader.begin(); it != data_loader.end(); ++it) {
    if (batch_idx >= stabilized_perf_start_step) {
      end_to_end_start = std::chrono::high_resolution_clock::now();
    }

    std::vector<Ort::Value> inputs;
    data_loader.GetNextBatch(inputs);

    std::vector<Ort::Value> fetches;
    ORT_ENFORCE(module.TrainStep(inputs, fetches).IsOK());

    float loss = *(fetches[0].GetTensorMutableData<float>());
    tensorboard->AddSummary(std::to_string(loss), batch_idx, tag);
    std::cout << "Batch # : " << batch_idx << " Loss: " << loss << std::endl;

    if ((batch_idx + 1) % params.gradient_accumulation_steps == 0) {
      // Gradient accumulation steps completed.
      ORT_ENFORCE(optimizer.Step().IsOK());
      // Update learning rate.
      ORT_ENFORCE(scheduler.Step().IsOK());
      ORT_ENFORCE(module.ResetGrad().IsOK());
    }

    if (do_eval && (batch_idx + 1) % params.eval_interval == 0) {
      std::vector<Ort::Value> eval_results;
      ORT_ENFORCE(module.EvalStep(inputs, eval_results).IsOK());
    }

    if ((batch_idx + 1) % params.checkpoint_interval == 0) {
      // Save trained weights
      CheckpointState state_to_save;
      ORT_ENFORCE(module.GetStateDict(state_to_save.module_checkpoint_state).IsOK());
      ORT_ENFORCE(optimizer.GetStateDict(state_to_save.optimizer_checkpoint_state).IsOK());
      state_to_save.property_bag.AddProperty<int64_t>(std::string("epoch"), static_cast<int64_t>(epoch));
      std::string ckpt_file = params.output_dir + "/ckpt_" + params.model_name + std::to_string(batch_idx);
      ORT_ENFORCE(SaveCheckpoint(state_to_save, ckpt_file).IsOK());
    }

    batch_idx++;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration_seconds = end - end_to_end_start;
  stabilized_total_end_to_end_time = duration_seconds.count();

  std::cout << "Training completed - end to end latency: " << stabilized_total_end_to_end_time << "(s)" << std::endl;

#ifdef USE_CUDA
  // Finally, don't forget to release the provider options
  api.ReleaseCUDAProviderOptions(cuda_options);
#endif

  return Status::OK();
}

#define RETURN_IF_FAIL(expr)                                \
  do {                                                      \
    auto status = (expr);                                   \
    if ((!status.IsOK())) {                                 \
      printf("Fail: %s \n", status.ErrorMessage().c_str()); \
      return -1;                                            \
    }                                                       \
  } while (0);

int main(int argc, char* argv[]) {
  TestRunnerParameters params;
  OrtTestRunnerParameters ort_params{};
  RETURN_IF_FAIL(ParseArguments(argc, argv, params, ort_params));

  // setup logger, be noted: LOGS_DEFAULT must be after logging manager initialization.
  string default_logger_id{"Default"};
  logging::LoggingManager default_logging_manager{std::make_unique<logging::CLogSink>(),
                                                  ort_params.log_severity,
                                                  false,
                                                  logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id,
                                                  ort_params.vlog_level};
#ifdef USE_CUDA
  OrtCUDAProviderOptions provider_options{};
  params.input_allocator = CreateCUDAPinnedAllocator(provider_options.device_id, CUDA_PINNED);
#endif

  // start training session
  RETURN_IF_FAIL(RunTraining(params));
  return 0;
}
