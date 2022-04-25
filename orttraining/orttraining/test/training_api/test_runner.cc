// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
#include "orttraining/training_api/interfaces.h"
#include "orttraining/training_api/utils.h"

using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace onnxruntime::training;
using namespace onnxruntime::training::tensorboard;
using namespace onnxruntime::training::api_test;
using namespace std;

#ifdef USE_CUDA
namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(const OrtCUDAProviderOptions* provider_options);
std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name);

}  // namespace onnxruntime
#endif

static SessionOptions session_options;

struct TestRunnerParameters {
  PathString model_training_graph_path;
  PathString model_evaluation_graph_path;
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
  std::unique_ptr<IExecutionProvider> provider;
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
      ("checkpoint_to_load_path",
       "The path to the checkpoint to load. If not provided, the latest "
       "checkpoint in checkpoints_dir, if any, is used.",
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
    params.model_evaluation_graph_path = ToPathString(flags["model_evaluation_graph_path"].as<std::string>());
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

    session_options.use_deterministic_compute = flags["use_deterministic_compute"].as<bool>();

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

template <typename T>
static void CreateInputOrtValue(gsl::span<const int64_t> dims,
                                const std::vector<T>& value,
                                OrtValue* p_ortvalue,
                                AllocatorPtr alloc = nullptr) {
  static CPUExecutionProviderInfo info;
  static CPUExecutionProvider cpu_provider(info);
  static AllocatorPtr cpu_allocator = cpu_provider.GetAllocator(0, OrtMemTypeDefault);

  TensorShape shape(dims);
  assert(shape.Size() == static_cast<int64_t>(value.size()));
  auto element_type = DataTypeImpl::GetType<T>();
  auto allocator = alloc ? alloc : cpu_allocator;
  auto p_tensor = std::make_unique<Tensor>(element_type, shape, allocator);

  if (value.size() > 0) {
    memcpy(p_tensor->MutableDataRaw(), value.data(), p_tensor->SizeInBytes());
  }

  p_ortvalue->Init(p_tensor.release(),
                   DataTypeImpl::GetType<Tensor>(),
                   DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
}

std::vector<std::vector<OrtValue>> CreateSyntheticDataLoader(size_t batch_size,
                                                             AllocatorPtr alloc = nullptr) {
  OrtValue input, positions;
  // hard coded each sample to have 4 elements so far.
  // todo: we can make it support more generic once we are clear what our offline process graph needed.
  CreateInputOrtValue(std::array<int64_t, 4>{4}, std::vector<int64_t>{1, 2, 3, 4}, &input, alloc = alloc);
  CreateInputOrtValue(std::array<int64_t, 4>{4}, std::vector<int64_t>{1, 2, 3, 3}, &positions, alloc = alloc);
  return std::vector<std::vector<OrtValue>>(batch_size, std::vector<OrtValue>{input, positions});
}

float GetLossValue(OrtValue& ort_value) {
  const Tensor& loss_tensor = ort_value.Get<Tensor>();
  float loss = 0;
  if (DataTypeImpl::GetType<float>() == loss_tensor.DataType()) {
    loss = *(loss_tensor.template Data<float>());
  } else {
    ORT_THROW("loss data type not supported.");
  }
  return loss;
}

Status RunTraining(const TestRunnerParameters& params) {
  std::string tensorboard_file = params.output_dir + "/tb.event";
  std::shared_ptr<EventWriter> tensorboard = std::make_shared<EventWriter>(tensorboard_file);

  api_test::utils::CheckpointStates state_dicts;
  ORT_ENFORCE(api_test::utils::Ort_Load(params.checkpoint_to_load_path, state_dicts).IsOK());

  Module module(params.model_training_graph_path,
                state_dicts.named_parameters,
                params.model_evaluation_graph_path);

  Optimizer optimizer(params.optimizer_training_graph_path,
                      state_dicts.named_parameters);

#ifdef USE_CUDA
  api_test::utils::SetExecutionProvider(module, optimizer, params.provider.get());
#endif

  auto scheduler = std::make_unique<LinearScheduler>(optimizer, 0.3333f, 1.0f, 5);
  std::vector<std::vector<OrtValue>>
      data_loader = CreateSyntheticDataLoader(params.train_batch_size,
                                              params.input_allocator);

  size_t NUM_EPOCHS = params.num_train_epochs;
  size_t GRAD_ACC_STEPS = params.gradient_accumulation_steps;
  size_t EVAL_STEPS = params.eval_interval;
  size_t SAVE_STEPS = params.checkpoint_interval;
  std::string tag("train");

  for (size_t epoch = 0, batch_idx = 0; epoch < NUM_EPOCHS; ++epoch) {
    for (auto it = data_loader.begin(); it != data_loader.end(); ++it) {
      std::vector<OrtValue>& inputs = *it;
      std::vector<OrtValue> fetches;
      ORT_ENFORCE(module.TrainStep(inputs, fetches).IsOK());

      float loss = GetLossValue(fetches[3]);
      tensorboard->AddSummary(std::to_string(loss), batch_idx, tag);
      std::cout << "Batch # : " << batch_idx << " Loss: " << loss << std::endl;

      if (batch_idx % GRAD_ACC_STEPS == 0) {
        // gradient accumulation steps completed
        ORT_ENFORCE(optimizer.Step().IsOK());
        // modify learning rate
        ORT_ENFORCE(scheduler->Step().IsOK());
        ORT_ENFORCE(optimizer.ResetGrad().IsOK());
      }

      if (batch_idx % EVAL_STEPS == 0) {
        std::vector<OrtValue> eval_results;
        ORT_ENFORCE(module.EvalStep(inputs, eval_results).IsOK());
      }

      if (batch_idx % SAVE_STEPS == 0) {
        // save trained weights
        api_test::utils::CheckpointStates state_dicts_to_save;
        ORT_ENFORCE(module.GetStateDict(state_dicts_to_save.named_parameters).IsOK());
        ORT_ENFORCE(optimizer.GetStateDict(state_dicts_to_save.optimizer_states).IsOK());
        std::string ckpt_file = params.output_dir + "/ckpt_" + params.model_name + std::to_string(batch_idx);
        ORT_ENFORCE(api_test::utils::Ort_Save(state_dicts_to_save, ckpt_file).IsOK());
      }

      batch_idx++;
    }
  }

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
  if (auto factory = CreateExecutionProviderFactory_Cuda(&provider_options))
    params.provider = std::move(factory->CreateProvider());

  params.input_allocator = CreateCUDAPinnedAllocator(provider_options.device_id, CUDA_PINNED);
#endif

  // start training session
  RETURN_IF_FAIL(RunTraining(params));
  return 0;
}
