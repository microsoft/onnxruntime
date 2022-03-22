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
#include "core/framework/bfc_arena.h"
#include "core/providers/cpu/cpu_provider_factory_creator.h"
#include "orttraining/core/framework/tensorboard/event_writer.h"
#include "orttraining/training_api/interfaces.h"

using namespace onnxruntime;
using namespace onnxruntime::common;
using namespace onnxruntime::training;
using namespace onnxruntime::training::tensorboard;
using namespace std;

#ifdef USE_CUDA
namespace onnxruntime {

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_Cuda(const OrtCUDAProviderOptions* provider_options);
std::unique_ptr<IAllocator> CreateCUDAPinnedAllocator(int16_t device_id, const char* name);

}  // namespace onnxruntime
#endif

static SessionOptions session_options = {
    ExecutionMode::ORT_SEQUENTIAL,     // execution_mode
    ExecutionOrder::PRIORITY_BASED,    // execution_order
    false,                             // enable_profiling
    ORT_TSTR(""),                      // optimized_model_filepath
    true,                              // enable_mem_pattern
    true,                              // enable_mem_reuse
    true,                              // enable_cpu_mem_arena
    ORT_TSTR("onnxruntime_profile_"),  // profile_file_prefix
    "",                                // session_logid
    -1,                                // session_log_severity_level
    0,                                 // session_log_verbosity_level
    5,                                 // max_num_graph_transformation_steps
    TransformerLevel::Level1,          // graph_optimization_level
    {},                                // intra_op_param
    {},                                // inter_op_param
    {},                                // free_dimension_overrides
    true,                              // use_per_session_threads
    true,                              // thread_pool_allow_spinning
    false,                             // use_deterministic_compute
    {},                                // config_options
    {},                                // initializers_to_share_map
};

namespace onnxruntime {
namespace training {
namespace api_test {

struct Parameters {
  PathString model_with_training_graph_path;
  // path to checkpoint to load
  PathString checkpoint_to_load_path;
  PathString train_data_dir;
  PathString test_data_dir;
  PathString output_dir;  // Output of training, e.g., trained model files.

  size_t train_batch_size;
  size_t eval_batch_size;
  size_t num_train_steps;
  int gradient_accumulation_steps = 1;

  // Enable gradient clipping.
  bool enable_grad_norm_clip = true;

  // Allocator to use for allocating inputs from the dataset (optional).
  AllocatorPtr input_allocator;
  std::unique_ptr<IExecutionProvider> provider;
};

struct OrtParameters {
  logging::Severity log_severity{logging::Severity::kWARNING};
  int vlog_level{-1};
};

Status ParseArguments(int argc, char* argv[], Parameters& params, OrtParameters& ort_params) {
  cxxopts::Options options("Training API Test", "Main Program to test training C++ APIs.");
  // clang-format off
  options
    .add_options()
      ("model_with_training_graph_path", "The path to the model to load. ",
        cxxopts::value<std::string>()->default_value(""))
      ("checkpoint_to_load_path",
       "The path to the checkpoint to load. If not provided, the latest "
       "checkpoint in checkpoints_dir, if any, is used.",
        cxxopts::value<std::string>()->default_value(""))

      ("train_data_dir", "Input ONNX example files (can be a glob or comma separated).",
        cxxopts::value<std::string>()->default_value("bert_data/128/books_wiki_en_corpus/train"))
      ("test_data_dir", "Input ONNX example files (can be a glob or comma separated).",
        cxxopts::value<std::string>()->default_value("bert_data/128/books_wiki_en_corpus/test"))
      ("output_dir", "The output directory where the trained model files will be written.",
        cxxopts::value<std::string>()->default_value(""))

      ("train_batch_size", "Total batch size for training.", cxxopts::value<int>())
      ("eval_batch_size", "Total batch size for eval.", cxxopts::value<int>())
      ("num_train_steps", "Total number of training steps to perform.", cxxopts::value<int>()->default_value("100000"))
      ("gradient_accumulation_steps", "The number of gradient accumulation steps before performing a backward/update pass.",
        cxxopts::value<int>()->default_value("1"))

      ("enable_grad_norm_clip", "Specify whether to enable gradient clipping for optimizers.",
        cxxopts::value<bool>()->default_value("true"));

  options
    .add_options("ORT configuration")
      ("ort_log_severity", "ORT minimum logging severity (see onnxruntime::logging::Severity values)",
        cxxopts::value<int>()->default_value("2"/*logging::Severity::kWARNING*/))
      ("ort_vlog_level", "ORT maximum VLOG level (verbose debug logging)",
        cxxopts::value<int>()->default_value("-1"));
  // clang-format on

  try {
    auto flags = options.parse(argc, argv);

    params.num_train_steps = flags["num_train_steps"].as<int>();
    params.train_batch_size = flags["train_batch_size"].as<int>();
    if (flags.count("eval_batch_size")) {
      params.eval_batch_size = flags["eval_batch_size"].as<int>();
    } else {
      params.eval_batch_size = params.train_batch_size;
    }

    params.gradient_accumulation_steps = flags["gradient_accumulation_steps"].as<int>();
    if (params.gradient_accumulation_steps < 1) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT, "Invalid gradient_accumulation_steps parameter: should be >= 1");
    }

    params.train_data_dir = ToPathString(flags["train_data_dir"].as<std::string>());
    params.test_data_dir = ToPathString(flags["test_data_dir"].as<std::string>());
    params.output_dir = ToPathString(flags["output_dir"].as<std::string>());
    if (params.output_dir.empty()) {
      printf("No output directory specified. Trained model files will not be saved.\n");
    }
    params.checkpoint_to_load_path = ToPathString(flags["checkpoint_to_load_path"].as<std::string>());

    params.enable_grad_norm_clip = flags["enable_grad_norm_clip"].as<bool>();
    session_options.use_deterministic_compute = flags["use_deterministic_compute"].as<bool>();

    ort_params.log_severity = static_cast<logging::Severity>(flags["ort_log_severity"].as<int>());
    ORT_RETURN_IF_NOT(
        logging::Severity::kVERBOSE <= ort_params.log_severity &&
            ort_params.log_severity <= logging::Severity::kFATAL,
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

std::vector<std::vector<OrtValue>> CreateTestDataLoader(AllocatorPtr alloc = nullptr) {
  OrtValue input, positions;
  // hard code batch size = 4.
  CreateInputOrtValue(std::array<int64_t, 4>{4}, std::vector<int64_t>{1, 2, 3, 4}, &input, alloc = alloc);
  CreateInputOrtValue(std::array<int64_t, 4>{4}, std::vector<int64_t>{1, 2, 3, 3}, &positions, alloc = alloc);
  return std::vector<std::vector<OrtValue>>(20, std::vector<OrtValue>{input, positions});
}

float GetLossValue(const Tensor& loss_tensor) {
  float loss = 0;
  if (DataTypeImpl::GetType<float>() == loss_tensor.DataType()) {
    loss = *(loss_tensor.template Data<float>());
  } else if (DataTypeImpl::GetType<MLFloat16>() == loss_tensor.DataType()) {
    loss = math::halfToFloat(loss_tensor.template Data<MLFloat16>()->val);
  }
  return loss;
}

Status RunTraining(const Parameters& params) {
  std::string tensorboard_file = params.output_dir + "/tb.event";
  std::shared_ptr<EventWriter> tensorboard = std::make_shared<EventWriter>(tensorboard_file);

  utils::CheckpointStates state_dicts;
  ORT_ENFORCE(utils::Ort_Load("resnet50_0.ckpt", state_dicts).IsOK());

  Module module("train_resnet50.onnx", state_dicts.named_parameters, "eval_resnet50.onnx");
  Optimizer optimizer("adam_resnet50.onnx", state_dicts.named_parameters);

#ifdef USE_CUDA
  utils::SetExecutionProvider(module, optimizer, params.provider.get());
#endif

  auto scheduler = std::make_unique<LinearScheduler>(optimizer, 0.3333f, 1.0f, 5);

  std::vector<std::vector<OrtValue>> data_loader = CreateTestDataLoader(params.input_allocator);

  size_t NUM_EPOCHS = 10;
  size_t GRAD_ACC_STEPS = 2;
  size_t EVAL_STEPS = 20;
  size_t SAVE_STEPS = 20;
  std::string tag("train");

  for (size_t epoch = 0, batch_idx = 0; epoch < NUM_EPOCHS; ++epoch) {
    for (auto it = data_loader.begin(); it != data_loader.end(); ++it) {
      std::vector<OrtValue>& inputs = *it;
      std::vector<OrtValue> fetches;
      ORT_ENFORCE(module.TrainStep(inputs, fetches).IsOK());

      const Tensor& loss_tensor = fetches[3].Get<Tensor>();
      tensorboard->AddSummary(*(loss_tensor.template Data<std::string>()), batch_idx, tag);

      std::cout << "Batch # : " << batch_idx << " Loss: " << GetLossValue(loss_tensor) << std::endl;

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
        utils::CheckpointStates state_dicts_to_save;
        ORT_ENFORCE(module.GetStateDict(state_dicts_to_save.named_parameters).IsOK());
        ORT_ENFORCE(optimizer.GetStateDict(state_dicts_to_save.optimizer_states).IsOK());
        ORT_ENFORCE(utils::Ort_Save(state_dicts_to_save, "resnet50_2000.ckpt").IsOK());
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
  Parameters params;
  OrtParameters ort_params{};
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
  provider_options.do_copy_in_default_stream = true;
  if (auto factory = CreateExecutionProviderFactory_Cuda(&provider_options))
    params.provider = std::move(factory->CreateProvider());

  params.input_allocator = CreateCUDAPinnedAllocator(provider_options.device_id, CUDA_PINNED);
#endif

  // start training session
  RETURN_IF_FAIL(RunTraining(params));
  return 0;
}
}  // namespace api_test
}  // namespace training
}  // namespace onnxruntime