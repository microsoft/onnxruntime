// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "onnxruntime_c_api.h"
#include "onnxruntime_training_c_api.h"
#include "onnxruntime_training_cxx_api.h"

#include "cxxopts.hpp"
#include "core/common/path_string.h"
#include "core/platform/path_lib.h"
#include "../common/synthetic_data_loader.h"

#if defined(USE_CUDA) && defined(ENABLE_NVTX_PROFILE)
// This header is for profile using Nvidia's visual profiler.
#include "core/providers/cuda/nvtx_profile.h"
#include "core/providers/cuda/nvtx_profile_context.h"
#endif

using namespace onnxruntime;
using namespace std;

const OrtApi* g_ort_api = nullptr;
const OrtTrainingApi* g_ort_training_api = nullptr;

struct TestRunnerParameters {
  // Models configs.
  PathString model_training_graph_path;
  std::optional<PathString> model_evaluation_graph_path;
  PathString optimizer_training_graph_path;
  PathString checkpoint_to_load_path;
  std::string model_name;
  std::string synthetic_input_type;

  // Data configs.
  PathString train_data_dir;
  PathString test_data_dir;
  PathString output_dir;  // Output of training, e.g., trained model files.

  // Training configs.
  int64_t train_batch_size;
  int64_t num_train_epochs;
  int64_t eval_batch_size;
  int64_t eval_interval;
  int64_t checkpoint_interval;
  int64_t gradient_accumulation_steps = 1;
};

void EnforceCheck(bool run_ret, std::string err_msg) {
  if (!run_ret) {
    throw std::runtime_error("EnforceCheck failed: " + err_msg);
  }
}

bool ParseArguments(int argc, char* argv[], TestRunnerParameters& params) {
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
      ("model_name", "The name of the model.",
        cxxopts::value<std::string>()->default_value("model_test"))
      ("synthetic_input_type", "Input type can be 'dummy' for test model, 'S', 'U', 'R' which represent some internal"
      " models.",
        cxxopts::value<std::string>()->default_value("attention"))

      ("train_data_dir", "Input ONNX example files (can be a glob or comma separated).",
        cxxopts::value<std::string>()->default_value("bert_data/128/books_wiki_en_corpus/train"))
      ("test_data_dir", "Input ONNX example files (can be a glob or comma separated).",
        cxxopts::value<std::string>()->default_value("bert_data/128/books_wiki_en_corpus/test"))
      ("output_dir", "The output directory where the trained model files will be written.",
        cxxopts::value<std::string>()->default_value(""))

      ("train_batch_size", "Total batch size for training.", cxxopts::value<int>())
      ("eval_batch_size", "Total batch size for eval.", cxxopts::value<int>())
      ("num_train_epochs", "Total number of training epochs to perform.", cxxopts::value<int>()->default_value("100"))
      ("eval_interval", "Number of training steps before doing evaluation.", cxxopts::value<int>()->
      default_value("1000"))
      ("checkpoint_interval", "Number of training steps before saving checkpoint.", cxxopts::value<int>()->
      default_value("1000"))
      ("gradient_accumulation_steps", "The number of gradient accumulation steps before performing a "
      "backward/update pass.",
        cxxopts::value<int>()->default_value("1"));

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
    params.synthetic_input_type = flags["synthetic_input_type"].as<std::string>();

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
    EnforceCheck(params.gradient_accumulation_steps >= 1,
                 "Invalid gradient_accumulation_steps parameter: should be >= 1");

    params.train_data_dir = ToPathString(flags["train_data_dir"].as<std::string>());
    params.test_data_dir = ToPathString(flags["test_data_dir"].as<std::string>());
    params.output_dir = ToPathString(flags["output_dir"].as<std::string>());
    if (params.output_dir.empty()) {
      printf("No output directory specified. Trained model files will not be saved.\n");
    }
  } catch (const std::exception& e) {
    const std::string msg = "Failed to parse the command line arguments";
    std::cerr << msg << ": " << e.what() << "\n"
              << options.help() << "\n";
    return false;
  }

  return true;
}

void InitSyntheticDataLoader(
    onnxruntime::training::test::training_api::SyntheticDataLoader& data_loader,
    const TestRunnerParameters& params,
    int64_t num_of_batches_per_epoch) {
  if (params.synthetic_input_type == "dummy") {
    std::vector<int64_t> input1_shape{params.train_batch_size, 784};
    std::vector<int64_t> target_shape{params.train_batch_size};
    for (int64_t i = 0; i < num_of_batches_per_epoch; ++i) {
      auto sample = onnxruntime::training::test::training_api::SyntheticSampleBatch();
      sample.AddFloatInput(input1_shape);
      sample.AddInt32Input(target_shape, 0, 1);
      data_loader.AddSyntheticSampleBatch(std::move(sample));
    }
  } else if (params.synthetic_input_type == "S") {
    int64_t sequence_length = 128;
    std::vector<int64_t> input_ids_shape{params.train_batch_size, sequence_length};
    std::vector<int64_t> attention_mask_shape{params.train_batch_size, sequence_length};
    std::vector<int64_t> target_shape{params.train_batch_size};
    for (int64_t i = 0; i < num_of_batches_per_epoch; ++i) {
      auto sample = onnxruntime::training::test::training_api::SyntheticSampleBatch();
      sample.AddInt64Input(input_ids_shape, 0, 250002 - 1);
      sample.AddInt64Input(attention_mask_shape, 0, 1);
      sample.AddInt32Input(target_shape, 0, 1);
      data_loader.AddSyntheticSampleBatch(std::move(sample));
    }
  } else if (params.synthetic_input_type == "U") {
    int64_t sequence_length = 128;
    std::vector<int64_t> input_ids_shape{params.train_batch_size, sequence_length};
    std::vector<int64_t> attention_mask_shape{params.train_batch_size, sequence_length};
    std::vector<int64_t> target1_shape{params.train_batch_size};
    std::vector<int64_t> target2_shape{params.train_batch_size, 81};
    for (int64_t i = 0; i < num_of_batches_per_epoch; ++i) {
      auto sample = onnxruntime::training::test::training_api::SyntheticSampleBatch();
      sample.AddInt64Input(input_ids_shape, 0, 250002 - 1);
      sample.AddInt64Input(attention_mask_shape, 0, 1);
      sample.AddInt32Input(target1_shape, 0, 1);
      sample.AddInt32Input(target2_shape, 0, 1);
      data_loader.AddSyntheticSampleBatch(std::move(sample));
    }
  } else if (params.synthetic_input_type == "R") {
    int64_t sequence_length = 128;
    std::vector<int64_t> input_ids_shape{params.train_batch_size, sequence_length};
    std::vector<int64_t> attention_mask_shape{params.train_batch_size, sequence_length};
    std::vector<int64_t> labels_shape{params.train_batch_size, 7};
    for (int64_t i = 0; i < num_of_batches_per_epoch; ++i) {
      auto sample = onnxruntime::training::test::training_api::SyntheticSampleBatch();
      sample.AddInt64Input(input_ids_shape, 0, 250002 - 1);
      sample.AddInt64Input(attention_mask_shape, 0, 1);
      sample.AddInt32Input(labels_shape, 0, 1);
      data_loader.AddSyntheticSampleBatch(std::move(sample));
    }
  } else if (params.synthetic_input_type == "C") {
    int64_t section = 16;
    int64_t sequence_length = 128;
    std::vector<int64_t> input_ids_shape{params.train_batch_size, section, sequence_length};
    std::vector<int64_t> mask_clss_shape{params.train_batch_size, section};
    std::vector<int64_t> attention_mask_shape{params.train_batch_size, section, sequence_length};
    std::vector<int64_t> labels_shape{params.train_batch_size};
    for (int64_t i = 0; i < num_of_batches_per_epoch; ++i) {
      auto sample = onnxruntime::training::test::training_api::SyntheticSampleBatch();
      sample.AddInt64Input(input_ids_shape, 0, 250002 - 1);
      sample.AddBoolInput(mask_clss_shape);
      sample.AddInt64Input(attention_mask_shape, 0, 1);
      sample.AddInt32Input(labels_shape, 0, 1);
      data_loader.AddSyntheticSampleBatch(std::move(sample));
    }
  } else {
    throw std::runtime_error("unknown synthetic_input_type: " + params.synthetic_input_type);
  }
}

int RunTraining(const TestRunnerParameters& params) {
  g_ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  g_ort_training_api = g_ort_api->GetTrainingApi(ORT_API_VERSION);

  // TODO(askhade): enable global threadpool
  OrtThreadingOptions* threading_options = nullptr;
  Ort::ThrowOnError(g_ort_api->CreateThreadingOptions(&threading_options));
  // Create Env
  Ort::Env env(threading_options, ORT_LOGGING_LEVEL_WARNING, "log");
  g_ort_api->ReleaseThreadingOptions(threading_options);

  // Load Checkpoint State
  Ort::CheckpointState checkpoint_state = Ort::CheckpointState::LoadCheckpoint(params.checkpoint_to_load_path);

  // Create TrainingSession
  Ort::SessionOptions soptions;

#ifdef USE_CUDA
  OrtCUDAProviderOptionsV2* cuda_options = nullptr;
  Ort::ThrowOnError(g_ort_api->CreateCUDAProviderOptions(&cuda_options));
  std::unique_ptr<OrtCUDAProviderOptionsV2, decltype(g_ort_api->ReleaseCUDAProviderOptions)>
      rel_cuda_options(cuda_options, g_ort_api->ReleaseCUDAProviderOptions);
  soptions.AppendExecutionProvider_CUDA_V2(*rel_cuda_options);
#endif

  bool do_eval = params.model_evaluation_graph_path.has_value();
  Ort::TrainingSession session(env, soptions, checkpoint_state, params.model_training_graph_path,
                               params.model_evaluation_graph_path,
                               params.optimizer_training_graph_path.size() > 0
                                   ? std::optional<PathString>(params.optimizer_training_graph_path)
                                   : std::nullopt);

  int64_t sample_batch_count_per_epoch = params.train_batch_size;
  if (sample_batch_count_per_epoch < params.train_batch_size ||
      sample_batch_count_per_epoch % params.train_batch_size != 0) {
    throw std::runtime_error("sample_count cannot be divisible by batch_size");
  }
  int64_t num_of_batches_per_epoch = sample_batch_count_per_epoch / params.train_batch_size;

  onnxruntime::training::test::training_api::SyntheticDataLoader data_loader;
  InitSyntheticDataLoader(data_loader, params, num_of_batches_per_epoch);

  auto total_step_count = params.num_train_epochs * num_of_batches_per_epoch;
  auto warmup_step_count = total_step_count / 3;
  session.RegisterLinearLRScheduler(warmup_step_count, total_step_count, 0.001f);

  std::cout << "Initialization completed. Now starting training loop." << std::endl;
  constexpr int64_t stabilized_perf_start_step = 0;
  double stabilized_total_end_to_end_time{0};
  auto end_to_end_start = std::chrono::high_resolution_clock::now();

  for (int64_t epoch = 0, batch_idx = 0; epoch < params.num_train_epochs; ++epoch) {
    for (size_t step_in_cur_epoch = 0; step_in_cur_epoch < data_loader.NumOfSampleBatches(); ++step_in_cur_epoch) {
      if (batch_idx >= stabilized_perf_start_step) {
        end_to_end_start = std::chrono::high_resolution_clock::now();
      }

      std::vector<Ort::Value> inputs;
      data_loader.GetNextSampleBatch(inputs);

#if defined(USE_CUDA) && defined(ENABLE_NVTX_PROFILE)
      onnxruntime::profile::NvtxRangeCreator train_step_range(
          "module_TrainStep",
          onnxruntime::profile::Color::Green);
      train_step_range.Begin();
#endif

      auto fetches = session.TrainStep(inputs);
#if defined(USE_CUDA) && defined(ENABLE_NVTX_PROFILE)
      train_step_range.End();
#endif

      float* loss = fetches[0].GetTensorMutableData<float>();
      std::cout << "Batch # : " << batch_idx << " Loss: " << loss[0] << std::endl;

      if ((batch_idx + 1) % params.gradient_accumulation_steps == 0) {
        // Gradient accumulation steps completed.
#if defined(USE_CUDA) && defined(ENABLE_NVTX_PROFILE)
        onnxruntime::profile::NvtxRangeCreator opt_step_range(
            "opt_Step",
            onnxruntime::profile::Color::Blue);
        opt_step_range.Begin();
#endif
        session.OptimizerStep();

#if defined(USE_CUDA) && defined(ENABLE_NVTX_PROFILE)
        opt_step_range.End();
#endif

        // Update learning rate.
        session.SchedulerStep();

#if defined(USE_CUDA) && defined(ENABLE_NVTX_PROFILE)
        onnxruntime::profile::NvtxRangeCreator resetgrad_range(
            "LazyResetGrad",
            onnxruntime::profile::Color::Red);
        resetgrad_range.Begin();
#endif

        session.LazyResetGrad();

#if defined(USE_CUDA) && defined(ENABLE_NVTX_PROFILE)
        resetgrad_range.End();
#endif
      }

      if (do_eval && (batch_idx + 1) % params.eval_interval == 0) {
        auto eval_results = session.EvalStep(inputs);
      }

      if ((batch_idx + 1) % params.checkpoint_interval == 0) {
        // Save trained weights
        std::ostringstream oss;
        oss << "ckpt_" << params.model_name << std::to_string(batch_idx);
        PathString ckpt_file = ConcatPathComponent(params.output_dir, ToPathString(oss.str()));
        checkpoint_state.AddProperty("epoch", epoch);
        checkpoint_state.AddProperty("loss", *loss);
        checkpoint_state.AddProperty("framework", "onnxruntime");
        Ort::CheckpointState::SaveCheckpoint(checkpoint_state, ckpt_file);
      }
      batch_idx++;
    }

    data_loader.ResetIterateIndex();
  }

  // Save trained weights
  std::ostringstream oss;
  oss << "ckpt_" << params.model_name;
  PathString ckpt_file = ConcatPathComponent(params.output_dir, ToPathString(oss.str()));
  Ort::CheckpointState::SaveCheckpoint(checkpoint_state, ckpt_file);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration_seconds = end - end_to_end_start;
  stabilized_total_end_to_end_time = duration_seconds.count();

  std::cout << "Training completed - end to end latency: " << stabilized_total_end_to_end_time << "(s)" << std::endl;

  return 0;
}

int main(int argc, char* argv[]) {
  TestRunnerParameters params;
  EnforceCheck(ParseArguments(argc, argv, params), "Parse arguments failed.");

  // Start training session
  return RunTraining(params);
}
