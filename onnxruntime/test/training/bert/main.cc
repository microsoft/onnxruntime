// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cxxopts.hpp"
#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/session/environment.h"
#include "core/training/training_session.h"
#include "core/training/tensorboard/event_writer.h"
#include "test/training/runner/training_runner.h"
#include "test/training/runner/training_util.h"
#include "test/training/runner/data_loader.h"

#ifdef USE_HOROVOD
#include "core/graph/training/horovod_adapters.h"
#include <mpi.h>
#include <tuple>
#endif

using namespace onnxruntime;
using namespace onnxruntime::training;
using namespace onnxruntime::training::tensorboard;
using namespace std;

Status ParseArguments(int argc, char* argv[], TrainingRunner::Parameters& params) {
  cxxopts::Options options("BERT Training", "Main Program to train BERT");
  // clang-format off
  options
    .allow_unrecognised_options()
    .add_options()
      ("model_name", "model to be trained", cxxopts::value<std::string>())
      ("train_data_dir", "Input ONNX example files (can be a glob or comma separated).",
        cxxopts::value<std::string>()->default_value("bert_data/train"))
      ("test_data_dir", "Input ONNX example files (can be a glob or comma separated).",
        cxxopts::value<std::string>()->default_value("bert_data/test"))
      ("output_dir", "The output directory where the model checkpoints will be written.",
        cxxopts::value<std::string>())
      ("log_dir", "The directory to write tensorboard events.",
        cxxopts::value<std::string>()->default_value("logs/bert"))
      ("num_of_epoch", "Num of epoch", cxxopts::value<int>()->default_value("1"))
      ("train_batch_size", "Total batch size for training.", cxxopts::value<int>())
      ("eval_batch_size", "Total batch size for eval.", cxxopts::value<int>())
      ("learning_rate", "The initial learning rate for Adam.", cxxopts::value<float>()->default_value("5e-5"))
      ("num_train_steps", "Number of training steps.", cxxopts::value<int>()->default_value("100000"))
      ("num_warmup_steps", "Number of warmup steps.", cxxopts::value<int>()->default_value("10000"))
      ("evaluation_period",
        "How many training steps to make before making an evaluation.",
        cxxopts::value<size_t>()->default_value("100"))
      ("save_checkpoint_steps", "How often to save the model checkpoint.", cxxopts::value<int>()->default_value("1000"))
      ("iterations_per_loop", "How many steps to make in each estimator call.", cxxopts::value<int>()->default_value("1000"))
      ("max_eval_steps", "Maximum number of eval steps.", cxxopts::value<int>()->default_value("100"))
      ("use_fp16", "Whether to use fp32 or fp16 arithmetic on GPU.", cxxopts::value<bool>()->default_value("false"))
      ("mode", "mode for running, can be one of [train|perf]", cxxopts::value<std::string>()->default_value("train"))
      ("num_of_perf_samples", "Num of samples to run for the perf test", cxxopts::value<int>()->default_value("100"))
      ("max_seq_length",
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded. Must match data generation.", cxxopts::value<int>()->default_value("512"))
      ("max_predictions_per_seq",
        "Maximum number of masked LM predictions per sequence. "
        "Must match data generation.", cxxopts::value<int>()->default_value("80"));
  // clang-format on

  try {
    auto flags = options.parse(argc, argv);

    params.model_name = flags["model_name"].as<std::string>();
    params.learning_rate_ = flags["learning_rate"].as<float>();
    params.num_of_epoch_ = flags["num_of_epoch"].as<int>();
    params.num_of_perf_samples = flags["num_of_perf_samples"].as<int>();
    params.batch_size_ = flags["train_batch_size"].as<int>();
    if (flags.count("eval_batch_size")) {
      params.eval_batch_size = flags["eval_batch_size"].as<int>();
    } else {
      params.eval_batch_size = params.batch_size_;
    }
    params.evaluation_period = flags["evaluation_period"].as<size_t>();

    auto train_data_dir = flags["train_data_dir"].as<std::string>();
    auto test_data_dir = flags["test_data_dir"].as<std::string>();
    auto log_dir = flags["log_dir"].as<std::string>();
    params.train_data_dir.assign(train_data_dir.begin(), train_data_dir.end());
    params.test_data_dir.assign(test_data_dir.begin(), test_data_dir.end());
    params.log_dir.assign(log_dir.begin(), log_dir.end());

    std::string mode = flags["mode"].as<std::string>();
    if (mode == "perf" || mode == "train") {
      params.is_perf_test = mode == "perf";
    } else {
      printf("Incorrect command line for mode: it must be one of [perf|train]\n");
    }

  } catch (const exception& e) {
    std::string msg = "Failed to parse the command line arguments";
    cout << msg << e.what() << endl;
    return Status(ONNXRUNTIME, FAIL, msg);
  }
  return Status::OK();
}

#ifdef USE_HOROVOD
std::pair<int, int> setup_horovod() {
  using namespace horovod::common;
  // setup MPI amd horovod
  MPI_Init(0, 0);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int* ranks = (int*)malloc(sizeof(int) * world_size);

  MPI_Allgather(&world_rank, 1, MPI_INT, ranks, 1, MPI_INT, MPI_COMM_WORLD);

  horovod_init(ranks, world_size);

  return {world_rank, world_size};
}

void shutdown_horovod() {
  horovod::common::horovod_shutdown();
  MPI_Finalize();
}

#endif

// NOTE: these variables need to be alive when the error_function is called.
float total_loss = 0.0f;
float mlm_loss = 0.0f;
float nsp_loss = 0.0f;
std::vector<std::string> summary_loss;

void setup_training_params(TrainingRunner::Parameters& params) {
  params.model_path_ = params.model_name + ".onnx";
  params.model_with_loss_func_path_ = params.model_name + "_with_cost.onnx";
  params.model_with_training_graph_path_ = params.model_name + "_bw.onnx";
  params.model_actual_running_graph_path_ = params.model_name + "_bw_running.onnx";
  params.model_trained_path_ = params.model_name + "_trained.onnx";
  params.model_trained_with_loss_func_path_ = params.model_name + "_with_cost_trained.onnx";

  params.loss_func_info_ = LossFunctionInfo(OpDef("BertLoss", kOnnxDomain),
                                            "total_loss",
                                            {/*prediction_masked_lm*/ "output1",
                                             /*prediction_next_sentence*/ "output2",
                                             /*masked_lm_positions*/ "masked_lm_positions",
                                             /*masked_lm_ids*/ "masked_lm_ids",
                                             /*masked_lm_weights*/ "masked_lm_weights",
                                             /*next_sentence_labels*/ "next_sentence_labels",
                                             /*mlm_loss*/ "mlm_loss",
                                             /*nsp_loss*/ "nsp_loss",
                                             /*batch_size*/ std::to_string(params.batch_size_),
                                             /*max_sequence_len*/ std::to_string(512),
                                             /*max_predictions_per_sequence*/ std::to_string(80),
                                             /*summary_loss*/ "summary"});
  params.model_prediction_name_ = "output1";  //"output2";
  params.weights_not_to_train_ = {
      "position_01",            // Slice's dat input
      "op_min_ends_expand_10",  //op_min_ends_expand_10
  };
  params.fetch_names = {"total_loss", "mlm_loss", "nsp_loss", "summary"};

  params.immutable_weigths_ = {
      {"Div", {{1, 8.0f}, {1, 1.4142135381698608f}}},
      {"Add", {{1, 1.0f}, {1, 9.999999960041972e-13f}}},
      {"Mul", {{1, 0.5f}, {1, -10000.0f}}},
      {"Sub", {{0, 1.0f}}}};

  params.in_graph_optimizer_name_ = "AdamOptimizer";
  params.adam_opt_params_.alpha_ = 0.9f;
  params.adam_opt_params_.beta_ = 0.999f;
  params.adam_opt_params_.lambda_ = 0;
  params.adam_opt_params_.epsilon_ = 1e-6f;

  params.shuffle_data_ = false;

  // name_in_data_file -> name_in_model
  params.input_name_map_ = {
      {"input_ids", "input1"},
      {"segment_ids", "input2"},
      {"input_mask", "input3"},
      {"masked_lm_positions", "masked_lm_positions"},
      {"masked_lm_ids", "masked_lm_ids"},
      {"masked_lm_weights", "masked_lm_weights"},
      {"next_sentence_label", "next_sentence_labels"}};

  params.use_cuda_ = true;

  params.skip_evaluation_ = params.is_perf_test;

  params.error_function_ = [params](const std::vector<std::string>& /*feed_names*/,
                                    const std::vector<OrtValue>& /*feeds*/,
                                    const std::vector<std::string>& fetch_names,
                                    const std::vector<OrtValue>& fetches) {
    const Tensor& total_loss_t = fetches[0].Get<Tensor>();
    const Tensor& mlm_loss_t = fetches[1].Get<Tensor>();
    const Tensor& nsp_loss_t = fetches[2].Get<Tensor>();
    const Tensor& summary_loss_t = fetches[3].Get<Tensor>();

    const float* total_loss_val = total_loss_t.template Data<float>();
    const float* mlm_loss_val = mlm_loss_t.template Data<float>();
    const float* nsp_loss_val = nsp_loss_t.template Data<float>();
    const std::string* summary_loss_val = summary_loss_t.template Data<std::string>();

    total_loss += *total_loss_val;
    mlm_loss += *mlm_loss_val;
    nsp_loss += *nsp_loss_val;
    summary_loss.push_back(*summary_loss_val);

    if (params.dump_fetches) {
      ofstream ofs("fetches_dump.txt");
      for (size_t i = 0; i < fetch_names.size(); ++i) {
        TrainingUtil::PrintTensor(fetch_names[i], fetches[i].Get<Tensor>(), ofs);
      }
      ofs.close();
    }
  };

  auto tensorboard = std::make_shared<EventWriter>(params.log_dir);
  params.post_evaluation_callback_ = [tensorboard](size_t num_samples, size_t step) {
    float average_total_loss = total_loss / float(num_samples);
    float average_mlm_loss = mlm_loss / float(num_samples);
    float average_nsp_loss = nsp_loss / float(num_samples);

    for (const std::string& summary : summary_loss)
      tensorboard->AddSummary(summary, step);

    printf("Step: %zu, #examples: %d, total_loss: %0.04f, mlm_loss: %0.04f, nsp_loss: %0.04f \n\n",
           step,
           static_cast<int>(num_samples),
           average_total_loss,
           average_mlm_loss,
           average_nsp_loss);
    total_loss = 0.0f;
    mlm_loss = 0.0f;
    nsp_loss = 0.0f;
    summary_loss.clear();
  };
}

int main(int argc, char* argv[]) {
#ifndef USE_CUDA
  printf("BERT training is not supported in non-CUDA build. ");
#endif

  TrainingRunner::Parameters params;
  ParseArguments(argc, argv, params);
  setup_training_params(params);

  // setup logger
  string default_logger_id{"Default"};
  logging::LoggingManager default_logging_manager{unique_ptr<logging::ISink>{new logging::CLogSink{}},
                                                  logging::Severity::kWARNING,
                                                  false,
                                                  logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id};

  // setup onnxruntime env
  unique_ptr<Environment> env;
  ORT_ENFORCE(Environment::Create(env).IsOK());

  int device_id = 0, device_count = 1;

// setup horovod
#ifdef USE_HOROVOD
  std::tie(device_id, device_count) = setup_horovod();
#endif

  // TODO: This should be done in SGD optimizer. Will refactor when optimizing the kernel.
  // Adding another cuda kernel call for this division seems wasteful currently.
  // params.learning_rate_ = LEARNING_RATE / params.batch_size_;
  params.learning_rate_ = params.learning_rate_ / device_count;
  params.world_rank_ = device_id;
  params.world_size_ = device_count;

  printf("Using cuda device #%d, world_size %d \n", params.world_rank_, params.world_size_);

  // start training session
  std::unique_ptr<TrainingRunner> runner;
  if (params.is_perf_test) {
    // setup fake data
    int batch_size = static_cast<int>(params.batch_size_);
    int max_seq_len_in_batch = 512;
    std::vector<std::string> tensor_names = {"input1",
                                             "input2",
                                             "input3",
                                             "masked_lm_positions",
                                             "masked_lm_ids",
                                             "masked_lm_weights",
                                             "next_sentence_labels"};
    std::vector<TensorShape> tensor_shapes = {{batch_size, max_seq_len_in_batch},
                                              {batch_size, max_seq_len_in_batch},
                                              {batch_size, max_seq_len_in_batch},
                                              {batch_size, 80},
                                              {batch_size, 80},
                                              {batch_size, 80},
                                              {batch_size}};
    std::vector<onnx::TensorProto_DataType> tensor_types = {onnx::TensorProto_DataType_INT64,
                                                            onnx::TensorProto_DataType_INT64,
                                                            onnx::TensorProto_DataType_INT64,
                                                            onnx::TensorProto_DataType_INT64,
                                                            onnx::TensorProto_DataType_INT64,
                                                            onnx::TensorProto_DataType_FLOAT,
                                                            onnx::TensorProto_DataType_INT64};

    auto random_perf_data = std::make_shared<RandomDataSet>(params.num_of_perf_samples, tensor_names, tensor_shapes, tensor_types);

    runner = std::make_unique<TrainingRunner>(random_perf_data, random_perf_data, params);

  } else {
    const size_t max_num_files_preload = 2;

    auto training_data_loader = std::make_shared<DataLoader>(params.input_name_map_,
                                                             params.train_data_dir,
                                                             max_num_files_preload,
                                                             device_id,
                                                             device_count);
    auto test_data_loader = std::make_shared<DataLoader>(params.input_name_map_,
                                                         params.test_data_dir,
                                                         max_num_files_preload);
    RETURN_IF_FAIL(training_data_loader->Load());
    // Evaluation is only done in device #0
    if (device_id == 0) {
      RETURN_IF_FAIL(test_data_loader->Load());
    }

    runner = std::make_unique<TrainingRunner>(training_data_loader, test_data_loader, params);
  }

  RETURN_IF_FAIL(runner->Initialize());
  RETURN_IF_FAIL(runner->Run());

#ifdef USE_HOROVOD
  shutdown_horovod();
#endif
}
