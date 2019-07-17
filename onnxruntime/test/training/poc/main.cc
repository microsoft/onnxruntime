// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cxxopts.hpp"
#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/platform/env.h"
#include "core/session/environment.h"
#include "core/training/training_session.h"
#include "core/training/tensorboard/event_writer.h"
#include "test/training/poc/mnist_data_provider.h"
#include "test/training/runner/training_runner.h"
#include "test/training/runner/training_util.h"

#ifdef USE_HOROVOD
#include "core/graph/training/horovod_adapters.h"
#include <mpi.h>
#endif

#include <condition_variable>
#include <mutex>
#include <tuple>

using namespace onnxruntime;
using namespace onnxruntime::training;
using namespace onnxruntime::training::tensorboard;
using namespace std;

const static int NUM_CLASS = 10;
const static vector<int64_t> IMAGE_DIMS = {784};  //{1, 28, 28} for mnist_conv
const static vector<int64_t> LABEL_DIMS = {10};

Status ParseArguments(int argc, char* argv[], TrainingRunner::Parameters& params) {
  cxxopts::Options options("POC Training", "Main Program to train on MNIST");
  // clang-format off
  options
    .allow_unrecognised_options()
    .add_options()
      ("model_name", "model to be trained", cxxopts::value<std::string>())
      ("mnist_data_dir", "MNIST training and test data path.",
        cxxopts::value<std::string>()->default_value("mnist_data"))
      ("log_dir", "The directory to write tensorboard events.",
        cxxopts::value<std::string>()->default_value("logs/poc"))
      ("use_cuda", "Use CUDA execution provider for training.")
      ("num_of_epoch", "Num of epoch", cxxopts::value<int>()->default_value("2"))
      ("train_batch_size", "Total batch size for training.", cxxopts::value<int>()->default_value("100"))
      ("eval_batch_size", "Total batch size for eval.", cxxopts::value<int>()->default_value("100"))
      ("learning_rate", "The initial learning rate for Adam.", cxxopts::value<float>()->default_value("0.1"))
      ("evaluation_period", "How many training steps to make before making an evaluation.",
        cxxopts::value<size_t>()->default_value("1"));
  // clang-format on

  try {
    auto flags = options.parse(argc, argv);

    params.model_name = flags["model_name"].as<std::string>();
    params.use_cuda_ = flags.count("use_cuda") > 0;
    params.learning_rate_ = flags["learning_rate"].as<float>();
    params.num_of_epoch_ = flags["num_of_epoch"].as<int>();
    params.batch_size_ = flags["train_batch_size"].as<int>();
    if (flags.count("eval_batch_size")) {
      params.eval_batch_size = flags["eval_batch_size"].as<int>();
    } else {
      params.eval_batch_size = params.batch_size_;
    }
    params.evaluation_period = flags["evaluation_period"].as<size_t>();

    auto train_data_dir = flags["mnist_data_dir"].as<std::string>();
    auto log_dir = flags["log_dir"].as<std::string>();
    params.train_data_dir.assign(train_data_dir.begin(), train_data_dir.end());
    params.log_dir.assign(log_dir.begin(), log_dir.end());

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
int true_count = 0;
float total_loss = 0.0f;

void setup_training_params(TrainingRunner::Parameters& params) {
  params.model_path_ = params.model_name + ".onnx";
  params.model_with_loss_func_path_ = params.model_name + "_with_cost.onnx";
  params.model_with_training_graph_path_ = params.model_name + "_bw.onnx";
  params.model_actual_running_graph_path_ = params.model_name + "_bw_running.onnx";
  params.model_trained_path_ = params.model_name + "_trained.onnx";
  params.model_trained_with_loss_func_path_ = params.model_name + "_with_cost_trained.onnx";
  params.model_prediction_name_ = "predictions";
  params.loss_func_info_ = LossFunctionInfo(OpDef("SoftmaxCrossEntropy"),
                                            "loss",
                                            {params.model_prediction_name_, "labels"});
  params.fetch_names = {"predictions", "loss"};
  params.weights_not_to_train_ = {""};

  // TODO: simplify provider/optimizer configuration. For now it is fixed to used SGD with CPU and Adam with GPU.
  if (params.use_cuda_) {
      // TODO: This should be done in SGD optimizer. Will refactor when optimizing the kernel.
      // Adding another cuda kernel call for this division seems wasteful currently.
      params.learning_rate_ /= params.batch_size_;
      params.in_graph_optimizer_name_ = "AdamOptimizer";

      params.adam_opt_params_.alpha_ = 0.9f;
      params.adam_opt_params_.beta_ = 0.999f;
      params.adam_opt_params_.lambda_ = 0;
      params.adam_opt_params_.epsilon_ = 0.1f;
  } else {
      params.learning_rate_ /= params.batch_size_;
      params.in_graph_optimizer_name_ = "SGDOptimizer";
  }

  params.error_function_ = [](const std::vector<std::string>& /*feed_names*/,
                              const std::vector<OrtValue>& feeds,
                              const std::vector<std::string>& /*fetch_names*/,
                              const std::vector<OrtValue>& fetches) {
    const Tensor& label_t = feeds[1].Get<Tensor>();
    const Tensor& predict_t = fetches[0].Get<Tensor>();
    const Tensor& loss_t = fetches[1].Get<Tensor>();

    const float* prediction_data = predict_t.template Data<float>();
    const float* label_data = label_t.template Data<float>();
    const float* loss_data = loss_t.template Data<float>();

    const TensorShape predict_shape = predict_t.Shape();
    const TensorShape label_shape = label_t.Shape();
    const TensorShape loss_shape = loss_t.Shape();
    ORT_ENFORCE(predict_shape == label_shape);

    int64_t batch_size = predict_shape[0];
    for (int n = 0; n < batch_size; ++n) {
      auto max_class_index = std::distance(prediction_data,
                                           std::max_element(prediction_data, prediction_data + NUM_CLASS));

      if (static_cast<int>(label_data[max_class_index]) == 1) {
        true_count++;
      }

      prediction_data += predict_shape.SizeFromDimension(1);
      label_data += label_shape.SizeFromDimension(1);
    }
    total_loss += *loss_data;
  };

  auto tensorboard = std::make_shared<EventWriter>(params.log_dir);
  params.post_evaluation_callback_ = [tensorboard](size_t num_samples, size_t step) {
    float precision = float(true_count) / num_samples;
    float average_loss = total_loss / float(num_samples);
    tensorboard->AddScalar("precision", precision, step);
    tensorboard->AddScalar("loss", average_loss, step);
    printf("Step: %zu, #examples: %d, #correct: %d, precision: %0.04f, loss: %0.04f \n\n",
           step,
           static_cast<int>(num_samples),
           true_count,
           precision,
           average_loss);
    true_count = 0;
    total_loss = 0.0f;
  };
}

int main(int argc, char* args[]) {
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

  // setup training params
  TrainingRunner::Parameters params;
  ParseArguments(argc, args, params);
  setup_training_params(params);

  // setup horovod
  int device_id = 0,
      device_count = 1;

#ifdef USE_HOROVOD
  std::tie(device_id, device_count) = setup_horovod();
#endif

#ifdef USE_CUDA
  params.learning_rate_ /= device_count;
  params.world_rank_ = device_id;
  if (params.use_cuda_) {
    printf("Using cuda device #%d \n", params.world_rank_);
  }
#endif

  // setup data
  std::vector<string> feeds{"X", "labels"};
  auto trainingData = std::make_shared<DataSet>(feeds);
  auto testData = std::make_shared<DataSet>(feeds);
  std::string mnist_data_path(params.train_data_dir.begin(), params.train_data_dir.end());
  PrepareMNISTData(mnist_data_path, IMAGE_DIMS, LABEL_DIMS, *trainingData, *testData, device_id /* shard_to_load */, device_count /* total_shards */);

  // start training session
  TrainingRunner runner(trainingData, testData, params);
  RETURN_IF_FAIL(runner.Initialize());
  RETURN_IF_FAIL(runner.Run());

#ifdef USE_HOROVOD
  shutdown_horovod();
#endif
}
