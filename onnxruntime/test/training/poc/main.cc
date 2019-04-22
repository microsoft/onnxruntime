// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/environment.h"
#include "core/training/training_optimizer.h"
#include "core/training/training_session.h"
#include "core/training/weight_updater.h"
#include "test/training/poc/mnist_data_provider.h"
#include "test/training/runner/training_runner.h"
#include "test/training/runner/training_util.h"

using namespace onnxruntime;
using namespace onnxruntime::training;
using namespace std;

const static int NUM_OF_EPOCH = 2;
const static float LEARNING_RATE = 0.1f;
const static int BATCH_SIZE = 100;
const static int NUM_CLASS = 10;
const static int NUM_SAMPLES_FOR_EVALUATION = 100;
const static vector<int64_t> IMAGE_DIMS = {1, 784};  //{1, 1, 28, 28} for mnist_conv
const static vector<int64_t> LABEL_DIMS = {1, 10};
const static std::string MNIST_DATA_PATH = "mnist_data";

int main(int argc, char* args[]) {
  if (argc < 2) {
    printf("Incorrect command line for %s\n", args[0]);
#ifdef USE_CUDA
    printf("usage: exe_name model_name [gpu]\n");
#else
    printf("usage: exe_name model_name\n");
#endif
    return -1;
  }

  string default_logger_id{"Default"};
  logging::LoggingManager default_logging_manager{unique_ptr<logging::ISink>{new logging::CLogSink{}},
                                                  logging::Severity::kWARNING,
                                                  false,
                                                  logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id};

  unique_ptr<Environment> env;
  ORT_ENFORCE(Environment::Create(env).IsOK());

  DataSet trainingData({"X", "labels"});
  DataSet testData({"X", "labels"});
  PrepareMNISTData(MNIST_DATA_PATH, IMAGE_DIMS, LABEL_DIMS, trainingData, testData);

  TrainingRunner::Parameters params;

  // Init params.
  std::string model_name = args[1];
  params.model_path_ = model_name + ".onnx";
  params.model_with_loss_func_path_ = model_name + "_with_cost.onnx";
  params.model_with_training_graph_path_ = model_name + "_bw.onnx";
  params.model_actual_running_graph_path_ = model_name + "_bw_running.onnx";
  params.model_trained_path_ = model_name + "_trained.onnx";
  params.model_trained_with_loss_func_path_ = model_name + "_with_cost_trained.onnx";
  params.loss_func_info_ = {"SoftmaxCrossEntropy", "predictions", "labels", "loss", kMSDomain};
  params.model_prediction_name_ = "predictions";
  //params.weights_to_train_ = {"W1", "W2", "W3", "B1", "B2", "B3"};
  params.weights_not_to_train_ = {""};
  params.batch_size_ = BATCH_SIZE;
  params.num_of_epoch_ = NUM_OF_EPOCH;
  params.learning_rate_ = LEARNING_RATE;
  params.num_of_samples_for_evaluation_ = NUM_SAMPLES_FOR_EVALUATION;

  int true_count = 0;
  float total_loss = 0.0f;

  params.error_function_ = [&true_count, &total_loss](const MLValue& predict, const MLValue& label, const MLValue& loss) {
    const Tensor& predict_t = predict.Get<Tensor>();
    const Tensor& label_t = label.Get<Tensor>();
    const Tensor& loss_t = loss.Get<Tensor>();

    const float* prediction_data = predict_t.template Data<float>();
    const float* label_data = label_t.template Data<float>();
    const float* loss_data = loss_t.template Data<float>();

    const TensorShape predict_shape = predict_t.Shape();
    const TensorShape label_shape = label_t.Shape();
    const TensorShape loss_shape = loss_t.Shape();
    ORT_ENFORCE(predict_shape == label_shape);
    ORT_ENFORCE(loss_shape.NumDimensions() == 1 && loss_shape[0] == 1);

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

  params.post_evaluation_callback_ = [&true_count, &total_loss](size_t num_samples) {
    float precision = float(true_count) / num_samples;
    float average_loss = total_loss / float(num_samples);
    printf("");
    printf("#examples: %d, #correct: %d, precision: %0.04f, loss: %0.04f \n\n",
           static_cast<int>(num_samples),
           true_count,
           precision,
           average_loss);
    true_count = 0;
    total_loss = 0.0f;
  };

#ifdef USE_CUDA
  params.use_cuda_ = (argc > 2 && string(args[2]) == "gpu");
#endif

  TrainingRunner runner(trainingData, testData, params);
  RETURN_IF_FAIL(runner.Initialize());
  RETURN_IF_FAIL(runner.Run());
}
