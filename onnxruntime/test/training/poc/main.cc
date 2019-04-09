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
const static float LEARNING_RATE = 0.5f;
const static int BATCH_SIZE = 100;
const static int NUM_CLASS = 10;
const static int NUM_SAMPLES_FOR_EVALUATION = 1000;

const static char* ORIGINAL_MODEL_PATH = "mnist_fc_model.onnx";
const static char* GENERATED_MODEL_WITH_COST_PATH = "mnist_fc_model_with_cost.onnx";
const static char* BACKWARD_MODEL_PATH = "mnist_fc_model_bw.onnx";
const static char* TRAINED_MODEL_PATH = "mnist_fc_model_trained.onnx";
const static char* TRAINED_MODEL_WITH_COST_PATH = "mnist_fc_model_with_cost_trained.onnx";
const static char* MNIST_DATA_PATH = "mnist_data";

int main(int /*argc*/, char* /*args*/[]) {
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
  PrepareMNISTData(MNIST_DATA_PATH, trainingData, testData);

  TrainingRunner::Parameters params;

  // Init params.
  params.model_path_ = ORIGINAL_MODEL_PATH;
  params.model_with_loss_func_path_ = GENERATED_MODEL_WITH_COST_PATH;
  params.model_with_training_graph_path_ = BACKWARD_MODEL_PATH;
  params.model_trained_path_ = TRAINED_MODEL_PATH;
  params.model_trained_with_loss_func_path_ = TRAINED_MODEL_WITH_COST_PATH;
  params.loss_func_info_ = {"SoftmaxCrossEntropy", "predictions", "labels", "loss", kMSDomain};
  params.model_prediction_name_ = "predictions";
  params.weights_to_train_ = {"W1", "W2", "W3", "B1", "B2", "B3"};
  params.batch_size_ = BATCH_SIZE;
  params.num_of_epoch_ = NUM_OF_EPOCH;
  params.learning_rate_ = LEARNING_RATE;
  params.num_of_samples_for_evaluation_ = NUM_SAMPLES_FOR_EVALUATION;

  int true_count = 0;
  float total_loss = 0.0f;
  params.error_function_ = [&true_count, &total_loss](const MLValue& predict, const MLValue& label, const MLValue& loss) {
    const float* prediction_data = predict.Get<Tensor>().template Data<float>();

    auto max_class_index = std::distance(prediction_data,
                                         std::max_element(prediction_data, prediction_data + NUM_CLASS));

    const float* label_data = label.Get<Tensor>().template Data<float>();

    if (static_cast<int>(label_data[max_class_index]) == 1) {
      true_count++;
    }

    total_loss += *(loss.Get<Tensor>().Data<float>());
  };

  params.post_evaluation_callback_ = [&true_count, &total_loss](size_t num_of_test_run) {
    float precision = float(true_count) / num_of_test_run;
    printf("");
    printf("#examples: %d, #correct: %d, precision: %0.04f, loss:%0.04f \n\n",
           static_cast<int>(num_of_test_run),
           true_count,
           precision,
           total_loss / num_of_test_run);
    true_count = 0;
    total_loss = 0.0f;
  };

  TrainingRunner runner(trainingData, testData, params);
  RETURN_IF_FAIL(runner.Initialize());
  RETURN_IF_FAIL(runner.Run());
}
