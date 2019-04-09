// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/framework/environment.h"
#include "core/training/training_optimizer.h"
#include "core/training/training_session.h"
#include "core/training/weight_updater.h"
#include "test/training/runner/data_loader.h"
#include "test/training/runner/training_runner.h"
#include "test/training/runner/training_util.h"

using namespace onnxruntime;
using namespace onnxruntime::training;
using namespace std;

const static int NUM_OF_EPOCH = 15;
const static float LEARNING_RATE = 0.01f;
const static int BATCH_SIZE = 8;
const static int NUM_CLASS = 2;
const static int NUM_SAMPLES_FOR_EVALUATION = 8;

const static char* ORIGINAL_MODEL_PATH = "squeezenet.onnx";
const static char* GENERATED_MODEL_WITH_COST_PATH = "squeezenet_with_cost.onnx";
const static char* BACKWARD_MODEL_PATH = "squeezenet_bw.onnx";
const static char* TRAINED_MODEL_PATH = "squeezenet_trained.onnx";
const static char* TRAINED_MODEL_WITH_COST_PATH = "squeezenet_with_cost_trained.onnx";
const PATH_STRING_TYPE TRAINING_DATA_PATH = ORT_TSTR("squeezenet_data/training");
const PATH_STRING_TYPE TEST_DATA_PATH = ORT_TSTR("squeezenet_data/test");

int main(int /*argc*/, char* /*args*/[]) {
  string default_logger_id{"Default"};
  logging::LoggingManager default_logging_manager{unique_ptr<logging::ISink>{new logging::CLogSink{}},
                                                  logging::Severity::kWARNING,
                                                  false,
                                                  logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id};

  unique_ptr<Environment> env;
  ORT_ENFORCE(Environment::Create(env).IsOK());

  DataLoader training_data_loader;
  training_data_loader.Load(TRAINING_DATA_PATH);
  DataLoader test_data_loader;
  test_data_loader.Load(TEST_DATA_PATH);

  TrainingRunner::Parameters params;

  // Init params.
  params.model_path_ = ORIGINAL_MODEL_PATH;
  params.model_with_loss_func_path_ = GENERATED_MODEL_WITH_COST_PATH;
  params.model_with_training_graph_path_ = BACKWARD_MODEL_PATH;
  params.model_trained_path_ = TRAINED_MODEL_PATH;
  params.model_trained_with_loss_func_path_ = TRAINED_MODEL_WITH_COST_PATH;
  params.loss_func_info_ = {"SoftmaxCrossEntropy", "pool10_1_reshaped", "labels", "loss", kMSDomain};
  params.model_prediction_name_ = "softmaxout_1";
  //  params.weights_not_to_train_ = {"pool10_1_shape"};  // Use not-to-train list
  params.weights_to_train_ = {"conv10_w_0__71", "conv10_b_0__70"};
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
    printf("#examples: %d, #correct: %d, precision: %0.04f, loss:%0.04f \n\n",
           static_cast<int>(num_of_test_run),
           true_count,
           precision,
           total_loss / num_of_test_run);
    true_count = 0;
    total_loss = 0.0f;
  };

  TrainingRunner runner(training_data_loader.MutableDataSet(), test_data_loader.MutableDataSet(), params);
  RETURN_IF_FAIL(runner.Initialize());
  RETURN_IF_FAIL(runner.Run());

  return 0;
}
