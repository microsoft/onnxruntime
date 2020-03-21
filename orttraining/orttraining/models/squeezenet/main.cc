// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/logging/logging.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "core/session/environment.h"
#include "orttraining/core/framework/weight_updater.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/models/runner/data_loader.h"
#include "orttraining/models/runner/training_runner.h"
#include "orttraining/models/runner/training_util.h"

using namespace onnxruntime;
using namespace onnxruntime::training;
using namespace std;

const static int NUM_OF_EPOCH = 15;
const static float LEARNING_RATE = 0.001f;
const static int BATCH_SIZE = 32;
const static int NUM_CLASS = 2;
const static int NUM_SAMPLES_FOR_EVALUATION = 32;

const static char* ORIGINAL_MODEL_PATH = "squeezenet.onnx";
const static char* GENERATED_MODEL_WITH_COST_PATH = "squeezenet_with_cost.onnx";
const static char* BACKWARD_MODEL_PATH = "squeezenet_bw.onnx";
const static char* OUTPUT_DIR = ".";
const PATH_STRING_TYPE TRAINING_DATA_PATH = ORT_TSTR("squeezenet_data/training");
const PATH_STRING_TYPE TEST_DATA_PATH = ORT_TSTR("squeezenet_data/test");

int main(int /*argc*/, char* /*args*/ []) {
  string default_logger_id{"Default"};
  logging::LoggingManager default_logging_manager{unique_ptr<logging::ISink>{new logging::CLogSink{}},
                                                  logging::Severity::kWARNING,
                                                  false,
                                                  logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id};

  unique_ptr<Environment> env;
  ORT_ENFORCE(Environment::Create(nullptr, env).IsOK());

  DataLoader training_data_loader;
  // Uncomment below line to load only shard-0 training data (total shards = 3), for data parallelism.
  // training_data_loader.Load(TRAINING_DATA_PATH, 0, 3);
  training_data_loader.Load(TRAINING_DATA_PATH);
  DataLoader test_data_loader;
  test_data_loader.Load(TEST_DATA_PATH);

  TrainingRunner::Parameters params;

  // Init params.
  params.model_path_ = ORIGINAL_MODEL_PATH;
  params.model_with_loss_func_path_ = GENERATED_MODEL_WITH_COST_PATH;
  params.model_with_training_graph_path_ = BACKWARD_MODEL_PATH;
  params.output_dir = OUTPUT_DIR;
  params.model_prediction_name_ = "pool10_1_reshaped";
  params.loss_func_info_ = LossFunctionInfo(OpDef("SoftmaxCrossEntropy"),
                                            "loss",
                                            {params.model_prediction_name_, "labels"});
  //params.weights_not_to_train_ = {"pool10_1_shape"};  // Use not-to-train list
  params.weights_to_train_ = {"conv10_w_0__71", "conv10_b_0__70"};
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
    printf("#examples: %d, #correct: %d, precision: %0.04f, loss:%0.04f \n\n",
           static_cast<int>(num_samples),
           true_count,
           precision,
           average_loss);
    true_count = 0;
    total_loss = 0.0f;
  };

  TrainingRunner runner(training_data_loader.MutableDataSet(), test_data_loader.MutableDataSet(), *env, params);
  RETURN_IF_FAIL(runner.Initialize());
  RETURN_IF_FAIL(runner.Run());

  return 0;
}
