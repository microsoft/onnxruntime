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

//const static std::string MODEL_NAME = "mnist_fc_model";
//const static vector<int64_t> IMAGE_DIMS = {1, 784};
//const static vector<int64_t> LABEL_DIMS = {1, 10};

//const static std::string MODEL_NAME = "mnist_conv";
const static std::string MODEL_NAME = "mnist_conv_batch_unknown";
const static vector<int64_t> IMAGE_DIMS = {1, 1, 28, 28};
const static vector<int64_t> LABEL_DIMS = {1, 10};
const static std::string ORIGINAL_MODEL_PATH = MODEL_NAME + ".onnx";
const static std::string GENERATED_MODEL_WITH_COST_PATH = MODEL_NAME + "_with_cost.onnx";
const static std::string BACKWARD_MODEL_PATH = MODEL_NAME + "_bw.onnx";
const static std::string TRAINED_MODEL_PATH = MODEL_NAME + "_trained.onnx";
const static std::string TRAINED_MODEL_WITH_COST_PATH = MODEL_NAME + "_with_cost_trained.onnx";
const static std::string MNIST_DATA_PATH = "mnist_data";

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
  PrepareMNISTData(MNIST_DATA_PATH, IMAGE_DIMS, LABEL_DIMS, trainingData, testData);

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

  TrainingRunner runner(trainingData, testData, params);
  RETURN_IF_FAIL(runner.Initialize());
  RETURN_IF_FAIL(runner.Run());
}
