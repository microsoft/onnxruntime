// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// onnxruntime dependencies
#include <random>
#include "core/graph/model.h"
#include "core/common/logging/logging.h"
#include "core/framework/environment.h"
#include "core/common/logging/sinks/clog_sink.h"
#include "orttraining/core/session/training_session.h"
#include "mnist_reader/mnist_reader.hpp"
#include "mnist_reader/mnist_utils.hpp"

using namespace onnxruntime;
using namespace onnxruntime::training;

//const std::string MODEL_NAME = "inceptionv1";
//const std::string PREDICTION_NAME = "prob_1";
//const std::vector<std::string> EXCLUDE_WEIGHTS = {"OC2_DUMMY_1", "OC2_DUMMY_3"};

//const std::string MODEL_NAME = "alexnet";
//const std::string PREDICTION_NAME = "prob_1";
//const std::vector<std::string> EXCLUDE_WEIGHTS = {"OC2_DUMMY_1"};

//const std::string MODEL_NAME = "vgg19";
//const std::string PREDICTION_NAME = "prob_1";
//const std::vector<std::string> EXCLUDE_WEIGHTS = {"OC2_DUMMY_1"};

//const std::string MODEL_NAME = "caffenet";
//const std::string PREDICTION_NAME = "prob_1";
//const std::vector<std::string> EXCLUDE_WEIGHTS = {"OC2_DUMMY_1"};

//const std::string MODEL_NAME = "zfnet512";
//const std::string PREDICTION_NAME = "gpu_0/softmax_1";
//const std::vector<std::string> EXCLUDE_WEIGHTS = {"OC2_DUMMY_1"};

const std::string MODEL_NAME = "squeezenet";
const std::string PREDICTION_NAME = "pool10_1_reshaped";
const std::vector<std::string> EXCLUDE_WEIGHTS = {"pool10_1_shape"};

const std::string SHARED_PATH = "test_models/";
const std::string ORIGINAL_MODEL_PATH = SHARED_PATH + MODEL_NAME + "/model.onnx";
const std::string TRANSFORMED_MODEL_PATH = SHARED_PATH + MODEL_NAME + "/model_transformed.onnx";
const std::string GENERATED_MODEL_WITH_COST_PATH = SHARED_PATH + MODEL_NAME + "/model_with_cost.onnx";
const std::string BACKWARD_MODEL_PATH = SHARED_PATH + MODEL_NAME + "/model_bw.onnx";

#define TERMINATE_IF_FAILED(status)                                    \
  {                                                                    \
    if (!status.IsOK()) {                                              \
      LOGF_DEFAULT(ERROR, "Failed:%s", status.ErrorMessage().c_str()); \
      return -1;                                                       \
    }                                                                  \
  }

int main(int /*argc*/, char* /*args*/ []) {
  std::string default_logger_id{"Default"};
  logging::LoggingManager default_logging_manager{std::unique_ptr<logging::ISink>{new logging::CLogSink{}},
                                                  logging::Severity::kWARNING, false,
                                                  logging::LoggingManager::InstanceType::Default,
                                                  &default_logger_id};

  std::unique_ptr<Environment> env;
  TERMINATE_IF_FAILED(Environment::Create(nullptr, env));

  // Step 1: Load the model and generate gradient graph in a training session.
  SessionOptions so;
  TrainingSession training_session{so, *env};

  // TODO: TERMINATE_IF_FAILED swallows some errors and messes up the call stack. Perhaps, find an alternative for debug mode ?
  TERMINATE_IF_FAILED(training_session.Load(ORIGINAL_MODEL_PATH));

  TERMINATE_IF_FAILED(training_session.AddLossFuncion({"SoftmaxCrossEntropy", PREDICTION_NAME, "labels", "loss", kMSDomain}));

  TERMINATE_IF_FAILED(training_session.Save(GENERATED_MODEL_WITH_COST_PATH,
                                            TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC));

  auto weights = training_session.GetModelInitializers();
  for (auto weight : EXCLUDE_WEIGHTS) {
    weights.erase(weight);
  }
  std::vector<std::string> weights_to_train(weights.begin(), weights.end());

  if (!training_session.BuildGradientGraph(weights_to_train, "loss", true).IsOK()) {
    return -1;
  }

  //TERMINATE_IF_FAILED(training_session.Save(TRANSFORMED_MODEL_PATH,
  //                                          TrainingSession::SaveOption::NO_RELOAD));

  TERMINATE_IF_FAILED(training_session.Save(BACKWARD_MODEL_PATH,
                                            TrainingSession::SaveOption::WITH_UPDATED_WEIGHTS_AND_LOSS_FUNC_AND_GRADIENTS));

  TERMINATE_IF_FAILED(training_session.Initialize());
  return 0;
}
