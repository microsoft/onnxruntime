// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "OnnxPrediction.h"
#include "onnxruntime_session_options_config_keys.h"
#include "src/libfuzzer/libfuzzer_macro.h"
#include "fuzzer/FuzzedDataProvider.h"

Ort::Env env;

void predict(onnx::ModelProto& msg, unsigned int seed, Ort::Env& env) {
  // Create object for prediction
  //
  OnnxPrediction predict(msg, env);

  // Give predict a function to generate the data
  // to run prediction on.
  //
  predict.SetupInput(GenerateDataForInputTypeTensor, seed);

  // Run the prediction on the data
  //
  predict.RunInference();
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  FuzzedDataProvider data_provider(data, size);
  onnx::ModelProto msg;
  try {
    if (!msg.ParseFromArray(data, static_cast<int>(size))) {
      return 0;  // Ignore invalid inputs
    }
    predict(msg, data_provider.ConsumeIntegral<int>(), env);
  } catch (const std::exception& e) {
    // Optionally log or suppress the exception
    // std::cerr << "Caught exception: " << e.what() << std::endl;
  } catch (...) {
    // Handle any other exceptions
    // std::cerr << "Caught unknown exception." << std::endl;
  }
  return 0;
}
