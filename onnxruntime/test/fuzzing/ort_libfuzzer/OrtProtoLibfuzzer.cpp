// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "src/mutator.h"
#include "OnnxPrediction.h"
#include "onnxruntime_session_options_config_keys.h"
#include "src/libfuzzer/libfuzzer_macro.h"
#include "onnx/onnx_pb.h"

#include <type_traits>

Ort::Env env;

std::string wstring_to_string(const std::wstring& wstr) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
  return converter.to_bytes(wstr);
}

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

  // View the output
  //
  predict.PrintOutputValues();
}

template <class Proto>
using PostProcessor =
    protobuf_mutator::libfuzzer::PostProcessorRegistration<Proto>;

// Helper function to generate random strings
std::string generate_random_string(size_t length, std::mt19937& rng) {
  const std::string characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  std::uniform_int_distribution<> dist(0, characters.size() - 1);
  std::string result;
  for (size_t i = 0; i < length; ++i) {
    result += characters[dist(rng)];
  }
  return result;
}

// Helper function to generate random float
float generate_random_float(std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  return dist(rng);
}

// PostProcessor for ONNX ModelProto with random values
static PostProcessor<onnx::ModelProto> reg1 = {
    [](onnx::ModelProto* model_proto, unsigned int seed) {
      std::mt19937 rng(seed);

      // Set model's IR version
      model_proto->set_ir_version(7);

      model_proto->set_producer_name("onnx");
      model_proto->set_producer_version("7.0");
      model_proto->set_domain("example.com");

      // Add a dummy opset import
      auto* opset_import = model_proto->add_opset_import();
      opset_import->set_version(10);

      // Access the graph from the model
      auto* graph = model_proto->mutable_graph();

      // Set a random name for the graph
      graph->set_name(generate_random_string(10, rng));
    }};

DEFINE_PROTO_FUZZER(const onnx::ModelProto& msg) {
  try {
    auto seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    onnx::ModelProto msg_proto = msg;
    predict(msg_proto, seed, env);
  } catch (const std::exception& e) {
    // Optionally log or suppress the exception
    // std::cerr << "Caught exception: " << e.what() << std::endl;
  } catch (...) {
    // Handle any other exceptions
    // std::cerr << "Caught unknown exception." << std::endl;
  }
}
