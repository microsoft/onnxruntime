// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * This sample application demonstrates how to use components of the experimental C++ API
 * to query for model inputs/outputs and how to run inferrence on a model.
 *
 * This example is best run with one of the ResNet models (i.e. ResNet18) from the onnx model zoo at
 *   https://github.com/onnx/models
 *
 * Assumptions made in this example:
 *  1) The onnx model has 1 input node and 1 output node
 *
 * 
 * In this example, we do the following:
 *  1) read in an onnx model
 *  2) print out some metadata information about inputs and outputs that the model expects
 *  3) generate random data for an input tensor
 *  4) pass tensor through the model and check the resulting tensor
 *
 */

#include <algorithm>  // std::generate
#include <assert.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <experimental_onnxruntime_cxx_api.h>

// pretty prints a shape dimension vector
std::string print_shape(const std::vector<int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++)
    ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

int calculate_product(const std::vector<int64_t>& v) {
  int total = 1;
  for (auto& i : v) total *= i;
  return total;
}

using namespace std;

int main(int argc, char** argv) {
  if (argc != 2) {
    cout << "Usage: ./onnx-api-example <onnx_model.onnx>" << endl;
    return -1;
  }

#ifdef _WIN32
  std::string str = argv[1];
  std::wstring wide_string = std::wstring(str.begin(), str.end());
  std::basic_string<ORTCHAR_T> model_file = std::basic_string<ORTCHAR_T>(wide_string);
#else
  std::string model_file = argv[1];
#endif

  // onnxruntime setup
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "example-model-explorer");
  Ort::SessionOptions session_options;
  Ort::Experimental::Session session = Ort::Experimental::Session(env, model_file, session_options);  // access experimental components via the Experimental namespace

  // print name/shape of inputs
  std::vector<std::string> input_names = session.GetInputNames();
  std::vector<std::vector<int64_t> > input_shapes = session.GetInputShapes();
  cout << "Input Node Name/Shape (" << input_names.size() << "):" << endl;
  for (size_t i = 0; i < input_names.size(); i++) {
    cout << "\t" << input_names[i] << " : " << print_shape(input_shapes[i]) << endl;
  }

  // print name/shape of outputs
  std::vector<std::string> output_names = session.GetOutputNames();
  std::vector<std::vector<int64_t> > output_shapes = session.GetOutputShapes();
  cout << "Output Node Name/Shape (" << output_names.size() << "):" << endl;
  for (size_t i = 0; i < output_names.size(); i++) {
    cout << "\t" << output_names[i] << " : " << print_shape(output_shapes[i]) << endl;
  }

  // Assume model has 1 input node and 1 output node.
  assert(input_names.size() == 1 && output_names.size() == 1);

  // Create a single Ort tensor of random numbers
  auto input_shape = input_shapes[0];
  int total_number_elements = calculate_product(input_shape);
  std::vector<float> input_tensor_values(total_number_elements);
  std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&] { return rand() % 255; });  // generate random numbers in the range [0, 255]
  std::vector<Ort::Value> input_tensors;
  input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(input_tensor_values.data(), input_tensor_values.size(), input_shape));

  // double-check the dimensions of the input tensor
  assert(input_tensors[0].IsTensor() &&
         input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_shape);
  cout << "\ninput_tensor shape: " << print_shape(input_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << endl;

  // pass data through model
  cout << "Running model...";
  try {
    auto output_tensors = session.Run(session.GetInputNames(), input_tensors, session.GetOutputNames());
    cout << "done" << endl;

    // double-check the dimensions of the output tensors
    // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
    assert(output_tensors.size() == session.GetOutputNames().size() &&
           output_tensors[0].IsTensor());
    cout << "output_tensor_shape: " << print_shape(output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << endl;

  } catch (const Ort::Exception& exception) {
    cout << "ERROR running model inference: " << exception.what() << endl;
    exit(-1);
  }
}
