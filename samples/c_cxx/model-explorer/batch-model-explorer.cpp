// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * This example demonstrates how to batch process data using the experimental C++ API.
 *
 * This example is based on the model-explorer.cpp example except it demonstrates how to
 * batch process data. Please start by checking out model-explorer.cpp first.
 *
 * This example is best run with one of the ResNet models (i.e. ResNet18) from the onnx model zoo at
 *   https://github.com/onnx/models
 *
 * Assumptions made in this example:
 *  1) The onnx model has 1 input node and 1 output node
 *  2) The onnx model has a symbolic first dimension (i.e. -1x3x224x224)
 *
 * 
 * In this example, we do the following:
 *  1) read in an onnx model
 *  2) print out some metadata information about inputs and outputs that the model expects
 *  3) create tensors by generating 3 random batches of data (with batch_size = 5) for input to the model
 *  4) pass each batch through the model and check the resulting output
 *
 *
 * NOTE: Some onnx models may not have a symbolic first dimension. To prepare the onnx model, see the python code snippet below.
 * =============  Python Example  ======================
 * import onnx
 * model = onnx.load_model('model.onnx')
 * model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = 'None'
 * onnx.save_model(model, 'model-symbolic.onnx')
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
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "batch-model-explorer");
  Ort::SessionOptions session_options;
  Ort::Experimental::Session session = Ort::Experimental::Session(env, model_file, session_options);

  // print name/shape of inputs
  auto input_names = session.GetInputNames();
  auto input_shapes = session.GetInputShapes();
  cout << "Input Node Name/Shape (" << input_names.size() << "):" << endl;
  for (size_t i = 0; i < input_names.size(); i++) {
    cout << "\t" << input_names[i] << " : " << print_shape(input_shapes[i]) << endl;
  }

  // print name/shape of outputs
  auto output_names = session.GetOutputNames();
  auto output_shapes = session.GetOutputShapes();
  cout << "Output Node Name/Shape (" << output_names.size() << "):" << endl;
  for (size_t i = 0; i < output_names.size(); i++) {
    cout << "\t" << output_names[i] << " : " << print_shape(output_shapes[i]) << endl;
  }

  // Assume model has 1 input node and 1 output node.
  assert(input_names.size() == 1 && output_names.size() == 1);

  int batch_size = 5;
  int num_batches = 3;
  auto input_shape = input_shapes[0];
  assert(input_shape[0] == -1);  // symbolic dimensions are represented by a -1 value
  input_shape[0] = batch_size;
  int num_elements_per_batch = calculate_product(input_shape);

  // process multiple batches
  for (int i = 0; i < num_batches; i++) {
    cout << "\nProcessing batch #" << i << endl;

    // Create an Ort tensor containing random numbers
    std::vector<float> batch_input_tensor_values(num_elements_per_batch);
    std::generate(batch_input_tensor_values.begin(), batch_input_tensor_values.end(), [&] { return rand() % 255; });  // generate random numbers in the range [0, 255]
    std::vector<Ort::Value> batch_input_tensors;
    batch_input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(batch_input_tensor_values.data(), batch_input_tensor_values.size(), input_shape));

    // double-check the dimensions of the input tensor
    assert(batch_input_tensors[0].IsTensor() &&
           batch_input_tensors[0].GetTensorTypeAndShapeInfo().GetShape() == input_shape);
    cout << "batch_input_tensor shape: " << print_shape(batch_input_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << endl;

    // pass data through model
    try {
      auto batch_output_tensors = session.Run(input_names, batch_input_tensors, output_names);
      // double-check the dimensions of the output tensors
      // NOTE: the number of output tensors is equal to the number of output nodes specifed in the Run() call
      assert(batch_output_tensors.size() == output_names.size() &&
             batch_output_tensors[0].IsTensor() &&
             batch_output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()[0] == batch_size);
      cout << "batch_output_tensor_shape: " << print_shape(batch_output_tensors[0].GetTensorTypeAndShapeInfo().GetShape()) << endl;
    } catch (const Ort::Exception& exception) {
      cout << "ERROR running model inference: " << exception.what() << endl;
      exit(-1);
    }
  }
  cout << "\nDone" << endl;
}
