// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <vector>

void run_ort(const char* model_path, const char* lib_name = nullptr) {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
  const auto& api = Ort::GetApi();
  OrtTensorRTProviderOptionsV2* tensorrt_options;

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  OrtCUDAProviderOptions cuda_options;
  cuda_options.device_id = 0;
  session_options.AppendExecutionProvider_CUDA(cuda_options);
  session_options.RegisterCustomOpsLibrary(lib_name);

  Ort::Session session(env, model_path, session_options);

  Ort::AllocatorWithDefaultOptions allocator;

  const size_t num_input_nodes = session.GetInputCount();
  std::vector<Ort::AllocatedStringPtr> input_names_ptr;
  std::vector<const char*> input_node_names;
  input_names_ptr.reserve(num_input_nodes);
  input_node_names.reserve(num_input_nodes);
  std::vector<int64_t> input_node_dims = {1, 3, 224, 224};  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

  std::cout << "=== FP Custom Op Test ===" << std::endl;
  std::cout << "Number of inputs = " << num_input_nodes << std::endl;

  // iterate over all input nodes
  for (size_t i = 0; i < num_input_nodes; i++) {
    // print input node names
    auto input_name = session.GetInputNameAllocated(i, allocator);
    std::cout << "Input " << i << " : name =" << input_name.get() << std::endl;
    input_node_names.push_back(input_name.get());
    input_names_ptr.push_back(std::move(input_name));

    // print input node types
    auto type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::cout << "Input " << i << " : type = " << type << std::endl;

    // print input shapes/dims
    //input_node_dims = tensor_info.GetShape();
    std::cout << "Input " << i << " : num_dims = " << input_node_dims.size() << '\n';
    for (size_t j = 0; j < input_node_dims.size(); j++) {
      std::cout << "Input " << i << " : dim[" << j << "] =" << input_node_dims[j] << '\n';
    }
    std::cout << std::flush;
  }

  int input_tensor_size = 1 * 224 * 224 * 3;  // simplify ... using known dim values to calculate size
                                                       // use OrtGetTensorShapeElementCount() to get official size!

  std::vector<const char*> output_node_names = {"output"};
  std::vector<float> input_tensor_values(input_tensor_size);

  // initialize input data with values in [0.0, 1.0]
  for (unsigned int i = 0; i < input_tensor_size; i++) input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  auto input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size,
                                                      input_node_dims.data(), 4);

  assert(input_tensor.IsTensor());
  // warn up
  for (int i = 0; i < 1; i++) {
   session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
  }

  auto output_tensors = 
    session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // Get pointer to output tensor float values
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();

  std::cout << "Print first 5 elements of output ..." << std::endl;
  for (int i = 0; i < 5; i++) {
    std::cout << floatarr[i] << std::endl;
  }
  std::cout << "Done!" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc == 2) {
    const char* lib_name   = "./libcustom_op_ft_wrapper_library.so"; 
    run_ort(argv[1]);
  } else if (argc == 3) {
    run_ort(argv[1], argv[2]);
  } else {
    std::cout << "Run with following command:" << std::endl;
    std::cout << "./ft_custom_op_test model_path ft_wrapper_lib_path" << std::endl;
  }
  return 0;
}
