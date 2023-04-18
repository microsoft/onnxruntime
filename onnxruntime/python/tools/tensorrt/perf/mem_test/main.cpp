// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <vector>
#include <onnxruntime_cxx_api.h>
#include <tensorrt_provider_factory.h>

std::unique_ptr<OrtTensorRTProviderOptions> get_default_trt_provider_options() {
  auto tensorrt_options = std::make_unique<OrtTensorRTProviderOptions>();
  tensorrt_options->device_id = 0;
  tensorrt_options->has_user_compute_stream = 0;
  tensorrt_options->user_compute_stream = nullptr;
  tensorrt_options->trt_max_partition_iterations = 1000;
  tensorrt_options->trt_min_subgraph_size = 1;
  tensorrt_options->trt_max_workspace_size = 1 << 30;
  tensorrt_options->trt_fp16_enable = false;
  tensorrt_options->trt_int8_enable = false;
  tensorrt_options->trt_int8_calibration_table_name = "";
  tensorrt_options->trt_int8_use_native_calibration_table = false;
  tensorrt_options->trt_dla_enable = false;
  tensorrt_options->trt_dla_core = 0;
  tensorrt_options->trt_dump_subgraphs = false;
  tensorrt_options->trt_engine_cache_enable = false;
  tensorrt_options->trt_engine_cache_path = "";
  tensorrt_options->trt_engine_decryption_enable = false;
  tensorrt_options->trt_engine_decryption_lib_path = "";
  tensorrt_options->trt_force_sequential_engine_build = false;
  tensorrt_options->trt_context_memory_sharing_enable = false;
  tensorrt_options->trt_layer_norm_fp32_fallback = false;
  return tensorrt_options;
}

void run_ort_trt2() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
  const wchar_t* model_path = L"squeezenet.onnx";
  const wchar_t* calib_table = L"squeezenet_calibration.flatbuffers";
#else
  const char* model_path = "squeezenet.onnx";
  const char* calib_table = "squeezenet_calibration.flatbuffers";
#endif

  auto tensorrt_options = get_default_trt_provider_options();

  tensorrt_options->trt_engine_cache_enable = true;
  tensorrt_options->trt_int8_enable = true;
  tensorrt_options->trt_fp16_enable = true;
  tensorrt_options->trt_int8_calibration_table_name = calib_table;

  session_options.AppendExecutionProvider_TensorRT(*tensorrt_options.get());
  printf("Runing ORT TRT EP with:\n\tengine cache enabled\n\tfp16 enabled if supports\n\tint8 enabled if supports\n\tint8 calibration table provided\n");
  printf("First run ...\n");

  Ort::Session session(env, model_path, session_options);

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = session.GetInputCount();
  std::vector<Ort::AllocatedStringPtr> input_node_names_ptr;
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

  printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    auto input_name = session.GetInputNameAllocated(i, allocator);
    printf("Input %d : name=%s\n", i, input_name.get());
    input_node_names[i] = input_name.get();
    input_node_names_ptr.push_back(std::move(input_name));

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
    for (int j = 0; j < input_node_dims.size(); j++)
      printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
  }

  size_t input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size
                                             // use OrtGetTensorShapeElementCount() to get official size!

  std::vector<float> input_tensor_values(input_tensor_size);
  std::vector<const char*> output_node_names = {"softmaxout_1"};

  // initialize input data with values in [0.0, 1.0]
  for (unsigned int i = 0; i < input_tensor_size; i++)
    input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());

  // score model & input tensor, get back output tensor
  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // Get pointer to output tensor float values
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();
  assert(abs(floatarr[0] - 0.000045) < 1e-6);

  // score the model, and print scores for first 5 classes
  for (int i = 0; i < 5; i++)
    printf("Score for class [%d] =  %f\n", i, floatarr[i]);

  // Results should be as below...
  // Score for class[0] = 0.000045
  // Score for class[1] = 0.003846
  // Score for class[2] = 0.000125
  // Score for class[3] = 0.001180
  // Score for class[4] = 0.001317

  // we need another run in order to make TRT EP use engine cache
  printf("Second run ...\n");

  // score model & input tensor, get back output tensor
  output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // Get pointer to output tensor float values
  floatarr = output_tensors.front().GetTensorMutableData<float>();
  assert(abs(floatarr[0] - 0.000045) < 1e-6);

  // score the model, and print scores for first 5 classes
  for (int i = 0; i < 5; i++)
    printf("Score for class [%d] =  %f\n", i, floatarr[i]);

  // release buffers allocated by ORT alloctor
  for (const char* node_name : input_node_names)
    allocator.Free(const_cast<void*>(reinterpret_cast<const void*>(node_name)));

  printf("Done!\n");
}

void ort_trt_run_with_default_options() {
  //*************************************************************************
  // initialize  environment...one environment per process
  // environment maintains thread pools and other state info
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

  // initialize session options if needed
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

  // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this session
  // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);
  OrtSessionOptionsAppendExecutionProvider_Tensorrt(session_options, 0);

  // Sets graph optimization level
  // Available levels are
  // ORT_DISABLE_ALL -> To disable all optimizations
  // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
  // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
  // ORT_ENABLE_ALL -> To Enable All possible opitmizations
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  //*************************************************************************
  // create session and load model into memory
  // using squeezenet version 1.3
  // URL = https://github.com/onnx/models/tree/master/squeezenet
#ifdef _WIN32
  const wchar_t* model_path = L"squeezenet.onnx";
#else
  const char* model_path = "squeezenet.onnx";
#endif

  printf("Using Onnxruntime C++ API\n");
  Ort::Session session(env, model_path, session_options);

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = session.GetInputCount();
  std::vector<Ort::AllocatedStringPtr> input_node_names_ptr;
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

  printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    auto input_name = session.GetInputNameAllocated(i, allocator);
    printf("Input %d : name=%s\n", i, input_name.get());
    input_node_names[i] = input_name.get();
    input_node_names_ptr.push_back(std::move(input_name));

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
    for (int j = 0; j < input_node_dims.size(); j++)
      printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
  }

  // Results should be...
  // Number of inputs = 1
  // Input 0 : name = data_0
  // Input 0 : type = 1
  // Input 0 : num_dims = 4
  // Input 0 : dim 0 = 1
  // Input 0 : dim 1 = 3
  // Input 0 : dim 2 = 224
  // Input 0 : dim 3 = 224

  //*************************************************************************
  // Similar operations to get output node information.
  // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
  // OrtSessionGetOutputTypeInfo() as shown above.

  //*************************************************************************
  // Score the model using sample data, and inspect values

  size_t input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size
                                             // use OrtGetTensorShapeElementCount() to get official size!

  std::vector<float> input_tensor_values(input_tensor_size);
  std::vector<const char*> output_node_names = {"softmaxout_1"};

  // initialize input data with values in [0.0, 1.0]
  for (unsigned int i = 0; i < input_tensor_size; i++)
    input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());

  // score model & input tensor, get back output tensor
  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // Get pointer to output tensor float values
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();
  assert(abs(floatarr[0] - 0.000045) < 1e-6);

  // score the model, and print scores for first 5 classes
  for (int i = 0; i < 5; i++)
    printf("Score for class [%d] =  %f\n", i, floatarr[i]);

  // Results should be as below...
  // Score for class[0] = 0.000045
  // Score for class[1] = 0.003846
  // Score for class[2] = 0.000125
  // Score for class[3] = 0.001180
  // Score for class[4] = 0.001317

  // release buffers allocated by ORT alloctor
  for (const char* node_name : input_node_names)
    allocator.Free(const_cast<void*>(reinterpret_cast<const void*>(node_name)));

  printf("Done!\n");
}

void run_ort_trt() {
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
  const wchar_t* model_path = L"squeezenet.onnx";
  const wchar_t* calib_table = L"squeezenet_calibration.flatbuffers";
#else
  const char* model_path = "squeezenet.onnx";
  const char* calib_table = "squeezenet_calibration.flatbuffers";
#endif

  auto tensorrt_options = get_default_trt_provider_options();

  session_options.AppendExecutionProvider_TensorRT(*tensorrt_options.get());
  printf("Runing ORT TRT EP with default provider options\n");

  Ort::Session session(env, model_path, session_options);

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = session.GetInputCount();
  std::vector<Ort::AllocatedStringPtr> input_node_names_ptr;
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

  printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    auto input_name = session.GetInputNameAllocated(i, allocator);
    printf("Input %d : name=%s\n", i, input_name.get());
    input_node_names[i] = input_name.get();
    input_node_names_ptr.push_back(std::move(input_name));

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
    for (int j = 0; j < input_node_dims.size(); j++)
      printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
  }

  size_t input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size
                                             // use OrtGetTensorShapeElementCount() to get official size!

  std::vector<float> input_tensor_values(input_tensor_size);
  std::vector<const char*> output_node_names = {"softmaxout_1"};

  // initialize input data with values in [0.0, 1.0]
  for (unsigned int i = 0; i < input_tensor_size; i++)
    input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());

  // score model & input tensor, get back output tensor
  auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
  assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());

  // Get pointer to output tensor float values
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();
  assert(abs(floatarr[0] - 0.000045) < 1e-6);

  // score the model, and print scores for first 5 classes
  for (int i = 0; i < 5; i++)
    printf("Score for class [%d] =  %f\n", i, floatarr[i]);

  // Results should be as below...
  // Score for class[0] = 0.000045
  // Score for class[1] = 0.003846
  // Score for class[2] = 0.000125
  // Score for class[3] = 0.001180
  // Score for class[4] = 0.001317

  // release buffers allocated by ORT alloctor
  for (const char* node_name : input_node_names)
    allocator.Free(const_cast<void*>(reinterpret_cast<const void*>(node_name)));

  printf("Done!\n");
}

int main(int argc, char* argv[]) {
  run_ort_trt();
  run_ort_trt2();
  return 0;
}
