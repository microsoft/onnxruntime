// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <onnxruntime_c_api.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>

//*****************************************************************************
// helper function to check for status
#define CHECK_STATUS(expr)                               \
  {                                                      \
    OrtStatus* onnx_status = (expr);                     \
    if (onnx_status != NULL) {                           \
      const char* msg = OrtGetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                      \
      OrtReleaseStatus(onnx_status);                     \
      exit(1);                                           \
    }                                                    \
  }

int main(int argc, char* argv[]) {
  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info
  OrtEnv* env;
  CHECK_STATUS(OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

  // initialize session options if needed
  OrtSessionOptions* session_options;
  CHECK_STATUS(OrtCreateSessionOptions(&session_options));
  OrtSetSessionThreadPoolSize(session_options, 1);

  // Sets graph optimization level
  OrtSetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);

  // Optionally add more execution providers via session_options
  // E.g. for CUDA include cuda_provider_factory.h and uncomment the following line:
  // OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

  //*************************************************************************
  // create session and load model into memory
  // using squeezenet version 1.3
  // URL = https://github.com/onnx/models/tree/master/squeezenet
  OrtSession* session;
#ifdef _WIN32
  const wchar_t* model_path = L"squeezenet.onnx";
#else
  const char* model_path = "squeezenet.onnx";
#endif

  printf("Using Onnxruntime C API\n");
  CHECK_STATUS(OrtCreateSession(env, model_path, session_options, &session));

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  size_t num_input_nodes;
  OrtStatus* status;
  OrtAllocator* allocator;
  CHECK_STATUS(OrtGetAllocatorWithDefaultOptions(&allocator));

  // print number of model input nodes
  status = OrtSessionGetInputCount(session, &num_input_nodes);
  std::vector<const char*> input_node_names(num_input_nodes);
  std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

  printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (size_t i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name;
    status = OrtSessionGetInputName(session, i, allocator, &input_name);
    printf("Input %zu : name=%s\n", i, input_name);
    input_node_names[i] = input_name;

    // print input node types
    OrtTypeInfo* typeinfo;
    status = OrtSessionGetInputTypeInfo(session, i, &typeinfo);
    const OrtTensorTypeAndShapeInfo* tensor_info;
    CHECK_STATUS(OrtCastTypeInfoToTensorInfo(typeinfo, &tensor_info));
    ONNXTensorElementDataType type;
    CHECK_STATUS(OrtGetTensorElementType(tensor_info, &type));
    printf("Input %zu : type=%d\n", i, type);

    // print input shapes/dims
    size_t num_dims;
    CHECK_STATUS(OrtGetDimensionsCount(tensor_info, &num_dims));
    printf("Input %zu : num_dims=%zu\n", i, num_dims);
    input_node_dims.resize(num_dims);
    OrtGetDimensions(tensor_info, (int64_t*)input_node_dims.data(), num_dims);
    for (size_t j = 0; j < num_dims; j++)
      printf("Input %zu : dim %zu=%jd\n", i, j, input_node_dims[j]);

    OrtReleaseTypeInfo(typeinfo);
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
  for (size_t i = 0; i < input_tensor_size; i++)
    input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // create input tensor object from data values
  OrtAllocatorInfo* allocator_info;
  CHECK_STATUS(OrtCreateCpuAllocatorInfo(OrtArenaAllocator, OrtMemTypeDefault, &allocator_info));
  OrtValue* input_tensor = NULL;
  CHECK_STATUS(OrtCreateTensorWithDataAsOrtValue(allocator_info, input_tensor_values.data(), input_tensor_size * sizeof(float), input_node_dims.data(), 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
  int is_tensor;
  CHECK_STATUS(OrtIsTensor(input_tensor, &is_tensor));
  assert(is_tensor);
  OrtReleaseAllocatorInfo(allocator_info);

  // score model & input tensor, get back output tensor
  OrtValue* output_tensor = NULL;
  CHECK_STATUS(OrtRun(session, NULL, input_node_names.data(), (const OrtValue* const*)&input_tensor, 1, output_node_names.data(), 1, &output_tensor));
  CHECK_STATUS(OrtIsTensor(output_tensor, &is_tensor));
  assert(is_tensor);

  // Get pointer to output tensor float values
  float* floatarr;
  CHECK_STATUS(OrtGetTensorMutableData(output_tensor, (void**)&floatarr));
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

  OrtReleaseValue(output_tensor);
  OrtReleaseValue(input_tensor);
  OrtReleaseSession(session);
  OrtReleaseSessionOptions(session_options);
  OrtReleaseEnv(env);
  printf("Done!\n");
  return 0;
}
