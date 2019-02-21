# C API

## Features

* Creating an InferenceSession from an on-disk model file and a set of SessionOptions.
* Registering customized loggers.
* Registering customized allocators.
* Registering predefined providers and set the priority order. ONNXRuntime has a set of predefined execution providers,like CUDA, MKLDNN. User can register providers to their InferenceSession. The order of registration indicates the preference order as well.
* Running a model with inputs. These inputs must be in CPU memory, not GPU. If the model has multiple outputs, user can specify which outputs they want.
* Converting an in-memory ONNX Tensor encoded in protobuf format, to a pointer that can be used as model input.
* Setting the thread pool size for each session.
* Dynamically loading custom ops.

## Usage Overview

1. Include [onnxruntime_c_api.h](/include/onnxruntime/core/session/onnxruntime_c_api.h).
2. Call OrtCreateEnv
3. Create Session: OrtCreateSession(env, model_uri, nullptr,...)
4. Create Tensor
   1) OrtCreateAllocatorInfo
   2) OrtCreateTensorWithDataAsONNXValue
5. OrtRun

### Sample code

The example below shows a sample run using the SqueezeNet model from ONNX model zoo, including dynamically reading model inputs, outputs, shape and type information, as well as running a sample vector and fetching the resulting class probabilities for inspection. 


```c
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//

#include <assert.h>
#include <core/session/onnxruntime_c_api.h>
#include <core/providers/cpu/cpu_provider_factory.h>
#include <stdlib.h>
#include <stdio.h>

//*****************************************************************************
// helper function to check for status
#define CHECK_STATUS(expr)                               \
  do {                                                   \
    OrtStatus* onnx_status = (expr);                     \
    if (onnx_status != NULL) {                           \
      const char* msg = OrtGetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                      \
      OrtReleaseStatus(onnx_status);                     \
      abort();                                           \
    }                                                    \
  } while (0);

int main(int argc, char *argv[])
{
	//*************************************************************************
	// initialize  enviroment...one enviroment per process
	// enviroment maintains thread pools and other state info
	OrtEnv* env;
	CHECK_STATUS(OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

	// initialize session options if needed
	OrtSessionOptions* session_option = OrtCreateSessionOptions();
	OrtSetSessionThreadPoolSize(session_option, 1);

	//*************************************************************************
	// create session and load model into memory
	// using squeezenet version 1.3 
	// URL = https://github.com/onnx/models/tree/master/squeezenet
	OrtSession* session;
	const wchar_t * model_path = L"model.onnx";
	CHECK_STATUS(OrtCreateSession(env, model_path, session_option, &session));

	//*************************************************************************
	// print model input layer (node names, types, shape etc.)

	size_t num_inputs;
	OrtStatus* status;
	OrtAllocator* allocator;
	OrtCreateDefaultAllocator(&allocator);

	// print number of model input nodes
	status = OrtSessionGetInputCount(session, &num_inputs);
	char **input_names = (char**)malloc(num_inputs * sizeof(char*));
	printf("Number of inputs = %zu\n", num_inputs);

	// iterate over all input nodes
	for (int i = 0; i < num_inputs; i++)
	{
		// print input node names
		char* input_name;
		status = OrtSessionGetInputName(session, i, allocator, &input_name);
		printf("Input %d : name=%s\n", i, input_name);
		input_names[i] = input_name;

		// print input node types 
		OrtTypeInfo* typeinfo;
		status = OrtSessionGetInputTypeInfo(session, i, &typeinfo);
		const OrtTensorTypeAndShapeInfo* tensor_info = OrtCastTypeInfoToTensorInfo(typeinfo);
		ONNXTensorElementDataType type = OrtGetTensorElementType(tensor_info);
		printf("Input %d : type=%d\n", i, type);

		// print input shapes
		size_t num_dims = OrtGetNumOfDimensions(tensor_info);
		int64_t* dims = (int64_t*)malloc(num_dims * sizeof(int64_t));

		printf("Input %d : num_dims=%zu\n", i, num_dims);

		OrtGetDimensions(tensor_info, dims, num_dims);

		for (int j = 0; j < num_dims; j++)
			printf("Input %d : dim %d=%jd\n", i, j, dims[j]);

		OrtReleaseTypeInfo(typeinfo);
	}
	OrtReleaseAllocator(allocator);

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

	size_t input_dims[] = { 1, 3, 224, 224 };
	size_t input_count = 3 * 224 * 224;     // input tensor count = product of dims
	float* input_data = (float *) malloc(sizeof(float) * input_count);
	const char* output_names[] = { "softmaxout_1"};

	// initialize input data with values in [0.0, 1.0]
	for (unsigned int i = 0; i < input_count; i++)
		input_data[i] = (float)i / (float)(input_count + 1);

	// create input tensor object from data values
	OrtAllocatorInfo* allocator_info;
	CHECK_STATUS(OrtCreateCpuAllocatorInfo(OrtArenaAllocator, OrtMemTypeDefault, &allocator_info));
	OrtValue* input_tensor = NULL;
	CHECK_STATUS(OrtCreateTensorWithDataAsOrtValue(allocator_info, input_data, input_count * sizeof(float), input_dims, 4, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
	assert(OrtIsTensor(input_tensor));
	OrtReleaseAllocatorInfo(allocator_info);

	// score model & input tensor, get back output tensor
	OrtValue* output_tensor = NULL;
	CHECK_STATUS(OrtRun(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor));
	assert(OrtIsTensor(output_tensor));

	// copy output tensor values to float array
	// model produces scores for 1000 classes
	float* floatarr = (float *) malloc(1000 * sizeof(float));
	OrtGetTensorMutableData(output_tensor, (void **) &floatarr);

	// score the model, and print scores for first 5 classes
	for (int i = 0; i < 5; i++)
		printf("Score for class [%d] =  %f\n", i, floatarr[i]);

	// Results should be as below...
	// Score for class[0] = 0.000045
	// Score for class[1] = 0.003846
	// Score for class[2] = 0.000125
	// Score for class[3] = 0.001180
	// Score for class[4] = 0.001317

	free(input_data);
	OrtReleaseValue(output_tensor);
	OrtReleaseValue(input_tensor);
	OrtReleaseEnv(env);
	printf("Done!\n");
	return 0;
}



