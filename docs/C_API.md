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

## Very simplified outline of how to use it

1. Include [onnxruntime_c_api.h](/include/onnxruntime/core/session/onnxruntime_c_api.h).
2. Call OrtCreateEnv
3. Create Session: OrtCreateSession(env, model_uri, nullptr,...)
4. Create Tensor
   1) OrtCreateAllocatorInfo
   2) OrtCreateTensorWithDataAsONNXValue
5. OrtRun

## Sample code

The example below shows a sample run using the SqueezeNet model from ONNX model zoo, including dynamically reading model inputs, outputs, shape and type information, as well as running a sample vector and fetching the resulting class probabilities for inspection. 


```c
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
//

#include "pch.h"
#include <assert.h>
#include <core/session/onnxruntime_c_api.h>
#include <core/providers/cpu/cpu_provider_factory.h>
#include <stdlib.h>
#include <stdio.h>

#define ORT_ABORT_ON_ERROR(expr)                         \
  do {                                                   \
    OrtStatus* onnx_status = (expr);                     \
    if (onnx_status != NULL) {                           \
      const char* msg = OrtGetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                      \
      OrtReleaseStatus(onnx_status);                     \
      abort();                                           \
    }                                                    \
  } while (0);


using namespace std;

namespace sample_c_api
{
	// initialized from model metadata 
	char** input_names;     // { "data_0"}
	char** output_names;    // { "softmaxout_1"}
	size_t input_num_dims;  // 4
	size_t* input_dims;     // {1, 3, 224, 224 } 
	size_t input_tensor_element_count;  // 150328
	size_t output_tensor_element_count; // 1000

	// generate sample tensor data
	float* get_float_data(size_t count)
	{
		float* floatarr = (float *)malloc(sizeof(float) * count);
		for (unsigned int i = 0; i < count; i++)
			floatarr[i] = (float)i / (float)(count + 1);
		return floatarr;
	}

	// score the model using sample data. 
	float* run_inference(OrtSession* session)
	{
		// get sample data
		float* input_data = get_float_data(input_tensor_element_count);
		const size_t input_size = input_tensor_element_count * sizeof(float);

		// create input tensor
		OrtAllocatorInfo* allocator_info;
		ORT_ABORT_ON_ERROR(OrtCreateCpuAllocatorInfo(OrtArenaAllocator, OrtMemTypeDefault, &allocator_info));
		OrtValue* input_tensor = NULL;
		ORT_ABORT_ON_ERROR(OrtCreateTensorWithDataAsOrtValue(allocator_info, input_data, input_size, input_dims, input_num_dims, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
		assert(input_tensor != NULL);
		assert(OrtIsTensor(input_tensor));
		OrtReleaseAllocatorInfo(allocator_info);

        // score model, receive output as a tensor
		OrtValue* output_tensor = NULL;
		ORT_ABORT_ON_ERROR(OrtRun(session, NULL, input_names, (const OrtValue* const*)&input_tensor, 1, output_names, 1, &output_tensor));
		assert(output_tensor != NULL);
		assert(OrtIsTensor(output_tensor));

		// copy the output tensor into float array
		void* floatarr = malloc(output_tensor_element_count * sizeof(float));
		OrtGetTensorMutableData(output_tensor, &floatarr);

		// free resources and return float values
		free(input_data);
		OrtReleaseValue(output_tensor);
		OrtReleaseValue(input_tensor);
		return (float*) floatarr;
	}

	// prints model input names and shapes
	void print_input_info(OrtSession* session)
	{
		size_t num_inputs;
		OrtStatus* status;
		OrtAllocator* allocator;
		OrtCreateDefaultAllocator(&allocator);

		// print number of model inputs
		status = OrtSessionGetInputCount(session, &num_inputs);
		input_names = (char**)malloc(num_inputs * sizeof(char*));
		printf("Number of inputs = %zu\n", num_inputs);

		// iterate over all inputs
		for (int i = 0; i < num_inputs; i++)
		{
			// print input names
			char* input_name;
			status = OrtSessionGetInputName(session, i, allocator, &input_name);
			printf("Input %d : name=%s\n", i, input_name);
			input_names[i] = input_name;

			// print input types 
			OrtTypeInfo* typeinfo;
			status = OrtSessionGetInputTypeInfo(session, i, &typeinfo);
			const OrtTensorTypeAndShapeInfo* tensor_info = OrtCastTypeInfoToTensorInfo(typeinfo);
			ONNXTensorElementDataType type = OrtGetTensorElementType(tensor_info);
			printf("Input %d : type=%d\n", i, type);

			// print input shapes
			size_t num_dims = OrtGetNumOfDimensions(tensor_info);
			int64_t* dims = (int64_t*)malloc(num_dims * sizeof(int64_t));

			// store tensor shape info in global
			input_num_dims = num_dims;
			input_dims = (size_t *) dims;
			printf("Input %d : num_dims=%zu\n", i, num_dims);

			OrtGetDimensions(tensor_info, dims, num_dims);
			input_tensor_element_count = 1;
			for (int j = 0; j < num_dims; j++)
			{
				printf("Input %d : dim %d=%jd\n", i, j, dims[j]);
				input_tensor_element_count *= dims[j];
			}
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
	}

	// prints model output names and shapes
	void print_output_info(OrtSession* session)
	{
		size_t num_outputs;
		OrtStatus* status;
		OrtAllocator* allocator;
		OrtCreateDefaultAllocator(&allocator);

		//get number of model outputs
		status = OrtSessionGetOutputCount(session, &num_outputs);
		output_names = (char**)malloc(num_outputs * sizeof(char*));
		printf("Number of outputs = %zu\n", num_outputs);

		// iterate over all inputs
		for (int i = 0; i < num_outputs; i++)
		{
			// print output names
			char* output_name;
			status = OrtSessionGetOutputName(session, i, allocator, &output_name);
			printf("Output %d : name=%s\n", i, output_name);
			output_names[i] = output_name;

			// print output types 
			OrtTypeInfo* typeinfo;
			status = OrtSessionGetOutputTypeInfo(session, i, &typeinfo);
			const OrtTensorTypeAndShapeInfo* tensor_info = OrtCastTypeInfoToTensorInfo(typeinfo);
			ONNXTensorElementDataType type = OrtGetTensorElementType(tensor_info);
			printf("Output %d : type=%d\n", i, type);

			// print output shapes
			size_t num_dims = OrtGetNumOfDimensions(tensor_info);
			int64_t* dims = (int64_t*)malloc(num_dims * sizeof(int64_t));
			printf("Output %d : num_dims=%zu\n", i, num_dims);

			OrtGetDimensions(tensor_info, dims, num_dims);
			output_tensor_element_count = 1;
			for (int j = 0; j < num_dims; j++)
			{
				printf("Output %i : dim %d=%jd\n", i, j, dims[j]);
				output_tensor_element_count *= dims[j];
			}
			OrtReleaseTypeInfo(typeinfo);
		}
		OrtReleaseAllocator(allocator);

		// Results should be...
		// Number of outputs = 1
		// Output 0 : name = softmaxout_1
		// Output 0 : type = 1
		// Output 0 : num_dims = 4
		// Output 0 : dim 0 = 1
		// Output 0 : dim 1 = 1000
		// Output 0 : dim 2 = 1
		// Output 0 : dim 3 = 1
	}

	int test(int argc, char *argv[])
	{
		// set up enviroment...one enviroment per process
		OrtEnv* env;
		ORT_ABORT_ON_ERROR(OrtCreateEnv(ORT_LOGGING_LEVEL_WARNING, "test", &env));

		// set up session options
		OrtSessionOptions* session_option = OrtCreateSessionOptions();
		OrtSetSessionThreadPoolSize(session_option, 1);

		// create session and load model into memory
		// using squeezenet version 1.3 
		// URL = https://github.com/onnx/models/tree/master/squeezenet
		OrtSession* session;
		const wchar_t * model_path = L"model.onnx";
		ORT_ABORT_ON_ERROR(OrtCreateSession(env, model_path, session_option, &session));

		// print model input/output names and shapes
		print_input_info(session);
		print_output_info(session);

		// score the model, and print scores for first 5 classes
		float *floatarr = run_inference(session);
		for (int i = 0; i < 5; i++)
			printf("Score for class [%d] =  %f\n", i, floatarr[i]);

		// Results should be as below...
		// Score for class[0] = 0.000045
		// Score for class[1] = 0.003846
		// Score for class[2] = 0.000125
		// Score for class[3] = 0.001180
		// Score for class[4] = 0.001317

		OrtReleaseEnv(env);
		printf("Done!\n");
		return 0;
	}
}

int main(int argc, char *argv[])
{
	return sample_c_api::test(argc, argv);
}
