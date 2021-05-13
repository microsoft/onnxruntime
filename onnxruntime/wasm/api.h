// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// NOTE: This file contains declarations of exported functions as WebAssembly API.
// Unlike a normal C-API, the purpose of this API is to make emcc to generate correct exports for the WebAssembly. The
// macro "EMSCRIPTEN_KEEPALIVE" helps the compiler to mark the function as an exported funtion of the WebAssembly
// module. Users are expected to consume those functions from JavaScript side.

#pragma once

#include <emscripten.h>

#include <stddef.h>

struct OrtSession;
using ort_session_handle_t = OrtSession*;

struct OrtSessionOptions;
using ort_session_options_handle_t = OrtSessionOptions*;

struct OrtRunOptions;
using ort_run_options_handle_t = OrtRunOptions*;

struct OrtValue;
using ort_tensor_handle_t = OrtValue*;

extern "C" {

/**
 * perform global initialization. should be called only once.
 * @param numThreads number of total threads to use.
 * @param logging_level default logging level.
 */
int EMSCRIPTEN_KEEPALIVE OrtInit(int numThreads, int logging_level);

/**
 * create an instance of ORT session options.
 * @returns a pointer to a session option handle and must be freed by calling OrtReleaseSessionOptions().
 */
ort_session_options_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateSessionOptions();

/**
 * release the specified ORT session options.
 */
void EMSCRIPTEN_KEEPALIVE OrtReleaseSessionOptions(ort_session_options_handle_t session_options);

/**
 * set an optimization level for session.
 */
int EMSCRIPTEN_KEEPALIVE OrtSetSessionGraphOptimizationLevel(ort_session_options_handle_t session_options, size_t level);

/**
 * enable CPU memory arena for session.
 */
int EMSCRIPTEN_KEEPALIVE OrtEnableCpuMemArena(ort_session_options_handle_t session_options);

/**
 * disable CPU memory arena for session.
 */
int EMSCRIPTEN_KEEPALIVE OrtDisableCpuMemArena(ort_session_options_handle_t session_options);

/**
 * enable memory pattern for session.
 */
int EMSCRIPTEN_KEEPALIVE OrtEnableMemPattern(ort_session_options_handle_t session_options);

/**
 * disable memory pattern for session.
 */
int EMSCRIPTEN_KEEPALIVE OrtDisableMemPattern(ort_session_options_handle_t session_options);

/**
 * set an execution mode for session.
 */
int EMSCRIPTEN_KEEPALIVE OrtSetSessionExecutionMode(ort_session_options_handle_t session_options, size_t mode);

/**
 * set a log ID for session.
 */
int EMSCRIPTEN_KEEPALIVE OrtSetSessionLogId(ort_session_options_handle_t session_options, const char* logid);

/**
 * set a log severity level for session.
 */
int EMSCRIPTEN_KEEPALIVE OrtSetSessionLogSeverityLevel(ort_session_options_handle_t session_options, size_t level);

/**
 * create an instance of ORT session.
 * @param data a pointer to a buffer that contains the ONNX or ORT format model.
 * @param data_length the size of the buffer in bytes.
 * @returns a handle of the ORT session.
 */
ort_session_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateSession(void* data,
                                                           size_t data_length,
                                                           ort_session_options_handle_t session_options);

/**
 * release the specified ORT session.
 */
void EMSCRIPTEN_KEEPALIVE OrtReleaseSession(ort_session_handle_t session);

size_t EMSCRIPTEN_KEEPALIVE OrtGetInputCount(ort_session_handle_t session);
size_t EMSCRIPTEN_KEEPALIVE OrtGetOutputCount(ort_session_handle_t session);

/**
 * get the model's input name.
 * @param session handle of the specified session
 * @param index the input index
 * @returns a pointer to a buffer which contains C-style string. Caller must release the C style string after use by
 * calling OrtFree().
 */
char* EMSCRIPTEN_KEEPALIVE OrtGetInputName(ort_session_handle_t session, size_t index);
/**
 * get the model's output name.
 * @param session handle of the specified session
 * @param index the output index
 * @returns a pointer to a buffer which contains C-style string. Caller must release the C style string after use by
 * calling OrtFree().
 */
char* EMSCRIPTEN_KEEPALIVE OrtGetOutputName(ort_session_handle_t session, size_t index);

/**
 * free the specified buffer.
 * @param ptr a pointer to the buffer.
 */
void EMSCRIPTEN_KEEPALIVE OrtFree(void* ptr);

/**
 * create an instance of ORT tensor.
 * @param data_type data type defined in enum ONNXTensorElementDataType.
 * @param data a pointer to the tensor data.
 * @param data_length size of the tensor data in bytes.
 * @param dims a pointer to an array of dims. the array should contain (dims_length) element(s).
 * @param dims_length the length of the tensor's dimension
 * @returns a handle of the tensor.
 */
ort_tensor_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateTensor(int data_type, void* data, size_t data_length, size_t* dims, size_t dims_length);

/**
 * get type, shape info and data of the specified tensor.
 * @param tensor handle of the tensor.
 * @param data_type [out] specify the memory to write data type
 * @param data [out] specify the memory to write the tensor data
 * @param dims [out] specify the memory to write address of the buffer containing value of each dimension.
 * @param dims_length [out] specify the memory to write dims length
 * @remarks a temporary buffer 'dims' is allocated during the call. Caller must release the buffer after use by calling OrtFree().
 */
int EMSCRIPTEN_KEEPALIVE OrtGetTensorData(ort_tensor_handle_t tensor, int* data_type, void** data, size_t** dims, size_t* dims_length);

/**
 * release the specified tensor.
 */
void EMSCRIPTEN_KEEPALIVE OrtReleaseTensor(ort_tensor_handle_t tensor);

/**
 * create an instance of ORT run options.
 * @returns a pointer to a run option handle and must be freed by calling OrtReleaseRunOptions().
 */
ort_run_options_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateRunOptions();

/**
 * release the specified ORT run options.
 */
void EMSCRIPTEN_KEEPALIVE OrtReleaseRunOptions(ort_run_options_handle_t run_options);

/**
 * set log severity level for run.
 */
int EMSCRIPTEN_KEEPALIVE OrtRunOptionsSetRunLogSeverityLevel(ort_run_options_handle_t run_options, size_t level);

/**
 * set a tag for the Run() calls using this.
 */
int EMSCRIPTEN_KEEPALIVE OrtRunOptionsSetRunTag(ort_run_options_handle_t run_options, const char* tag);

/**
 * inference the model.
 * @param session handle of the specified session
 * @returns error code defined in enum OrtErrorCode
 */
int EMSCRIPTEN_KEEPALIVE OrtRun(ort_session_handle_t session,
                                const char** input_names,
                                const ort_tensor_handle_t* inputs,
                                size_t input_count,
                                const char** output_names,
                                size_t output_count,
                                ort_tensor_handle_t* outputs,
                                ort_run_options_handle_t run_options);
};
