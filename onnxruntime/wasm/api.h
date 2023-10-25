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

struct OrtIoBinding;
using ort_io_binding_handle_t = OrtIoBinding*;

struct OrtSessionOptions;
using ort_session_options_handle_t = OrtSessionOptions*;

struct OrtRunOptions;
using ort_run_options_handle_t = OrtRunOptions*;

struct OrtValue;
using ort_tensor_handle_t = OrtValue*;

#ifdef ENABLE_TRAINING_APIS
struct OrtTrainingSession;
using ort_training_session_handle_t = OrtTrainingSession*;

struct OrtCheckpointState;
using ort_training_checkpoint_handle_t = OrtCheckpointState*;
#endif

extern "C" {

/**
 * perform global initialization. should be called only once.
 * @param num_threads number of total threads to use.
 * @param logging_level default logging level.
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtInit(int num_threads, int logging_level);

/**
 * get the last error.
 * @param error_code [out] a pointer to accept the error code.
 * @param error_message [out] a pointer to accept the error message. The message buffer is only available before any ORT API is called.
 */
void EMSCRIPTEN_KEEPALIVE OrtGetLastError(int* error_code, const char** error_message);

/**
 * create an instance of ORT session options.
 * assume that all enum type parameters, such as graph_optimization_level, execution_mode, and log_severity_level,
 * are checked and set properly at JavaScript.
 * @param graph_optimization_level disabled, basic, extended, or enable all
 * @param enable_cpu_mem_arena enable or disable cpu memory arena
 * @param enable_mem_pattern enable or disable memory pattern
 * @param execution_mode sequential or parallel execution mode
 * @param enable_profiling enable or disable profiling.
 * @param profile_file_prefix file prefix for profiling data. it's a no-op and for a future use.
 * @param log_id logger id for session output
 * @param log_severity_level verbose, info, warning, error or fatal
 * @param log_verbosity_level vlog level
 * @param optimized_model_filepath filepath of the optimized model to dump.
 * @returns a session option handle. Caller must release it after use by calling OrtReleaseSessionOptions().
 */
ort_session_options_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateSessionOptions(size_t graph_optimization_level,
                                                                          bool enable_cpu_mem_arena,
                                                                          bool enable_mem_pattern,
                                                                          size_t execution_mode,
                                                                          bool enable_profiling,
                                                                          const char* profile_file_prefix,
                                                                          const char* log_id,
                                                                          size_t log_severity_level,
                                                                          size_t log_verbosity_level,
                                                                          const char* optimized_model_filepath);

/**
 * append an execution provider for a session.
 * @param name the name of the execution provider
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtAppendExecutionProvider(ort_session_options_handle_t session_options,
                                                    const char* name);

/**
 * add a free dimension override for one dimension of a session's input.
 */
int EMSCRIPTEN_KEEPALIVE OrtAddFreeDimensionOverride(ort_session_options_handle_t session_options,
                                                     const char* dim_param_name,
                                                     int dim_value);

/**
 * store configurations for a session.
 * @param session_options a handle to session options created by OrtCreateSessionOptions
 * @param config_key configuration keys and value formats are defined in
 *                   include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h
 * @param config_value value for config_key
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtAddSessionConfigEntry(ort_session_options_handle_t session_options,
                                                  const char* config_key,
                                                  const char* config_value);

/**
 * release the specified ORT session options.
 */
void EMSCRIPTEN_KEEPALIVE OrtReleaseSessionOptions(ort_session_options_handle_t session_options);

/**
 * create an instance of ORT session.
 * @param data a pointer to a buffer that contains the ONNX or ORT format model.
 * @param data_length the size of the buffer in bytes.
 * @returns an ORT session handle. Caller must release it after use by calling OrtReleaseSession().
 */
ort_session_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateSession(void* data,
                                                           size_t data_length,
                                                           ort_session_options_handle_t session_options);

/**
 * release the specified ORT session.
 */
void EMSCRIPTEN_KEEPALIVE OrtReleaseSession(ort_session_handle_t session);

/**
 * get model's input count and output count.
 * @param session handle of the specified session
 * @param input_count [out] a pointer to a size_t variable to accept input_count.
 * @param output_count [out] a pointer to a size_t variable to accept output_count.
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtGetInputOutputCount(ort_session_handle_t session,
                                                size_t* input_count,
                                                size_t* output_count);

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
 * @param data for numeric tensor: a pointer to the tensor data buffer. for string tensor: a pointer to a C-Style null terminated string array.
 * @param data_length size of the buffer 'data' in bytes.
 * @param dims a pointer to an array of dims. the array should contain (dims_length) element(s).
 * @param dims_length the length of the tensor's dimension
 * @param data_location specify the memory location of the tensor data. 0 for CPU, 1 for GPU buffer.
 * @returns a tensor handle. Caller must release it after use by calling OrtReleaseTensor().
 */
ort_tensor_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateTensor(int data_type, void* data, size_t data_length, size_t* dims, size_t dims_length, int data_location);

/**
 * get type, shape info and data of the specified tensor.
 * @param tensor handle of the tensor.
 * @param data_type [out] specify the memory to write data type
 * @param data [out] specify the memory to write the tensor data. for string tensor: an array of C-Style null terminated string.
 * @param dims [out] specify the memory to write address of the buffer containing value of each dimension.
 * @param dims_length [out] specify the memory to write dims length
 * @remarks following temporary buffers are allocated during the call. Caller must release the buffers after use by calling OrtFree():
 *           'dims' (for all types of tensor), 'data' (only for string tensor)
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtGetTensorData(ort_tensor_handle_t tensor, int* data_type, void** data, size_t** dims, size_t* dims_length);

/**
 * release the specified tensor.
 */
void EMSCRIPTEN_KEEPALIVE OrtReleaseTensor(ort_tensor_handle_t tensor);

/**
 * create an instance of ORT run options.
 * @param log_severity_level verbose, info, warning, error or fatal
 * @param log_verbosity_level vlog level
 * @param terminate if true, all incomplete OrtRun calls will exit as soon as possible
 * @param tag tag for this run
 * @returns a run option handle. Caller must release it after use by calling OrtReleaseRunOptions().
 */
ort_run_options_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateRunOptions(size_t log_severity_level,
                                                                  size_t log_verbosity_level,
                                                                  bool terminate,
                                                                  const char* tag);

/**
 * set a single run configuration entry
 * @param run_options a handle to run options created by OrtCreateRunOptions
 * @param config_key configuration keys and value formats are defined in
 *                   include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h
 * @param config_value value for config_key
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtAddRunConfigEntry(ort_run_options_handle_t run_options,
                                              const char* config_key,
                                              const char* config_value);

/**
 * release the specified ORT run options.
 */
void EMSCRIPTEN_KEEPALIVE OrtReleaseRunOptions(ort_run_options_handle_t run_options);

/**
 * create an instance of ORT IO binding.
 */
ort_io_binding_handle_t EMSCRIPTEN_KEEPALIVE OrtCreateBinding(ort_session_handle_t session);

/**
 * bind an input tensor to the IO binding instance. A cross device copy will be performed if necessary.
 * @param io_binding handle of the IO binding
 * @param name name of the input
 * @param input handle of the input tensor
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtBindInput(ort_io_binding_handle_t io_binding,
                                      const char* name,
                                      ort_tensor_handle_t input);

/**
 * bind an output tensor or location to the IO binding instance.
 * @param io_binding handle of the IO binding
 * @param name name of the output
 * @param output handle of the output tensor. nullptr for output location binding.
 * @param output_location specify the memory location of the output tensor data.
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtBindOutput(ort_io_binding_handle_t io_binding,
                                       const char* name,
                                       ort_tensor_handle_t output,
                                       int output_location);

/**
 * clear all bound outputs.
 */
void EMSCRIPTEN_KEEPALIVE OrtClearBoundOutputs(ort_io_binding_handle_t io_binding);

/**
 * release the specified ORT IO binding.
 */
void EMSCRIPTEN_KEEPALIVE OrtReleaseBinding(ort_io_binding_handle_t io_binding);

/**
 * inference the model.
 * @param session handle of the specified session
 * @param io_binding handle of the IO binding
 * @param run_options handle of the run options
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtRunWithBinding(ort_session_handle_t session,
                                           ort_io_binding_handle_t io_binding,
                                           size_t output_count,
                                           ort_tensor_handle_t* outputs,
                                           ort_run_options_handle_t run_options);

/**
 * inference the model.
 * @param session handle of the specified session
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtRun(ort_session_handle_t session,
                                const char** input_names,
                                const ort_tensor_handle_t* inputs,
                                size_t input_count,
                                const char** output_names,
                                size_t output_count,
                                ort_tensor_handle_t* outputs,
                                ort_run_options_handle_t run_options);

/**
 * end profiling.
 * @param session handle of the specified session
 * @returns a pointer to a buffer which contains C-style string of profile filename.
 * Caller must release the C style string after use by calling OrtFree().
 */
char* EMSCRIPTEN_KEEPALIVE OrtEndProfiling(ort_session_handle_t session);

// Training API Section

#ifdef ENABLE_TRAINING_APIS
/**
 * @brief Load the checkpoint for training.
 *
 * @param checkpoint_data_buffer pointer to a buffer containing the CheckpointState
 * @param checkpoint_size size of the CheckpointState in bytes
 * @return ort_training_checkpoint_handle_t
 */
ort_training_checkpoint_handle_t EMSCRIPTEN_KEEPALIVE OrtTrainingLoadCheckpoint(void* checkpoint_data_buffer, size_t checkpoint_size);

/**
 * @brief Release the specified ORT training checkpoint state.
 *
 * @param training_checkpoint_state_handle handle for the CheckpointState
 */
void EMSCRIPTEN_KEEPALIVE OrtTrainingReleaseCheckpoint(ort_training_checkpoint_handle_t training_checkpoint_state_handle);

/**
 * Creates an instance of a training session that can be used to begin or resume training from a given checkpoint state
 * for the given onnx models.
 * @param options Session options that the user can customize for this training session.
 * @param training_checkpoint_state_handle Training states that the training session uses as a starting point for training.
 * @param train_model pointer to a buffer containing the ONNX training model
 * @param train_size size of the train_model buffer in bytes
 * @param eval_model pointer to a buffer containing the ONNX evaluation model
 * @param eval_size size of the eval_model buffer in bytes
 * @param optimizer_model pointer to a buffer containing the ONNX optimizer model
 * @param optimizer_size size of the optimizer_model buffer in bytes
 * @return a handle of the ORT training session
 *
 */
ort_training_session_handle_t EMSCRIPTEN_KEEPALIVE OrtTrainingCreateSession(ort_session_options_handle_t options,
                                                                            ort_training_checkpoint_handle_t training_checkpoint_state_handle,
                                                                            void* train_model,
                                                                            size_t train_size,
                                                                            void* eval_model,
                                                                            size_t eval_size,
                                                                            void* optimizer_model,
                                                                            size_t optimizer_size);

/**
 * Resets the gradients of all trainable parameters to zero for the specified TrainingSession
 * @param training_handle handle of the training session
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtTrainingLazyResetGrad(ort_training_session_handle_t training_handle);

/**
 * @brief Run a single training step.
 *
 * @param training_handle session handle of the specified session
 * @param inputs user inputs to the training model
 * @param input_count number of user inputs to the training model
 * @param outputs [out] user outputs computed by train step
 * @param output_count [out] number of user outputs expected from this train step
 * @param run_options handle of the run options
 * @return int ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtTrainingRunTrainStep(ort_training_session_handle_t training_handle,
                                                 ort_tensor_handle_t* inputs, size_t input_count,
                                                 ort_tensor_handle_t* outputs,
                                                 size_t output_count,
                                                 ort_run_options_handle_t run_options = nullptr);

/**
 * Performs weight updates for the trainable parameters in the given training session using the optimizer model.
 * @param training_handle handle of the training session
 * @param run_options optional parameter of run options for this training step
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtTrainingOptimizerStep(ort_training_session_handle_t training_handle,
                                                  ort_run_options_handle_t run_options = nullptr);

/**
 * Computes outputs for the eval model associated with the given training session.
 * @param training_handle handle of the training session
 * @param options run options for this eval step
 * @param input_count number of user inputs to the eval model
 * @param inputs the user inputs to the eval model
 * @param output_count [out] number of user outputs expected from this eval step
 * @param outputs [out] user outputs computed by the eval step
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtTrainingEvalStep(ort_training_session_handle_t training_handle,
                                             ort_tensor_handle_t* inputs,
                                             size_t input_count,
                                             ort_tensor_handle_t* outputs,
                                             size_t output_count,
                                             ort_run_options_handle_t options = nullptr);

/**
 * Retrieves the size of all parameters for the training state.
 * When the trainable_only argument is true, the size is calculated for trainable params only.
 *
 * @param training_handle handle of the training session
 * @param param_size [out] size of all parameter elements
 * @param trainable_only skips non-trainable parameters when true.
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtTrainingGetParametersSize(ort_training_session_handle_t training_handle,
                                                      size_t* param_size,
                                                      bool trainable_only);

/**
 * Copy all parameters to a contiguous buffer held by the argument parameters_buffer
 *
 * User is responsible for allocating and freeing resources used by the parameters_buffer.
 * Parameter ordering is preserved.
 *
 * @param training_handle handle of the training session
 * @param parameters_buffer [out] pre-allocated OrtValue buffer to copy onto. Must be same size as results of
 *                          GetParametersSize api call
 * @param parameter_count number of parameters expected in the parameters_buffer
 * @param trainable_only whether to skip non-trainable parameters
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtTrainingCopyParametersToBuffer(ort_training_session_handle_t training_handle,
                                                           ort_tensor_handle_t parameters_buffer,
                                                           size_t parameter_count,
                                                           bool trainable_only);

/**
 * Copy parameters values from given contiguous buffer held by parameters_buffer to the training state.
 * Parameter ordering is preserved.
 * @param training_handle handle of the training session
 * @param parameters_buffer OrtValue buffer to copy from. Must be same size as results of
 *                          GetParametersSize api call
 * @param parameter_count number of parameters expected in the parameters_buffer
 * @param trainable_only whether to skip non-trainable parameters
 * @returns ORT error code. If not zero, call OrtGetLastError() to get detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtTrainingCopyParametersFromBuffer(ort_training_session_handle_t training_handle,
                                                             ort_tensor_handle_t parameters_buffer,
                                                             size_t parameter_count,
                                                             bool trainable_only);

/**
 * Gets the input count and output count of the training or eval model associated with the given training handle.
 * @param traning_handle handle of the traning session
 * @param input_count [out] a pointer to a size_t variable to accept input_count
 * @param output_count [out] a pointer to a size_t variable to accept output_count
 * @param isEvalModel when false, returns input & output count of the training model. When true, returns input & output
 *                    count of the eval model.
 * @returns ORT error code. If not zero, call OrtGetLastError() to get a detailed error message.
 */
int EMSCRIPTEN_KEEPALIVE OrtTrainingGetInputOutputCount(ort_training_session_handle_t training_handle,
                                                        size_t* input_count,
                                                        size_t* output_count,
                                                        bool isEvalModel);

/**
 * Gets the input or output name at the specified index associated with the training or eval model from the
 * given training session.
 * @param traning_handle handle of the traning session
 * @param index the input or output index
 * @param isInput if true, this method retrieves an input name. If false, this method retrieves an output name.
 * @param isEvalModel when false, returns input & output names of the training model. When true, returns input & output
 *                    names of the eval model.
 * @returns a pointer to a buffer which contains C-style string. Caller must release the C style string after use by
 */
char* EMSCRIPTEN_KEEPALIVE OrtTrainingGetInputOutputName(ort_training_session_handle_t training_handle,
                                                         size_t index,
                                                         bool isInput,
                                                         bool isEvalModel);

/**
 * @brief Release the specified ORT training session.
 *
 * @param training_session_handle handle of the training session
 */
void EMSCRIPTEN_KEEPALIVE OrtTrainingReleaseSession(ort_training_session_handle_t training_session_handle);

#endif
};
