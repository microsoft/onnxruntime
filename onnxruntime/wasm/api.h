#pragma once

#include <emscripten.h>

#include <stddef.h>

namespace Ort {
struct Session;
}
typedef Ort::Session* ort_session_handle_t;

struct OrtValue;
typedef OrtValue* ort_tensor_handle_t;

extern "C" {

/**
 * perform global initialization. should be called only once.
 */
void EMSCRIPTEN_KEEPALIVE ort_init();

/**
 * create an instance of ORT session.
 * @param data a pointer to a buffer that contains the model.
 * @param data_length the size of the buffer in bytes.
 * @returns a handle of the ORT session.
 */
ort_session_handle_t EMSCRIPTEN_KEEPALIVE ort_create_session(void* data, size_t data_length);

/**
 * release the specified ORT session.
 */
void EMSCRIPTEN_KEEPALIVE ort_release_session(ort_session_handle_t session);

size_t EMSCRIPTEN_KEEPALIVE ort_get_input_count(ort_session_handle_t session);
size_t EMSCRIPTEN_KEEPALIVE ort_get_output_count(ort_session_handle_t session);

/**
 * get the model's input name.
 * @param session handle of the specified session
 * @param index the input index
 * @returns a pointer to a buffer which contains C-style string. Caller should release the buffer after use by calling ort_free().
 */
char* EMSCRIPTEN_KEEPALIVE ort_get_input_name(ort_session_handle_t session, size_t index);
/**
 * get the model's output name.
 * @param session handle of the specified session
 * @param index the output index
 * @returns a pointer to a buffer which contains C-style string. Caller should release the buffer after use by calling ort_free().
 */
char* EMSCRIPTEN_KEEPALIVE ort_get_output_name(ort_session_handle_t session, size_t index);

void EMSCRIPTEN_KEEPALIVE ort_free(void* ptr);

/**
 * create an instance of ORT tensor.
 * @param data_type data type defined in enum ONNXTensorElementDataType.
 * @param data a pointer to the tensor data.
 * @param data_length size of the tensor data in bytes.
 * @param dims a pointer to an array of dims. the array should contain (dims_length) element(s).
 * @param dims_length the length of the tensor's dimension
 * @returns a handle of the tensor.
 */
ort_tensor_handle_t EMSCRIPTEN_KEEPALIVE ort_create_tensor(int data_type, void* data, size_t data_length, size_t* dims, size_t dims_length);

/**
 * get type, shape info and data of the specified tensor.
 * @param tensor handle of the tensor.
 * @param data_type [out] specify the memory to write data type
 * @param data [out] specify the memory to write the tensor data
 * @param dims [out] specify the memory to write address of the buffer containing value of each dimension.
 * @param dims_length [out] specify the memory to write dims length
 * @remarks a temporary buffer 'dims' is allocated during the call. Caller should release the buffer after use by calling ort_free().
 */
void EMSCRIPTEN_KEEPALIVE ort_get_tensor_data(ort_tensor_handle_t tensor, int* data_type, void** data, size_t** dims, size_t* dims_length);

/**
 * release the specified tensor.
 */
void EMSCRIPTEN_KEEPALIVE ort_release_tensor(ort_tensor_handle_t tensor);

/**
 * inference the model.
 * @param session handle of the specified session
 */
void EMSCRIPTEN_KEEPALIVE ort_run(ort_session_handle_t session, const char** input_names, const ort_tensor_handle_t* inputs, size_t input_count, const char** output_names, size_t output_count, ort_tensor_handle_t* outputs);

};
