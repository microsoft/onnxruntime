#pragma once

#include <emscripten.h>

#include <stddef.h>

typedef size_t ort_session_handle_t;
typedef size_t ort_tensor_t;
typedef size_t ort_tensor_metadata_t;
typedef size_t ort_model_data_t;
typedef size_t ort_run_context_t;

extern "C" {

void EMSCRIPTEN_KEEPALIVE ort_init();

ort_session_handle_t EMSCRIPTEN_KEEPALIVE ort_create_session(ort_model_data_t data);
void EMSCRIPTEN_KEEPALIVE ort_release_session(ort_session_handle_t session);

ort_tensor_t EMSCRIPTEN_KEEPALIVE ort_create_tensor(ort_tensor_metadata_t metadata);
ort_tensor_metadata_t EMSCRIPTEN_KEEPALIVE ort_get_tensor_metadata(ort_tensor_t tensor);
void EMSCRIPTEN_KEEPALIVE ort_release_tensor_metadata(ort_tensor_metadata_t metadata);
void EMSCRIPTEN_KEEPALIVE ort_release_tensor(ort_tensor_t tensor);

void EMSCRIPTEN_KEEPALIVE ort_run(ort_session_handle_t session, ort_run_context_t context);
};
