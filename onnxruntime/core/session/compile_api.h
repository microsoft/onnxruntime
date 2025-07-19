// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/session/onnxruntime_c_api.h"

namespace OrtCompileAPI {

// implementation that returns the API struct
ORT_API(const OrtCompileApi*, GetCompileApi);

ORT_API(void, ReleaseModelCompilationOptions, _Frees_ptr_opt_ OrtModelCompilationOptions*);
ORT_API_STATUS_IMPL(CreateModelCompilationOptionsFromSessionOptions, _In_ const OrtEnv* env,
                    _In_ const OrtSessionOptions* session_options, _Outptr_ OrtModelCompilationOptions** out);
ORT_API_STATUS_IMPL(ModelCompilationOptions_SetInputModelPath, _In_ OrtModelCompilationOptions* model_compile_options,
                    _In_ const ORTCHAR_T* input_model_path);
ORT_API_STATUS_IMPL(ModelCompilationOptions_SetInputModelFromBuffer, _In_ OrtModelCompilationOptions* model_compile_options,
                    _In_ const void* input_model_data, size_t input_model_data_size);
ORT_API_STATUS_IMPL(ModelCompilationOptions_SetOutputModelPath, _In_ OrtModelCompilationOptions* model_compile_options,
                    _In_ const ORTCHAR_T* output_model_path);
ORT_API_STATUS_IMPL(ModelCompilationOptions_SetOutputModelExternalInitializersFile,
                    _In_ OrtModelCompilationOptions* model_compile_options,
                    _In_ const ORTCHAR_T* external_initializers_file_path,
                    size_t external_initializer_size_threshold);
ORT_API_STATUS_IMPL(ModelCompilationOptions_SetOutputModelBuffer, _In_ OrtModelCompilationOptions* model_compile_options,
                    _Inout_ OrtAllocator* allocator, void** output_model_buffer_ptr, size_t* output_model_buffer_size_ptr);
ORT_API_STATUS_IMPL(ModelCompilationOptions_SetEpContextEmbedMode, _In_ OrtModelCompilationOptions* model_compile_options,
                    bool embed_ep_context_in_model);
ORT_API_STATUS_IMPL(CompileModel, _In_ const OrtEnv* env, _In_ const OrtModelCompilationOptions* model_options);
ORT_API_STATUS_IMPL(ModelCompilationOptions_SetFlags, _In_ OrtModelCompilationOptions* model_options,
                    size_t flags);
ORT_API_STATUS_IMPL(ModelCompilationOptions_SetEpContextBinaryInformation, _In_ OrtModelCompilationOptions* model_compile_options,
                    _In_ const ORTCHAR_T* output_dir, _In_ const ORTCHAR_T* model_name);
ORT_API_STATUS_IMPL(ModelCompilationOptions_SetOutputModelWriteFunc,
                    _In_ OrtModelCompilationOptions* model_compile_options,
                    _In_ OrtWriteBufferFunc write_func, _In_ void* state);
ORT_API_STATUS_IMPL(ModelCompilationOptions_SetOutputModelHandleInitializerFunc,
                    _In_ OrtModelCompilationOptions* model_compile_options,
                    _In_ OrtHandleInitializerDataFunc handle_initializer_func, _In_ void* state);

}  // namespace OrtCompileAPI
