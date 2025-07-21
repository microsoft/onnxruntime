// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/compile_api.h"

#if !defined(ORT_MINIMAL_BUILD)
#include <memory>
#include <string>

#include "core/common/common.h"
#include "core/session/allocator_adapters.h"
#include "core/framework/error_code_helper.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/inference_session.h"
#include "core/session/model_compilation_options.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
#include "core/session/utils.h"
#else
#include "core/common/common.h"
#include "core/framework/error_code_helper.h"
#include "core/session/ort_apis.h"
#endif  // !defined(ORT_MINIMAL_BUILD)

using namespace onnxruntime;

ORT_API(void, OrtCompileAPI::ReleaseModelCompilationOptions,
        _Frees_ptr_opt_ OrtModelCompilationOptions* ort_model_compile_options) {
#if !defined(ORT_MINIMAL_BUILD)
  delete reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);
#else
  ORT_UNUSED_PARAMETER(ort_model_compile_options);
#endif  // !defined(ORT_MINIMAL_BUILD)
}

ORT_API_STATUS_IMPL(OrtCompileAPI::CreateModelCompilationOptionsFromSessionOptions, _In_ const OrtEnv* env,
                    _In_ const OrtSessionOptions* session_options, _Outptr_ OrtModelCompilationOptions** out) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  if (env == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "The env argument must be a non-null pointer");
  }

  if (session_options == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "The session_options argument must be a non-null pointer");
  }

  auto model_compile_options = std::make_unique<onnxruntime::ModelCompilationOptions>(env->GetEnvironment(),
                                                                                      *session_options);
  *out = reinterpret_cast<OrtModelCompilationOptions*>(model_compile_options.release());
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(env);
  ORT_UNUSED_PARAMETER(session_options);
  ORT_UNUSED_PARAMETER(out);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Compile API is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetInputModelPath,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    const ORTCHAR_T* input_model_path) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);
  std::string model_path = PathToUTF8String(input_model_path);

  if (model_path.empty()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid input model: path string is empty");
  }

  model_compile_options->SetInputModelPath(model_path);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_model_compile_options);
  ORT_UNUSED_PARAMETER(input_model_path);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Compile API is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetInputModelFromBuffer,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    const void* input_model_data, size_t input_model_data_size) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);

  if (input_model_data == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid input model: data pointer is null");
  }

  if (input_model_data_size == 0) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid input model: data size is 0");
  }

  model_compile_options->SetInputModelFromBuffer(input_model_data, input_model_data_size);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_model_compile_options);
  ORT_UNUSED_PARAMETER(input_model_data);
  ORT_UNUSED_PARAMETER(input_model_data_size);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Compile API is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetOutputModelPath,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    const ORTCHAR_T* output_model_path) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);

  std::string model_path = PathToUTF8String(output_model_path);
  if (model_path.empty()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid output model path: path is empty");
  }

  ORT_API_RETURN_IF_STATUS_NOT_OK(model_compile_options->SetOutputModelPath(model_path));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_model_compile_options);
  ORT_UNUSED_PARAMETER(output_model_path);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Compile API is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetEpContextBinaryInformation,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    const ORTCHAR_T* output_directory,
                    const ORTCHAR_T* model_name) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);

  std::filesystem::path output_directory_path = output_directory;
  if (output_directory_path.empty()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid output directory: path is empty");
  }

  std::filesystem::path model_name_path = model_name;
  if (model_name_path.empty()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid model name: string is empty");
  }

  ORT_API_RETURN_IF_STATUS_NOT_OK(model_compile_options->SetEpContextBinaryInformation(output_directory_path,
                                                                                       model_name_path));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_model_compile_options);
  ORT_UNUSED_PARAMETER(output_directory);
  ORT_UNUSED_PARAMETER(model_name);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Compile API is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetOutputModelExternalInitializersFile,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    const ORTCHAR_T* external_initializers_file_path,
                    size_t external_initializer_size_threshold) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  std::string initializers_file_path = PathToUTF8String(external_initializers_file_path);
  if (initializers_file_path.empty()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid external initializer file: path is empty");
  }

  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);
  model_compile_options->SetOutputModelExternalInitializersFile(initializers_file_path, external_initializer_size_threshold);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_model_compile_options);
  ORT_UNUSED_PARAMETER(external_initializers_file_path);
  ORT_UNUSED_PARAMETER(external_initializer_size_threshold);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Compile API is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetOutputModelBuffer,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    _Inout_ OrtAllocator* ort_allocator, void** output_model_data_ptr, size_t* output_model_data_size_ptr) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);

  if (output_model_data_ptr == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid output model buffer: data pointer is null");
  }

  if (output_model_data_size_ptr == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid output model buffer: size pointer is null");
  }

  if (ort_allocator == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid allocator for output model buffer: allocator pointer is null");
  }

  onnxruntime::AllocatorPtr allocator = std::make_shared<onnxruntime::IAllocatorImplWrappingOrtAllocator>(ort_allocator);
  ORT_API_RETURN_IF_STATUS_NOT_OK(model_compile_options->SetOutputModelBuffer(std::move(allocator),
                                                                              output_model_data_ptr,
                                                                              output_model_data_size_ptr));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_model_compile_options);
  ORT_UNUSED_PARAMETER(ort_allocator);
  ORT_UNUSED_PARAMETER(output_model_data_ptr);
  ORT_UNUSED_PARAMETER(output_model_data_size_ptr);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Compile API is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetOutputModelWriteFunc,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    _In_ OrtWriteBufferFunc write_func, _In_ void* state) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);

  if (write_func == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "OrtWriteBufferFunc function for output model is null");
  }

  model_compile_options->SetOutputModelWriteFunc(write_func, state);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_model_compile_options);
  ORT_UNUSED_PARAMETER(write_func);
  ORT_UNUSED_PARAMETER(state);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Compile API is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}
ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetOutputModelHandleInitializerFunc,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    _In_ OrtHandleInitializerDataFunc handle_initializer_func, _In_ void* state) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);

  if (handle_initializer_func == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "OrtHandleInitializerDataFunc function for output model is null");
  }

  model_compile_options->SetOutputModelHandleInitializerFunc(handle_initializer_func, state);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_model_compile_options);
  ORT_UNUSED_PARAMETER(handle_initializer_func);
  ORT_UNUSED_PARAMETER(state);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Compile API is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetEpContextEmbedMode,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    bool embed_ep_context_in_model) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);
  ORT_API_RETURN_IF_STATUS_NOT_OK(model_compile_options->SetEpContextEmbedMode(embed_ep_context_in_model));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_model_compile_options);
  ORT_UNUSED_PARAMETER(embed_ep_context_in_model);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Compile API is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetEpContextDataWriteFunc,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    _In_ OrtWriteEpContextDataFunc write_func, _In_ void* state) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);

  if (write_func == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT,
                                 "OrtWriteEpContextDataFunc function for writing out an EPContext node's "
                                 "binary data is NULL");
  }

  model_compile_options->SetEpContextDataWriteFunc(write_func, state);
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_model_compile_options);
  ORT_UNUSED_PARAMETER(write_func);
  ORT_UNUSED_PARAMETER(state);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Compile API is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetFlags,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options, size_t flags) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);
  ORT_API_RETURN_IF_STATUS_NOT_OK(model_compile_options->SetFlags(flags));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(ort_model_compile_options);
  ORT_UNUSED_PARAMETER(flags);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Compile API is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::CompileModel, _In_ const OrtEnv* env,
                    _In_ const OrtModelCompilationOptions* ort_model_compile_options) {
  API_IMPL_BEGIN
#if !defined(ORT_MINIMAL_BUILD)
  auto model_compile_options = reinterpret_cast<const onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);
  ORT_API_RETURN_IF_STATUS_NOT_OK(onnxruntime::CompileModel(env->GetEnvironment(), *model_compile_options));
  return nullptr;
#else
  ORT_UNUSED_PARAMETER(env);
  ORT_UNUSED_PARAMETER(ort_model_compile_options);
  return OrtApis::CreateStatus(ORT_NOT_IMPLEMENTED, "Compile API is not supported in this build");
#endif  // !defined(ORT_MINIMAL_BUILD)
  API_IMPL_END
}

static constexpr OrtCompileApi ort_compile_api = {
    // NOTE: Application compatibility with newer versions of ORT depends on the Api order within this struct so
    // all new functions must be added at the end, and no functions that already exist in an officially released version
    // of ORT can be reordered or removed.

    &OrtCompileAPI::ReleaseModelCompilationOptions,
    &OrtCompileAPI::CreateModelCompilationOptionsFromSessionOptions,
    &OrtCompileAPI::ModelCompilationOptions_SetInputModelPath,
    &OrtCompileAPI::ModelCompilationOptions_SetInputModelFromBuffer,
    &OrtCompileAPI::ModelCompilationOptions_SetOutputModelPath,
    &OrtCompileAPI::ModelCompilationOptions_SetOutputModelExternalInitializersFile,
    &OrtCompileAPI::ModelCompilationOptions_SetOutputModelBuffer,
    &OrtCompileAPI::ModelCompilationOptions_SetEpContextEmbedMode,
    &OrtCompileAPI::CompileModel,
    // End of Version 22 - DO NOT MODIFY ABOVE

    &OrtCompileAPI::ModelCompilationOptions_SetFlags,
    &OrtCompileAPI::ModelCompilationOptions_SetEpContextBinaryInformation,
    &OrtCompileAPI::ModelCompilationOptions_SetOutputModelWriteFunc,
    &OrtCompileAPI::ModelCompilationOptions_SetOutputModelHandleInitializerFunc,
    &OrtCompileAPI::ModelCompilationOptions_SetEpContextDataWriteFunc,
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtCompileApi, CompileModel) / sizeof(void*) == 8,
              "Size of version 22 Api cannot change");  // initial version in ORT 1.22

ORT_API(const OrtCompileApi*, OrtCompileAPI::GetCompileApi) {
  return &ort_compile_api;
}
