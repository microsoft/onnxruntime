// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/compile_api.h"

#include <memory>
#include <string>
#include <utility>

#include "core/framework/error_code_helper.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/inference_session.h"
#include "core/session/model_compilation_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
#include "core/session/utils.h"

#define ORT_C_API_RETURN_IF_ERROR(expr)                 \
  do {                                                  \
    auto _status = (expr);                              \
    if ((!_status.IsOK())) return ToOrtStatus(_status); \
  } while (0)

using namespace onnxruntime;

ORT_API(void, OrtCompileAPI::ReleaseModelCompilationOptions,
        _Frees_ptr_opt_ OrtModelCompilationOptions* ort_model_compile_options) {
  delete reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);
}

ORT_API_STATUS_IMPL(OrtCompileAPI::CreateModelCompilationOptions, _In_ const OrtEnv* env,
                    _Outptr_ OrtModelCompilationOptions** out) {
  API_IMPL_BEGIN
  if (env == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "The env argument must be a non-null pointer");
  }

  auto model_compile_options = std::make_unique<onnxruntime::ModelCompilationOptions>();
  model_compile_options->env = env;
  model_compile_options->session_options_ = std::make_unique<OrtSessionOptions>();

  OrtSessionOptions* session_options = model_compile_options->GetSessionOptions();
  session_options->value.has_explicit_ep_context_gen_options = true;
  session_options->value.ep_context_gen_options.enable = true;
  session_options->value.ep_context_gen_options.always_generate = true;
  ORT_C_API_RETURN_IF_ERROR(session_options->value.config_options.AddConfigEntry(
      kOrtSessionOptionEpContextEnable, "1"));
  *out = reinterpret_cast<OrtModelCompilationOptions*>(model_compile_options.release());

  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::CreateModelCompilationOptionsFromSessionOptions, _In_ const OrtEnv* env,
                    _In_ OrtSessionOptions* session_options, _Outptr_ OrtModelCompilationOptions** out) {
  API_IMPL_BEGIN
  if (env == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "The env argument must be a non-null pointer");
  }

  if (session_options == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "The session_options argument must be a non-null pointer");
  }

  auto model_compile_options = std::make_unique<onnxruntime::ModelCompilationOptions>();
  model_compile_options->session_options_.reset(nullptr);
  model_compile_options->session_options_override_ = session_options;

  session_options->value.has_explicit_ep_context_gen_options = true;
  session_options->value.ep_context_gen_options = session_options->value.GetEpContextGenerationOptions();
  session_options->value.ep_context_gen_options.enable = true;
  session_options->value.ep_context_gen_options.always_generate = true;
  ORT_C_API_RETURN_IF_ERROR(session_options->value.config_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1"));
  *out = reinterpret_cast<OrtModelCompilationOptions*>(model_compile_options.release());
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetInputModelPath,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    const ORTCHAR_T* input_model_path) {
  API_IMPL_BEGIN
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);

  // Clear settings related to the input model. We want to allow the user to, for example, say the input model
  // comes from a buffer and then change their mind and say it comes from file.
  model_compile_options->ResetInputModelSettings();

  std::string model_path = PathToUTF8String(input_model_path);

  if (model_path.empty()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid input model: path string is empty");
  }

  model_compile_options->input_model_path = std::move(model_path);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetInputModelFromBuffer,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    const void* input_model_data, size_t input_model_data_size) {
  API_IMPL_BEGIN
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);

  // Clear settings related to the input model. We want to allow the user to, for example, say the input model
  // comes from a file and then change their mind and say it comes from memory.
  model_compile_options->ResetInputModelSettings();

  if (input_model_data == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid input model: data pointer is null");
  }

  if (input_model_data_size == 0) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid input model: data size is 0");
  }

  model_compile_options->input_model_data = input_model_data;
  model_compile_options->input_model_data_size = input_model_data_size;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetOutputModelPath,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    const ORTCHAR_T* output_model_path) {
  API_IMPL_BEGIN
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);
  ORT_C_API_RETURN_IF_ERROR(model_compile_options->ResetOutputModelSettings());

  std::string model_path = PathToUTF8String(output_model_path);
  if (model_path.empty()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid output model path: path is empty");
  }

  OrtSessionOptions* session_options = model_compile_options->GetSessionOptions();
  session_options->value.ep_context_gen_options.output_model_file_path = std::move(model_path);
  ORT_C_API_RETURN_IF_ERROR(session_options->value.config_options.AddConfigEntry(
      kOrtSessionOptionEpContextFilePath,
      session_options->value.ep_context_gen_options.output_model_file_path.c_str()));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetOutputModelExternalInitializersFile,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    const ORTCHAR_T* external_initializers_file_path,
                    size_t external_initializer_size_threshold) {
  API_IMPL_BEGIN
  std::string initializers_file_path = PathToUTF8String(external_initializers_file_path);
  if (initializers_file_path.empty()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid external initializer file: path is empty");
  }

  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);

  OrtSessionOptions* session_options = model_compile_options->GetSessionOptions();
  session_options->value.ep_context_gen_options.output_external_initializers_file_path = std::move(initializers_file_path);
  session_options->value.ep_context_gen_options.output_external_initializer_size_threshold = external_initializer_size_threshold;
  ORT_C_API_RETURN_IF_ERROR(session_options->value.config_options.AddConfigEntry(
      kOrtSessionOptionsEpContextModelExternalInitializersFileName,
      session_options->value.ep_context_gen_options.output_external_initializers_file_path.c_str()));
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetOutputModelBuffer,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    _Inout_ OrtAllocator* allocator, void** output_model_data_ptr, size_t* output_model_data_size_ptr) {
  API_IMPL_BEGIN
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);
  ORT_C_API_RETURN_IF_ERROR(model_compile_options->ResetOutputModelSettings());

  if (output_model_data_ptr == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid output model buffer: data pointer is null");
  }

  if (output_model_data_size_ptr == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid output model buffer: size pointer is null");
  }

  if (allocator == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid allocator for output model buffer: allocator pointer is null");
  }

  OrtSessionOptions* session_options = model_compile_options->GetSessionOptions();
  session_options->value.ep_context_gen_options.output_model_buffer_ptr = output_model_data_ptr;
  session_options->value.ep_context_gen_options.output_model_buffer_size_ptr = output_model_data_size_ptr;
  session_options->value.ep_context_gen_options.output_model_buffer_allocator = allocator;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetEpContextEmbedMode,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    bool embed_ep_context_in_model) {
  API_IMPL_BEGIN
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);
  OrtSessionOptions* session_options = model_compile_options->GetSessionOptions();
  ORT_C_API_RETURN_IF_ERROR(session_options->value.config_options.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode,
                                                                                 embed_ep_context_in_model ? "1" : "0"));
  session_options->value.ep_context_gen_options.embed_ep_context_in_model = embed_ep_context_in_model;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::CompileModel, _In_ const OrtEnv* env,
                    _In_ const OrtModelCompilationOptions* ort_model_compile_options) {
  API_IMPL_BEGIN
  auto model_compile_options = reinterpret_cast<const onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);
  ORT_C_API_RETURN_IF_ERROR(model_compile_options->Check());

  std::unique_ptr<onnxruntime::InferenceSession> session;
  OrtStatus* status = nullptr;

  ORT_TRY {
    if (!model_compile_options->input_model_path.empty()) {
      PathString input_model_path = ToPathString(model_compile_options->input_model_path);
      ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(model_compile_options->GetSessionOptions(), env,
                                                        input_model_path.c_str(),
                                                        nullptr, 0, session));
    } else {
      ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(model_compile_options->GetSessionOptions(), env, nullptr,
                                                        model_compile_options->input_model_data,
                                                        model_compile_options->input_model_data_size, session));
    }
    ORT_API_RETURN_IF_ERROR(InitializeSession(model_compile_options->GetSessionOptions(), *session));
  }
  ORT_CATCH(const std::exception& e) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = OrtApis::CreateStatus(ORT_FAIL, e.what());
    });
  }

  return status;
  API_IMPL_END
}

static constexpr OrtCompileApi ort_compile_api = {
    // NOTE: The C# bindings depend on the Api order within this struct so all additions must be at the end,
    // and no functions can be removed (the implementation needs to change to return an error).

    &OrtCompileAPI::ReleaseModelCompilationOptions,
    &OrtCompileAPI::CreateModelCompilationOptions,
    &OrtCompileAPI::CreateModelCompilationOptionsFromSessionOptions,
    &OrtCompileAPI::ModelCompilationOptions_SetInputModelPath,
    &OrtCompileAPI::ModelCompilationOptions_SetInputModelFromBuffer,
    &OrtCompileAPI::ModelCompilationOptions_SetOutputModelPath,
    &OrtCompileAPI::ModelCompilationOptions_SetOutputModelExternalInitializersFile,
    &OrtCompileAPI::ModelCompilationOptions_SetOutputModelBuffer,
    &OrtCompileAPI::ModelCompilationOptions_SetEpContextEmbedMode,
    &OrtCompileAPI::CompileModel,
};

// checks that we don't violate the rule that the functions must remain in the slots they were originally assigned
static_assert(offsetof(OrtCompileApi, CompileModel) / sizeof(void*) == 9,
              "Size of version 22 Api cannot change");  // initial version in ORT 1.22

ORT_API(const OrtCompileApi*, OrtCompileAPI::GetCompileApi) {
  return &ort_compile_api;
}

#endif  // !defined(ORT_MINIMAL_BUILD)
