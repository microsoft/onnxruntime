// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include "core/session/compile_api.h"

#include <memory>
#include <string>
#include <utility>

#include "core/common/common.h"
#include "core/framework/error_code_helper.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/inference_session.h"
#include "core/session/model_compilation_options.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/ort_apis.h"
#include "core/session/ort_env.h"
#include "core/session/utils.h"

using namespace onnxruntime;

ORT_API(void, OrtCompileAPI::ReleaseModelCompilationOptions,
        _Frees_ptr_opt_ OrtModelCompilationOptions* ort_model_compile_options) {
  delete reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);
}

ORT_API_STATUS_IMPL(OrtCompileAPI::CreateModelCompilationOptionsFromSessionOptions, _In_ const OrtEnv* env,
                    _In_ const OrtSessionOptions* session_options, _Outptr_ OrtModelCompilationOptions** out) {
  API_IMPL_BEGIN
  if (env == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "The env argument must be a non-null pointer");
  }

  if (session_options == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "The session_options argument must be a non-null pointer");
  }

  auto model_compile_options = std::make_unique<onnxruntime::ModelCompilationOptions>();
  model_compile_options->session_options = std::make_unique<OrtSessionOptions>(*session_options);

  model_compile_options->session_options->value.has_explicit_ep_context_gen_options = true;
  model_compile_options->session_options->value.ep_context_gen_options = session_options->value.GetEpContextGenerationOptions();
  model_compile_options->session_options->value.ep_context_gen_options.enable = true;
  model_compile_options->session_options->value.ep_context_gen_options.overwrite_existing_output_file = true;
  model_compile_options->session_options->value.ep_context_gen_options.error_if_no_compiled_nodes = true;
  ORT_API_RETURN_IF_STATUS_NOT_OK(model_compile_options->session_options->value.config_options.AddConfigEntry(
      kOrtSessionOptionEpContextEnable, "1"));
  *out = reinterpret_cast<OrtModelCompilationOptions*>(model_compile_options.release());
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetInputModelPath,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    const ORTCHAR_T* input_model_path) {
  API_IMPL_BEGIN
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);
  std::string model_path = PathToUTF8String(input_model_path);

  if (model_path.empty()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid input model: path string is empty");
  }

  // Clear settings related to the input model. We want to allow the user to, for example, say the input model
  // comes from a buffer and then change their mind and say it comes from file.
  model_compile_options->ResetInputModelSettings();
  model_compile_options->input_model_path = std::move(model_path);
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetInputModelFromBuffer,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    const void* input_model_data, size_t input_model_data_size) {
  API_IMPL_BEGIN
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);

  if (input_model_data == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid input model: data pointer is null");
  }

  if (input_model_data_size == 0) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid input model: data size is 0");
  }

  // Clear settings related to the input model. We want to allow the user to, for example, say the input model
  // comes from a file and then change their mind and say it comes from memory.
  model_compile_options->ResetInputModelSettings();
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

  std::string model_path = PathToUTF8String(output_model_path);
  if (model_path.empty()) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid output model path: path is empty");
  }

  ORT_API_RETURN_IF_STATUS_NOT_OK(model_compile_options->ResetOutputModelSettings());

  OrtSessionOptions* session_options = model_compile_options->session_options.get();
  ConfigOptions& config_options = session_options->value.config_options;
  EpContextModelGenerationOptions& ep_context_gen_options = session_options->value.ep_context_gen_options;

  ep_context_gen_options.output_model_file_path = std::move(model_path);

  if (ep_context_gen_options.output_model_file_path.size() <= ConfigOptions::kMaxValueLength) {
    // A few things to note:
    //   - ORT core now uses session_options.ep_context_gen_options to read EPContext model configurations.
    //     It previously used session_options.config_options.
    //   - EPs still currently use session_options.config_options to read a subset (enabled, embed mode, output path) of
    //     EPContext model configurations.
    //     TODO(adrianlizarraga): Update EPs to use ep_context_gen_options in backward-compatible manner.
    //   - The output model file path is optional (generated from input path if absent).
    //   - EPs use the output model path to generate a path to the context binary data file IFF not embedded
    //     into EPContext nodes. If output model path is empty, EPs just create a path from input model path.
    //   - session_options.config_options limits the string length of values, which artificially limits the length
    //     of paths.
    //   - So, only add this output model file path to session_options.config_options if it is not too long. The only
    //     potential downside is that the context binary data file is using a different name, but the model will still
    //     be valid.
    Status status = config_options.AddConfigEntry(kOrtSessionOptionEpContextFilePath,
                                                  ep_context_gen_options.output_model_file_path.c_str());
    ORT_ENFORCE(status.IsOK());  // Should not fail because both key/value strings are below the min string lengths
                                 // required by ConfigOptions::AddConfigEntry().
  }

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

  OrtSessionOptions* session_options = model_compile_options->session_options.get();
  session_options->value.ep_context_gen_options.output_external_initializers_file_path = std::move(initializers_file_path);
  session_options->value.ep_context_gen_options.output_external_initializer_size_threshold = external_initializer_size_threshold;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::ModelCompilationOptions_SetOutputModelBuffer,
                    _In_ OrtModelCompilationOptions* ort_model_compile_options,
                    _Inout_ OrtAllocator* allocator, void** output_model_data_ptr, size_t* output_model_data_size_ptr) {
  API_IMPL_BEGIN
  auto model_compile_options = reinterpret_cast<onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);

  if (output_model_data_ptr == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid output model buffer: data pointer is null");
  }

  if (output_model_data_size_ptr == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid output model buffer: size pointer is null");
  }

  if (allocator == nullptr) {
    return OrtApis::CreateStatus(ORT_INVALID_ARGUMENT, "Invalid allocator for output model buffer: allocator pointer is null");
  }

  ORT_API_RETURN_IF_STATUS_NOT_OK(model_compile_options->ResetOutputModelSettings());

  OrtSessionOptions* session_options = model_compile_options->session_options.get();
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
  OrtSessionOptions* session_options = model_compile_options->session_options.get();
  ORT_API_RETURN_IF_STATUS_NOT_OK(session_options->value.config_options.AddConfigEntry(
      kOrtSessionOptionEpContextEmbedMode, embed_ep_context_in_model ? "1" : "0"));
  session_options->value.ep_context_gen_options.embed_ep_context_in_model = embed_ep_context_in_model;
  return nullptr;
  API_IMPL_END
}

ORT_API_STATUS_IMPL(OrtCompileAPI::CompileModel, _In_ const OrtEnv* env,
                    _In_ const OrtModelCompilationOptions* ort_model_compile_options) {
  API_IMPL_BEGIN
  auto model_compile_options = reinterpret_cast<const onnxruntime::ModelCompilationOptions*>(ort_model_compile_options);
  ORT_API_RETURN_IF_STATUS_NOT_OK(model_compile_options->Check());

  std::unique_ptr<onnxruntime::InferenceSession> session;

  if (!model_compile_options->input_model_path.empty()) {
    PathString input_model_path = ToPathString(model_compile_options->input_model_path);
    ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(model_compile_options->session_options.get(), env,
                                                      input_model_path.c_str(),
                                                      nullptr, 0, session));
  } else {
    ORT_API_RETURN_IF_ERROR(CreateSessionAndLoadModel(model_compile_options->session_options.get(), env, nullptr,
                                                      model_compile_options->input_model_data,
                                                      model_compile_options->input_model_data_size, session));
  }

  ORT_API_RETURN_IF_ERROR(InitializeSession(model_compile_options->session_options.get(), *session));
  return nullptr;
  API_IMPL_END
}

static constexpr OrtCompileApi ort_compile_api = {
    // NOTE: The C# bindings depend on the Api order within this struct so all additions must be at the end,
    // and no functions can be removed (the implementation needs to change to return an error).

    &OrtCompileAPI::ReleaseModelCompilationOptions,
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
static_assert(offsetof(OrtCompileApi, CompileModel) / sizeof(void*) == 8,
              "Size of version 22 Api cannot change");  // initial version in ORT 1.22

ORT_API(const OrtCompileApi*, OrtCompileAPI::GetCompileApi) {
  return &ort_compile_api;
}

#endif  // !defined(ORT_MINIMAL_BUILD)
