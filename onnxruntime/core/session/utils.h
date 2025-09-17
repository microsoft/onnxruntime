// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
#include <memory>
#include <string_view>
#include <vector>
#include "core/common/common.h"
#include "core/session/onnxruntime_c_api.h"

onnxruntime::common::Status CopyStringToOutputArg(std::string_view str, const char* err_msg, char* out, size_t* size);

struct OrtSessionOptions;
struct OrtStatus;
struct OrtPrepackedWeightsContainer;
namespace onnxruntime {
class InferenceSession;
class ModelCompilationOptions;
}  // namespace onnxruntime

#if !defined(ORT_MINIMAL_BUILD)
namespace onnxruntime {
class Environment;
class EpLibrary;
class EpFactoryInternal;
struct IExecutionProviderFactory;
struct SessionOptions;
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)

OrtStatus* CreateSessionAndLoadModel(_In_ const OrtSessionOptions* options,
                                     _In_ const OrtEnv* env,
                                     _In_opt_z_ const ORTCHAR_T* model_path,
                                     _In_opt_ const void* model_data,
                                     size_t model_data_length,
                                     std::unique_ptr<onnxruntime::InferenceSession>& sess);

OrtStatus* InitializeSession(_In_ const OrtSessionOptions* options,
                             _In_ onnxruntime::InferenceSession& sess,
                             _Inout_opt_ OrtPrepackedWeightsContainer* prepacked_weights_container = nullptr);

#if !defined(ORT_MINIMAL_BUILD)
namespace onnxruntime {

/// <summary>
/// Compiles an ONNX model into a model with EPContext nodes. Each EPContext node represents a subgraph compiled for
/// a specific execution provider.
/// </summary>
/// <param name="env">A reference to the Environment instance.</param>
/// <param name="model_compile_options">An object specifying the compilation options.</param>
/// <returns>A Status indicating an error or success.</returns>
Status CompileModel(const Environment& env, const ModelCompilationOptions& model_compile_options);

// load a library that is added using RegisterExecutionProviderLibrary.
// infer whether it's a provider bridge library or plugin library
Status LoadPluginOrProviderBridge(const std::string& registration_name,
                                  const ORTCHAR_T* library_path,
                                  std::unique_ptr<EpLibrary>& ep_library,
                                  std::vector<EpFactoryInternal*>& internal_factories);

// Creates an IExecutionProviderFactory instance for a list of OrtEpDevices that all refer to the same EP.
Status CreateIExecutionProviderFactoryForEpDevices(const Environment& env,
                                                   gsl::span<const OrtEpDevice* const> ep_devices,
                                                   /*output*/ std::unique_ptr<IExecutionProviderFactory>& out);

// Adds provider options to the OrtSessionOptions configuration.
Status AddEpOptionsToSessionOptions(gsl::span<const OrtEpDevice* const> ep_devices,
                                    gsl::span<const char* const> ep_options_keys,
                                    gsl::span<const char* const> ep_options_vals,
                                    SessionOptions& session_options);

}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
