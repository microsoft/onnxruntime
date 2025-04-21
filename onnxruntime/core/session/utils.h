// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string_view>
#include "core/common/common.h"
#include "core/session/onnxruntime_c_api.h"

onnxruntime::common::Status CopyStringToOutputArg(std::string_view str, const char* err_msg, char* out, size_t* size);

struct OrtSessionOptions;
struct OrtStatus;
struct OrtPrepackedWeightsContainer;
namespace onnxruntime {
class InferenceSession;
class EpLibrary;
class EpFactoryInternal;
}  // namespace onnxruntime

OrtStatus* CreateSessionAndLoadModel(_In_ const OrtSessionOptions* options,
                                     _In_ const OrtEnv* env,
                                     _In_opt_z_ const ORTCHAR_T* model_path,
                                     _In_opt_ const void* model_data,
                                     size_t model_data_length,
                                     std::unique_ptr<onnxruntime::InferenceSession>& sess);

OrtStatus* InitializeSession(_In_ const OrtSessionOptions* options,
                             _In_ onnxruntime::InferenceSession& sess,
                             _Inout_opt_ OrtPrepackedWeightsContainer* prepacked_weights_container = nullptr);

namespace onnxruntime {

// load a library that is added using RegisterExecutionProviderLibrary.
// infer whether it's a provider bridge library or plugin library
Status LoadPluginOrProviderBridge(const std::string& registration_name,
                                  const ORTCHAR_T* library_path,
                                  std::unique_ptr<EpLibrary>& ep_library,
                                  std::vector<EpFactoryInternal*>& internal_factories);

}  // namespace onnxruntime
