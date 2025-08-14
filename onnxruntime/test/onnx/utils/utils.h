// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/session/onnxruntime_cxx_api.h"
#include <memory>

namespace onnxruntime {
namespace test {
namespace utils {

void RegisterExecutionProviderLibrary(Ort::Env& env,
                                      std::basic_string<ORTCHAR_T>& plugin_ep_names_and_libs,
                                      std::vector<std::string>& registered_plugin_eps);

void UnregisterExecutionProviderLibrary(Ort::Env& env, std::vector<std::string>& registered_plugin_eps);

void ListEpDevices(const Ort::Env& env);

}  // namespace utils
}  // namespace test
}  // namespace onnxruntime
