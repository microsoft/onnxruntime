// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <optional>
#include <stdexcept>

#pragma push_macro("ORT_API_MANUAL_INIT")
#undef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#pragma pop_macro("ORT_API_MANUAL_INIT")

namespace onnxruntime {
namespace ep {

struct ApiPtrs {
  ApiPtrs(const OrtApi& ort_, const OrtEpApi& ep_, const OrtModelEditorApi& model_editor_)
      : ort(ort_), ep(ep_), model_editor(model_editor_) {}
  const OrtApi& ort;
  const OrtEpApi& ep;
  const OrtModelEditorApi& model_editor;
};

namespace detail {
inline std::optional<ApiPtrs> g_api_ptrs;
}

/// <summary>
/// Get the global instance of ApiPtrs.
/// </summary>
inline const ApiPtrs& Api() {
  if (!detail::g_api_ptrs.has_value()) {
    throw std::logic_error("onnxruntime::ep::Api() called before ApiInit().");
  }
  return *detail::g_api_ptrs;
}

/// <summary>
/// Initialize the EP API pointers and global OrtEnv if not already done.
/// Thread-safe via std::call_once.
/// </summary>
inline void ApiInit(const OrtApiBase* ort_api_base) {
  static std::once_flag init_flag;
  std::call_once(init_flag, [&]() {
    // Manual init for the C++ API
    const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
    const OrtEpApi* ep_api = ort_api->GetEpApi();
    const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();
    Ort::InitApi(ort_api);

    // Initialize the global API instance
    detail::g_api_ptrs.emplace(*ort_api, *ep_api, *model_editor_api);
  });
}

}  // namespace ep
}  // namespace onnxruntime
