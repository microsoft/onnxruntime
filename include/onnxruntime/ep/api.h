// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>

#pragma push_macro("ORT_API_MANUAL_INIT")
#undef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#include "onnxruntime_cxx_api.h"
#pragma pop_macro("ORT_API_MANUAL_INIT")

namespace onnxruntime {
namespace ep {

struct ApiPtrs {
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
  return *detail::g_api_ptrs;
}

/// <summary>
/// Initialize the EP API pointers and global OrtEnv if not already done.
/// </summary>
inline void ApiInit(const OrtApiBase* ort_api_base) {
  // Manual init for the C++ API
  const OrtApi* ort_api = ort_api_base->GetApi(ORT_API_VERSION);
  const OrtEpApi* ep_api = ort_api->GetEpApi();
  const OrtModelEditorApi* model_editor_api = ort_api->GetModelEditorApi();
  Ort::InitApi(ort_api);

  // Initialize the global API instance
  if (!detail::g_api_ptrs) {
    detail::g_api_ptrs.emplace(*ort_api, *ep_api, *model_editor_api);
  }
}

}  // namespace ep
}  // namespace onnxruntime
