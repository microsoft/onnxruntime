// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
namespace test {

// Shared EPContext read/write callback test doubles, used by both the EpContextDataUtils unit tests
// (ep_context_data_utils_test.cc) and the PluginEp end-to-end EPContext tests (test_execution.cc).
struct EpContextDataCallbackState {
  bool write_called = false;
  bool read_called = false;
  std::string write_file_name;
  std::string read_file_name;
  std::vector<char> payload;
};

inline OrtStatus* ORT_API_CALL StoreEpContextDataCallback(void* state, const char* file_name, const void* buffer,
                                                          size_t buffer_size) {
  auto* callback_state = static_cast<EpContextDataCallbackState*>(state);
  callback_state->write_called = true;
  callback_state->write_file_name = file_name;
  callback_state->payload.clear();
  if (buffer_size != 0) {
    if (buffer == nullptr) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "StoreEpContextDataCallback received a null buffer for non-empty data");
    }
    callback_state->payload.assign(static_cast<const char*>(buffer), static_cast<const char*>(buffer) + buffer_size);
  }
  return nullptr;
}

inline OrtStatus* ORT_API_CALL LoadEpContextDataCallback(void* state, const char* file_name, OrtAllocator* allocator,
                                                         void** buffer, size_t* data_size) {
  auto* callback_state = static_cast<EpContextDataCallbackState*>(state);
  callback_state->read_called = true;
  callback_state->read_file_name = file_name;

  *buffer = nullptr;
  *data_size = callback_state->payload.size();
  if (callback_state->payload.empty()) {
    return nullptr;
  }

  OrtStatus* status = Ort::GetApi().AllocatorAlloc(allocator, callback_state->payload.size(), buffer);
  if (status != nullptr) {
    return status;
  }

  std::copy(callback_state->payload.begin(), callback_state->payload.end(), static_cast<char*>(*buffer));
  return nullptr;
}

}  // namespace test
}  // namespace onnxruntime
