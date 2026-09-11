// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_experimental_cxx_api.h"

#include "gmock/gmock.h"
#include "gsl/gsl"
#include "gtest/gtest.h"
#include "test/util/include/api_asserts.h"

namespace {

void ExpectFailureOrtStatus(OrtStatus* status_ptr, OrtErrorCode expected_code, const char* expected_message) {
  Ort::Status status{status_ptr};
  ASSERT_NE(status_ptr, nullptr) << "Expected a failure status, but the API returned nullptr (OK).";
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), expected_code);
  EXPECT_THAT(status.GetErrorMessage(), ::testing::HasSubstr(expected_message));
}

struct EpContextReadCallbackState {
  bool called = false;
  std::string file_name;
  std::vector<char> payload;
};

OrtStatus* ORT_API_CALL EpContextReadCallback(void* state, const char* file_name, OrtAllocator* allocator,
                                              void** buffer, size_t* data_size) {
  auto* read_state = static_cast<EpContextReadCallbackState*>(state);
  read_state->called = true;
  read_state->file_name = file_name;

  *buffer = nullptr;
  *data_size = read_state->payload.size();

  if (read_state->payload.empty()) {
    return nullptr;
  }

  OrtStatus* status = Ort::GetApi().AllocatorAlloc(allocator, read_state->payload.size(), buffer);
  if (status != nullptr) {
    return status;
  }

  std::memcpy(*buffer, read_state->payload.data(), read_state->payload.size());
  return nullptr;
}

struct EpContextWriteCallbackState {
  bool called = false;
  std::string file_name;
  std::vector<char> payload;
};

OrtStatus* ORT_API_CALL EpContextWriteCallback(void* state, const char* file_name, const void* buffer,
                                               size_t buffer_size) {
  auto* write_state = static_cast<EpContextWriteCallbackState*>(state);
  write_state->called = true;
  write_state->file_name = file_name;
  write_state->payload.clear();
  if (buffer_size != 0) {
    if (buffer == nullptr) {
      return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                        "EpContextWriteCallback received a null buffer for non-empty data");
    }

    const char* buffer_bytes = static_cast<const char*>(buffer);
    write_state->payload.assign(buffer_bytes, buffer_bytes + buffer_size);
  }

  return nullptr;
}

}  // namespace

TEST(EpContextDataApiTest, ReadFuncIsReturnedByEpApi) {
  const auto& ort_api = Ort::GetApi();
  Ort::SessionOptions session_options;

  auto* set_read_func =
      Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_FnOrThrow(&ort_api);

  EpContextReadCallbackState callback_state{
      false,
      {},
      {'e', 'p', 'c', 't', 'x'},
  };
  ASSERT_ORTSTATUS_OK(set_read_func(session_options, EpContextReadCallback, &callback_state));

  Ort::Experimental::EpContextConfig ep_context_config{session_options};
  OrtReadNamedBufferFunc read_func = nullptr;
  void* callback_state_out = nullptr;
  ep_context_config.GetReadFunc(read_func, callback_state_out);
  ASSERT_EQ(read_func, EpContextReadCallback);
  ASSERT_EQ(callback_state_out, &callback_state);

  Ort::AllocatorWithDefaultOptions allocator;
  void* buffer = nullptr;
  size_t buffer_size = 0;
  ASSERT_ORTSTATUS_OK(read_func(callback_state_out, "context.bin", allocator, &buffer, &buffer_size));
  auto release_buffer = gsl::finally([&]() {
    if (buffer != nullptr) {
      allocator.Free(buffer);
    }
  });

  ASSERT_TRUE(callback_state.called);
  EXPECT_EQ(callback_state.file_name, "context.bin");
  ASSERT_EQ(buffer_size, callback_state.payload.size());
  EXPECT_TRUE(std::equal(callback_state.payload.begin(), callback_state.payload.end(),
                         static_cast<const char*>(buffer)));
}

TEST(EpContextDataApiTest, ApiRejectsInvalidArguments) {
  const auto& ort_api = Ort::GetApi();

  auto* get_config = Ort::Experimental::Get_OrtEpApi_SessionOptions_GetEpContextConfig_SinceV28_FnOrThrow(&ort_api);
  auto* release_config_func =
      Ort::Experimental::Get_OrtEpApi_ReleaseEpContextConfig_SinceV28_FnOrThrow(&ort_api);
  auto* get_read_func =
      Ort::Experimental::Get_OrtEpApi_EpContextConfig_GetEpContextDataReadFunc_SinceV28_FnOrThrow(&ort_api);
  auto* get_write_func =
      Ort::Experimental::Get_OrtEpApi_EpContextConfig_GetEpContextDataWriteFunc_SinceV28_FnOrThrow(&ort_api);
  auto* set_read_func =
      Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_FnOrThrow(&ort_api);

  Ort::SessionOptions session_options;
  OrtEpContextConfig* ep_context_config = nullptr;
  ExpectFailureOrtStatus(get_config(nullptr, &ep_context_config), ORT_INVALID_ARGUMENT, "OrtSessionOptions is NULL");
  ExpectFailureOrtStatus(get_config(session_options, nullptr), ORT_INVALID_ARGUMENT,
                         "Output OrtEpContextConfig is NULL");

  ExpectFailureOrtStatus(set_read_func(nullptr, EpContextReadCallback, nullptr), ORT_INVALID_ARGUMENT,
                         "'options' parameter must not be NULL");

  ASSERT_ORTSTATUS_OK(get_config(session_options, &ep_context_config));
  auto release_config = gsl::finally([&]() { release_config_func(ep_context_config); });

  OrtReadNamedBufferFunc read_func = nullptr;
  OrtWriteNamedBufferFunc write_func = nullptr;
  void* state = nullptr;
  ExpectFailureOrtStatus(get_read_func(nullptr, &read_func, &state), ORT_INVALID_ARGUMENT,
                         "OrtEpContextConfig is NULL");
  ExpectFailureOrtStatus(get_read_func(ep_context_config, nullptr, &state), ORT_INVALID_ARGUMENT,
                         "Output read_func is NULL");
  ExpectFailureOrtStatus(get_read_func(ep_context_config, &read_func, nullptr), ORT_INVALID_ARGUMENT,
                         "Output state is NULL");
  ExpectFailureOrtStatus(get_write_func(nullptr, &write_func, &state), ORT_INVALID_ARGUMENT,
                         "OrtEpContextConfig is NULL");
  ExpectFailureOrtStatus(get_write_func(ep_context_config, nullptr, &state), ORT_INVALID_ARGUMENT,
                         "Output write_func is NULL");
  ExpectFailureOrtStatus(get_write_func(ep_context_config, &write_func, nullptr), ORT_INVALID_ARGUMENT,
                         "Output state is NULL");

#if !defined(ORT_MINIMAL_BUILD)
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "EpContextDataApiRejectsInvalidArguments"};
  Ort::ModelCompilationOptions compilation_options{env, session_options};
  auto* set_write_func =
      Ort::Experimental::Get_OrtCompileApi_ModelCompilationOptions_SetEpContextDataWriteFunc_SinceV28_FnOrThrow(
          &ort_api);
  ExpectFailureOrtStatus(set_write_func(nullptr, EpContextWriteCallback, nullptr), ORT_INVALID_ARGUMENT,
                         "OrtModelCompilationOptions is NULL");
  // A null write_func is allowed: it clears any previously set callback (covered by WriteFuncCanBeCleared), so it is
  // not rejected here.
#endif  // !defined(ORT_MINIMAL_BUILD)
}

TEST(EpContextDataApiTest, AccessorsReturnNullWhenCallbacksUnset) {
  Ort::SessionOptions session_options;
  Ort::Experimental::EpContextConfig ep_context_config{session_options};

  OrtReadNamedBufferFunc read_func = EpContextReadCallback;
  OrtWriteNamedBufferFunc write_func = EpContextWriteCallback;
  void* state = reinterpret_cast<void*>(0x1);

  ep_context_config.GetReadFunc(read_func, state);
  EXPECT_EQ(read_func, nullptr);
  EXPECT_EQ(state, nullptr);

  state = reinterpret_cast<void*>(0x1);
  ep_context_config.GetWriteFunc(write_func, state);
  EXPECT_EQ(write_func, nullptr);
  EXPECT_EQ(state, nullptr);
}

TEST(EpContextDataApiTest, ConfigReturnsConfiguredCallbacks) {
  const auto& ort_api = Ort::GetApi();
  Ort::SessionOptions session_options;

  auto* set_read_func =
      Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_FnOrThrow(&ort_api);

  EpContextReadCallbackState callback_state{};
  ASSERT_ORTSTATUS_OK(set_read_func(session_options, EpContextReadCallback, &callback_state));

  Ort::Experimental::EpContextConfig ep_context_config{session_options};

  OrtReadNamedBufferFunc read_func = nullptr;
  void* read_state = nullptr;
  ep_context_config.GetReadFunc(read_func, read_state);
  EXPECT_EQ(read_func, EpContextReadCallback);
  EXPECT_EQ(read_state, &callback_state);

  OrtWriteNamedBufferFunc write_func = nullptr;
  void* write_state = nullptr;
  ep_context_config.GetWriteFunc(write_func, write_state);
  EXPECT_EQ(write_func, nullptr);
  EXPECT_EQ(write_state, nullptr);
}

TEST(EpContextDataApiTest, ReadFuncCanBeCleared) {
  const auto& ort_api = Ort::GetApi();
  Ort::SessionOptions session_options;

  auto* set_read_func =
      Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_FnOrThrow(&ort_api);

  EpContextReadCallbackState callback_state{};
  ASSERT_ORTSTATUS_OK(set_read_func(session_options, EpContextReadCallback, &callback_state));

  ASSERT_ORTSTATUS_OK(set_read_func(session_options, nullptr, &callback_state));

  Ort::Experimental::EpContextConfig ep_context_config{session_options};
  OrtReadNamedBufferFunc read_func = EpContextReadCallback;
  void* read_state = reinterpret_cast<void*>(0x1);
  ep_context_config.GetReadFunc(read_func, read_state);
  EXPECT_EQ(read_func, nullptr);
  EXPECT_EQ(read_state, nullptr);
}

#if !defined(ORT_MINIMAL_BUILD)
TEST(EpContextDataApiTest, WriteFuncCanBeSetOnModelCompilationOptions) {
  const auto& ort_api = Ort::GetApi();
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "EpContextDataWriteFuncCanBeSetOnModelCompilationOptions"};
  Ort::SessionOptions session_options;
  Ort::ModelCompilationOptions compilation_options{env, session_options};

  auto* set_write_func =
      Ort::Experimental::Get_OrtCompileApi_ModelCompilationOptions_SetEpContextDataWriteFunc_SinceV28_FnOrThrow(
          &ort_api);

  EpContextWriteCallbackState callback_state{};
  ASSERT_ORTSTATUS_OK(set_write_func(compilation_options, EpContextWriteCallback, &callback_state));

  const std::vector<char> payload{'b', 'i', 'n', 'a', 'r', 'y'};
  ASSERT_ORTSTATUS_OK(EpContextWriteCallback(&callback_state, "engine.bin", payload.data(), payload.size()));

  ASSERT_TRUE(callback_state.called);
  EXPECT_EQ(callback_state.file_name, "engine.bin");
  EXPECT_EQ(callback_state.payload, payload);
}

TEST(EpContextDataApiTest, WriteFuncCanBeCleared) {
  const auto& ort_api = Ort::GetApi();
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "EpContextDataWriteFuncCanBeCleared"};
  Ort::SessionOptions session_options;
  Ort::ModelCompilationOptions compilation_options{env, session_options};

  auto* set_write_func =
      Ort::Experimental::Get_OrtCompileApi_ModelCompilationOptions_SetEpContextDataWriteFunc_SinceV28_FnOrThrow(
          &ort_api);

  EpContextWriteCallbackState callback_state{};
  ASSERT_ORTSTATUS_OK(set_write_func(compilation_options, EpContextWriteCallback, &callback_state));

  // A null write_func clears the previously set callback (symmetric with the read setter) and must be accepted
  // rather than rejected with ORT_INVALID_ARGUMENT.
  ASSERT_ORTSTATUS_OK(set_write_func(compilation_options, nullptr, &callback_state));
}

TEST(EpContextDataApiTest, WriteFuncCanBeUsedWithEpContextBinaryInformation) {
  const auto& ort_api = Ort::GetApi();
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "EpContextDataWriteFuncCanBeUsedWithEpContextBinaryInformation"};
  Ort::SessionOptions session_options;
  Ort::ModelCompilationOptions compilation_options{env, session_options};

  auto* set_write_func =
      Ort::Experimental::Get_OrtCompileApi_ModelCompilationOptions_SetEpContextDataWriteFunc_SinceV28_FnOrThrow(
          &ort_api);

  // The EPContext write callback and the EPContext binary information may be configured together; neither call
  // rejects the other.
  ASSERT_NO_THROW(compilation_options.SetEpContextBinaryInformation(ORT_TSTR("ep_context_dir/"),
                                                                    ORT_TSTR("compiled_model.onnx")));

  EpContextWriteCallbackState callback_state{};
  ASSERT_ORTSTATUS_OK(set_write_func(compilation_options, EpContextWriteCallback, &callback_state));

  const std::vector<char> payload{'c', 't', 'x'};
  ASSERT_ORTSTATUS_OK(EpContextWriteCallback(&callback_state, "logical_context.bin", payload.data(), payload.size()));

  ASSERT_TRUE(callback_state.called);
  EXPECT_EQ(callback_state.file_name, "logical_context.bin");
  EXPECT_EQ(callback_state.payload, payload);
}
#endif  // !defined(ORT_MINIMAL_BUILD)

TEST(EpContextDataApiTest, ReturnedReadFuncAllowsEmptyPayloads) {
  const auto& ort_api = Ort::GetApi();
  Ort::SessionOptions session_options;

  auto* set_read_func =
      Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_FnOrThrow(&ort_api);

  EpContextReadCallbackState callback_state{};
  ASSERT_ORTSTATUS_OK(set_read_func(session_options, EpContextReadCallback, &callback_state));

  Ort::Experimental::EpContextConfig ep_context_config{session_options};
  OrtReadNamedBufferFunc read_func = nullptr;
  void* read_state = nullptr;
  ep_context_config.GetReadFunc(read_func, read_state);
  ASSERT_EQ(read_func, EpContextReadCallback);
  ASSERT_EQ(read_state, &callback_state);

  Ort::AllocatorWithDefaultOptions allocator;
  void* buffer = reinterpret_cast<void*>(0x1);
  size_t buffer_size = 1;
  ASSERT_ORTSTATUS_OK(read_func(read_state, "empty.bin", allocator, &buffer, &buffer_size));

  EXPECT_TRUE(callback_state.called);
  EXPECT_EQ(callback_state.file_name, "empty.bin");
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(buffer_size, 0U);
}
