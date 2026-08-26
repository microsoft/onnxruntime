// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "core/session/onnxruntime_c_api.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_experimental_cxx_api.h"

#include "gmock/gmock.h"
#include "gsl/gsl"
#include "gtest/gtest.h"
#include "test/util/include/api_asserts.h"

namespace {

constexpr size_t kEpContextApiTestMaxDataSize = size_t{1} << 20;

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
  size_t max_data_size = std::numeric_limits<size_t>::max();
  bool allocation_attempted = false;
};

OrtStatus* ORT_API_CALL EpContextReadCallback(void* state, const char* file_name, OrtAllocator* allocator,
                                              void** buffer, size_t* data_size) {
  auto* read_state = static_cast<EpContextReadCallbackState*>(state);
  read_state->called = true;
  read_state->file_name = file_name;

  *buffer = nullptr;
  *data_size = read_state->payload.size();

  if (read_state->payload.size() > read_state->max_data_size) {
    return Ort::GetApi().CreateStatus(ORT_INVALID_ARGUMENT,
                                      "EPContext artifact exceeds the application callback size limit");
  }
  if (read_state->payload.empty()) {
    return nullptr;
  }

  read_state->allocation_attempted = true;
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

#if !defined(ORT_MINIMAL_BUILD)
TEST(EpContextDataApiTest, ReadFuncIsReturnedByEpApi) {
  Ort::SessionOptions session_options;

  EpContextReadCallbackState callback_state{
      false,
      {},
      {'e', 'p', 'c', 't', 'x'},
  };
  session_options.SetEpContextDataReadFunc(EpContextReadCallback, &callback_state,
                                           kEpContextApiTestMaxDataSize);

  Ort::EpContextConfig ep_context_config{session_options};
  OrtReadNamedBufferFunc read_func = nullptr;
  void* callback_state_out = nullptr;
  size_t max_data_size = 0;
  ep_context_config.GetReadFunc(read_func, callback_state_out, max_data_size);
  ASSERT_EQ(read_func, EpContextReadCallback);
  ASSERT_EQ(callback_state_out, &callback_state);
  ASSERT_EQ(max_data_size, kEpContextApiTestMaxDataSize);

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
  const auto& ep_api = Ort::GetEpApi();

  Ort::SessionOptions session_options;
  OrtEpContextConfig* ep_context_config = nullptr;
  ExpectFailureOrtStatus(ep_api.SessionOptionsGetEpContextConfig(nullptr, &ep_context_config), ORT_INVALID_ARGUMENT,
                         "OrtSessionOptions is NULL");
  ExpectFailureOrtStatus(ep_api.SessionOptionsGetEpContextConfig(session_options, nullptr), ORT_INVALID_ARGUMENT,
                         "Output OrtEpContextConfig is NULL");

  const OrtEpContextDataReadOptions valid_read_options{ORT_EP_CONTEXT_DATA_READ_OPTIONS_VERSION,
                                                       kEpContextApiTestMaxDataSize};
  ExpectFailureOrtStatus(ort_api.SessionOptionsSetEpContextDataReadFunc(
                             nullptr, EpContextReadCallback, nullptr, &valid_read_options),
                         ORT_INVALID_ARGUMENT,
                         "'options' parameter must not be NULL");
  ExpectFailureOrtStatus(ort_api.SessionOptionsSetEpContextDataReadFunc(
                             session_options, EpContextReadCallback, nullptr, nullptr),
                         ORT_INVALID_ARGUMENT, "read options must be provided");

  OrtEpContextDataReadOptions invalid_read_options{ORT_EP_CONTEXT_DATA_READ_OPTIONS_VERSION + 1,
                                                   kEpContextApiTestMaxDataSize};
  ExpectFailureOrtStatus(ort_api.SessionOptionsSetEpContextDataReadFunc(
                             session_options, EpContextReadCallback, nullptr, &invalid_read_options),
                         ORT_INVALID_ARGUMENT, "Unsupported EPContext data read options version");
  invalid_read_options = {ORT_EP_CONTEXT_DATA_READ_OPTIONS_VERSION, 0};
  ExpectFailureOrtStatus(ort_api.SessionOptionsSetEpContextDataReadFunc(
                             session_options, EpContextReadCallback, nullptr, &invalid_read_options),
                         ORT_INVALID_ARGUMENT, "max_data_size must be finite and greater than zero");
  invalid_read_options = {ORT_EP_CONTEXT_DATA_READ_OPTIONS_VERSION, std::numeric_limits<size_t>::max()};
  ExpectFailureOrtStatus(ort_api.SessionOptionsSetEpContextDataReadFunc(
                             session_options, EpContextReadCallback, nullptr, &invalid_read_options),
                         ORT_INVALID_ARGUMENT, "max_data_size must be finite and greater than zero");

  ASSERT_ORTSTATUS_OK(ep_api.SessionOptionsGetEpContextConfig(session_options, &ep_context_config));
  auto release_config = gsl::finally([&]() { ep_api.ReleaseEpContextConfig(ep_context_config); });

  OrtReadNamedBufferFunc read_func = nullptr;
  OrtWriteNamedBufferFunc write_func = nullptr;
  void* state = nullptr;
  size_t max_data_size = 0;
  ExpectFailureOrtStatus(ep_api.EpContextConfigGetEpContextDataReadFunc(nullptr, &read_func, &state,
                                                                        &max_data_size),
                         ORT_INVALID_ARGUMENT,
                         "OrtEpContextConfig is NULL");
  ExpectFailureOrtStatus(ep_api.EpContextConfigGetEpContextDataReadFunc(ep_context_config, nullptr, &state,
                                                                        &max_data_size),
                         ORT_INVALID_ARGUMENT,
                         "Output read_func is NULL");
  ExpectFailureOrtStatus(ep_api.EpContextConfigGetEpContextDataReadFunc(ep_context_config, &read_func, nullptr,
                                                                        &max_data_size),
                         ORT_INVALID_ARGUMENT,
                         "Output state is NULL");
  ExpectFailureOrtStatus(ep_api.EpContextConfigGetEpContextDataReadFunc(ep_context_config, &read_func, &state,
                                                                        nullptr),
                         ORT_INVALID_ARGUMENT,
                         "Output max_data_size is NULL");
  ExpectFailureOrtStatus(ep_api.EpContextConfigGetEpContextDataWriteFunc(nullptr, &write_func, &state),
                         ORT_INVALID_ARGUMENT,
                         "OrtEpContextConfig is NULL");
  ExpectFailureOrtStatus(ep_api.EpContextConfigGetEpContextDataWriteFunc(ep_context_config, nullptr, &state),
                         ORT_INVALID_ARGUMENT,
                         "Output write_func is NULL");
  ExpectFailureOrtStatus(ep_api.EpContextConfigGetEpContextDataWriteFunc(ep_context_config, &write_func, nullptr),
                         ORT_INVALID_ARGUMENT,
                         "Output state is NULL");

#if !defined(ORT_MINIMAL_BUILD)
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "EpContextDataApiRejectsInvalidArguments"};
  Ort::ModelCompilationOptions compilation_options{env, session_options};
  ExpectFailureOrtStatus(Ort::GetCompileApi().ModelCompilationOptions_SetEpContextDataWriteFunc(
                             nullptr, EpContextWriteCallback, nullptr),
                         ORT_INVALID_ARGUMENT,
                         "OrtModelCompilationOptions is NULL");
  // A null write_func is allowed: it clears any previously set callback (covered by WriteFuncCanBeCleared), so it is
  // not rejected here.
#endif  // !defined(ORT_MINIMAL_BUILD)
}

TEST(EpContextDataApiTest, AccessorsReturnNullWhenCallbacksUnset) {
  Ort::SessionOptions session_options;
  Ort::EpContextConfig ep_context_config{session_options};

  OrtReadNamedBufferFunc read_func = EpContextReadCallback;
  OrtWriteNamedBufferFunc write_func = EpContextWriteCallback;
  void* state = reinterpret_cast<void*>(0x1);
  size_t max_data_size = 0;

  ep_context_config.GetReadFunc(read_func, state, max_data_size);
  EXPECT_EQ(read_func, nullptr);
  EXPECT_EQ(state, nullptr);
  EXPECT_EQ(max_data_size, std::numeric_limits<size_t>::max());

  state = reinterpret_cast<void*>(0x1);
  ep_context_config.GetWriteFunc(write_func, state);
  EXPECT_EQ(write_func, nullptr);
  EXPECT_EQ(state, nullptr);
}

TEST(EpContextDataApiTest, ConfigReturnsConfiguredCallbacks) {
  Ort::SessionOptions session_options;

  EpContextReadCallbackState callback_state{};
  session_options.SetEpContextDataReadFunc(EpContextReadCallback, &callback_state,
                                           kEpContextApiTestMaxDataSize);

  Ort::EpContextConfig ep_context_config{session_options};

  OrtReadNamedBufferFunc read_func = nullptr;
  void* read_state = nullptr;
  size_t max_data_size = 0;
  ep_context_config.GetReadFunc(read_func, read_state, max_data_size);
  EXPECT_EQ(read_func, EpContextReadCallback);
  EXPECT_EQ(read_state, &callback_state);
  EXPECT_EQ(max_data_size, kEpContextApiTestMaxDataSize);

  OrtWriteNamedBufferFunc write_func = nullptr;
  void* write_state = nullptr;
  ep_context_config.GetWriteFunc(write_func, write_state);
  EXPECT_EQ(write_func, nullptr);
  EXPECT_EQ(write_state, nullptr);
}

TEST(EpContextDataApiTest, ConfigIsAnImmutableSnapshotOfSessionOptions) {
  Ort::SessionOptions session_options;
  EpContextReadCallbackState first_state{};
  EpContextReadCallbackState second_state{};

  session_options.SetEpContextDataReadFunc(EpContextReadCallback, &first_state,
                                           kEpContextApiTestMaxDataSize);
  Ort::EpContextConfig first_config{session_options};

  session_options.SetEpContextDataReadFunc(EpContextReadCallback, &second_state,
                                           kEpContextApiTestMaxDataSize / 2);
  Ort::EpContextConfig second_config{session_options};
  session_options.ClearEpContextDataReadFunc();

  OrtReadNamedBufferFunc read_func = nullptr;
  void* read_state = nullptr;
  size_t max_data_size = 0;
  first_config.GetReadFunc(read_func, read_state, max_data_size);
  EXPECT_EQ(read_func, EpContextReadCallback);
  EXPECT_EQ(read_state, &first_state);
  EXPECT_EQ(max_data_size, kEpContextApiTestMaxDataSize);

  read_func = nullptr;
  read_state = nullptr;
  max_data_size = 0;
  second_config.GetReadFunc(read_func, read_state, max_data_size);
  EXPECT_EQ(read_func, EpContextReadCallback);
  EXPECT_EQ(read_state, &second_state);
  EXPECT_EQ(max_data_size, kEpContextApiTestMaxDataSize / 2);
}

TEST(EpContextDataApiTest, ReadFuncCanBeCleared) {
  Ort::SessionOptions session_options;

  EpContextReadCallbackState callback_state{};
  session_options.SetEpContextDataReadFunc(EpContextReadCallback, &callback_state,
                                           kEpContextApiTestMaxDataSize);
  session_options.ClearEpContextDataReadFunc();

  Ort::EpContextConfig ep_context_config{session_options};
  OrtReadNamedBufferFunc read_func = EpContextReadCallback;
  void* read_state = reinterpret_cast<void*>(0x1);
  size_t max_data_size = 0;
  ep_context_config.GetReadFunc(read_func, read_state, max_data_size);
  EXPECT_EQ(read_func, nullptr);
  EXPECT_EQ(read_state, nullptr);
  EXPECT_EQ(max_data_size, std::numeric_limits<size_t>::max());
}
#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)
TEST(EpContextDataApiTest, WriteFuncCanBeSetOnModelCompilationOptions) {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "EpContextDataWriteFuncCanBeSetOnModelCompilationOptions"};
  Ort::SessionOptions session_options;
  Ort::ModelCompilationOptions compilation_options{env, session_options};

  EpContextWriteCallbackState callback_state{};
  compilation_options.SetEpContextDataWriteFunc(EpContextWriteCallback, &callback_state);

  const std::vector<char> payload{'b', 'i', 'n', 'a', 'r', 'y'};
  ASSERT_ORTSTATUS_OK(EpContextWriteCallback(&callback_state, "engine.bin", payload.data(), payload.size()));

  ASSERT_TRUE(callback_state.called);
  EXPECT_EQ(callback_state.file_name, "engine.bin");
  EXPECT_EQ(callback_state.payload, payload);
}

TEST(EpContextDataApiTest, WriteFuncCanBeCleared) {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "EpContextDataWriteFuncCanBeCleared"};
  Ort::SessionOptions session_options;
  Ort::ModelCompilationOptions compilation_options{env, session_options};

  EpContextWriteCallbackState callback_state{};
  compilation_options.SetEpContextDataWriteFunc(EpContextWriteCallback, &callback_state);

  // A null write_func clears the previously set callback (symmetric with the read setter) and must be accepted
  // rather than rejected with ORT_INVALID_ARGUMENT.
  compilation_options.SetEpContextDataWriteFunc(nullptr, &callback_state);
}

TEST(EpContextDataApiTest, WriteFuncCanBeUsedWithEpContextBinaryInformation) {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "EpContextDataWriteFuncCanBeUsedWithEpContextBinaryInformation"};
  Ort::SessionOptions session_options;
  Ort::ModelCompilationOptions compilation_options{env, session_options};

  // The EPContext write callback and the EPContext binary information may be configured together; neither call
  // rejects the other.
  ASSERT_NO_THROW(compilation_options.SetEpContextBinaryInformation(ORT_TSTR("ep_context_dir/"),
                                                                    ORT_TSTR("compiled_model.onnx")));

  EpContextWriteCallbackState callback_state{};
  compilation_options.SetEpContextDataWriteFunc(EpContextWriteCallback, &callback_state);

  const std::vector<char> payload{'c', 't', 'x'};
  ASSERT_ORTSTATUS_OK(EpContextWriteCallback(&callback_state, "logical_context.bin", payload.data(), payload.size()));

  ASSERT_TRUE(callback_state.called);
  EXPECT_EQ(callback_state.file_name, "logical_context.bin");
  EXPECT_EQ(callback_state.payload, payload);
}
#endif  // !defined(ORT_MINIMAL_BUILD)

#if !defined(ORT_MINIMAL_BUILD)
TEST(EpContextDataApiTest, ReturnedReadFuncAllowsEmptyPayloads) {
  Ort::SessionOptions session_options;

  EpContextReadCallbackState callback_state{};
  session_options.SetEpContextDataReadFunc(EpContextReadCallback, &callback_state,
                                           kEpContextApiTestMaxDataSize);

  Ort::EpContextConfig ep_context_config{session_options};
  OrtReadNamedBufferFunc read_func = nullptr;
  void* read_state = nullptr;
  size_t max_data_size = 0;
  ep_context_config.GetReadFunc(read_func, read_state, max_data_size);
  ASSERT_EQ(read_func, EpContextReadCallback);
  ASSERT_EQ(read_state, &callback_state);
  ASSERT_EQ(max_data_size, kEpContextApiTestMaxDataSize);

  Ort::AllocatorWithDefaultOptions allocator;
  void* buffer = reinterpret_cast<void*>(0x1);
  size_t buffer_size = 1;
  ASSERT_ORTSTATUS_OK(read_func(read_state, "empty.bin", allocator, &buffer, &buffer_size));

  EXPECT_TRUE(callback_state.called);
  EXPECT_EQ(callback_state.file_name, "empty.bin");
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(buffer_size, 0U);
}
#endif  // !defined(ORT_MINIMAL_BUILD)

TEST(EpContextDataApiTest, ReadCallbackRejectsOversizedArtifactBeforeAllocation) {
  EpContextReadCallbackState callback_state;
  callback_state.payload = {'t', 'o', 'o', '-', 'l', 'a', 'r', 'g', 'e'};
  callback_state.max_data_size = callback_state.payload.size() - 1;

  Ort::AllocatorWithDefaultOptions allocator;
  void* buffer = reinterpret_cast<void*>(0x1);
  size_t buffer_size = 0;
  ExpectFailureOrtStatus(EpContextReadCallback(&callback_state, "oversized.bin", allocator, &buffer, &buffer_size),
                         ORT_INVALID_ARGUMENT, "application callback size limit");
  EXPECT_TRUE(callback_state.called);
  EXPECT_FALSE(callback_state.allocation_attempted);
  EXPECT_EQ(buffer, nullptr);
  EXPECT_EQ(buffer_size, callback_state.payload.size());
}

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4996)
#elif defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

TEST(EpContextDataApiTest, ExperimentalCxxWrapperRetainsLifecycleApi) {
  static_assert(std::is_same_v<decltype(std::declval<Ort::Experimental::EpContextConfig&>().reset()), void>);
  static_assert(noexcept(std::declval<Ort::Experimental::EpContextConfig&>().release()));

  const auto& ort_api = Ort::GetApi();
  auto* experimental_set_read =
      Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_FnOrThrow(&ort_api);
  Ort::SessionOptions session_options;
  EpContextReadCallbackState callback_state{};
  ASSERT_ORTSTATUS_OK(experimental_set_read(session_options, EpContextReadCallback, &callback_state));

  Ort::Experimental::EpContextConfig config{session_options};
  const auto* config_handle = config.get();
  config = std::move(config);
  EXPECT_EQ(config.get(), config_handle);

  OrtReadNamedBufferFunc read_func = nullptr;
  void* read_state = nullptr;
  config.GetReadFunc(read_func, read_state);
  EXPECT_EQ(read_func, EpContextReadCallback);
  EXPECT_EQ(read_state, &callback_state);

  config.reset();
  EXPECT_FALSE(config);
  EXPECT_EQ(config.release(), nullptr);
}

#if defined(_MSC_VER)
#pragma warning(pop)
#elif defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#if !defined(ORT_MINIMAL_BUILD)
TEST(EpContextDataApiTest, StableCxxWrapperHandlesSelfMove) {
  Ort::SessionOptions session_options;
  Ort::EpContextConfig config{session_options};
  const auto* config_handle = config.get();

  config = std::move(config);
  EXPECT_EQ(config.get(), config_handle);

  config.reset();
  EXPECT_FALSE(config);
}
#endif  // !defined(ORT_MINIMAL_BUILD)

TEST(EpContextDataApiTest, ExperimentalV28NamesForwardToStableTransport) {
  const auto& ort_api = Ort::GetApi();
  auto* experimental_set_read =
      Ort::Experimental::Get_OrtApi_SessionOptions_SetEpContextDataReadFunc_SinceV28_FnOrThrow(&ort_api);
  auto* experimental_get_config =
      Ort::Experimental::Get_OrtEpApi_SessionOptions_GetEpContextConfig_SinceV28_FnOrThrow(&ort_api);
  auto* experimental_release_config =
      Ort::Experimental::Get_OrtEpApi_ReleaseEpContextConfig_SinceV28_FnOrThrow(&ort_api);
  auto* experimental_get_read =
      Ort::Experimental::Get_OrtEpApi_EpContextConfig_GetEpContextDataReadFunc_SinceV28_FnOrThrow(&ort_api);
  auto* experimental_get_write =
      Ort::Experimental::Get_OrtEpApi_EpContextConfig_GetEpContextDataWriteFunc_SinceV28_FnOrThrow(&ort_api);

  Ort::SessionOptions session_options;
  EpContextReadCallbackState callback_state{};
  ASSERT_ORTSTATUS_OK(experimental_set_read(session_options, EpContextReadCallback, &callback_state));

  OrtEpContextConfig* config = nullptr;
  ASSERT_ORTSTATUS_OK(experimental_get_config(session_options, &config));
  auto release_config = gsl::finally([&]() { experimental_release_config(config); });

  OrtReadNamedBufferFunc read_func = nullptr;
  void* read_state = nullptr;
  ASSERT_ORTSTATUS_OK(experimental_get_read(config, &read_func, &read_state));
  EXPECT_EQ(read_func, EpContextReadCallback);
  EXPECT_EQ(read_state, &callback_state);

  OrtWriteNamedBufferFunc write_func = EpContextWriteCallback;
  void* write_state = reinterpret_cast<void*>(0x1);
  ASSERT_ORTSTATUS_OK(experimental_get_write(config, &write_func, &write_state));
  EXPECT_EQ(write_func, nullptr);
  EXPECT_EQ(write_state, nullptr);

#if !defined(ORT_MINIMAL_BUILD)
  auto* experimental_set_write =
      Ort::Experimental::Get_OrtCompileApi_ModelCompilationOptions_SetEpContextDataWriteFunc_SinceV28_FnOrThrow(
          &ort_api);
  ExpectFailureOrtStatus(experimental_set_write(nullptr, EpContextWriteCallback, nullptr), ORT_INVALID_ARGUMENT,
                         "OrtModelCompilationOptions is NULL");
#endif
}
