// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Unit tests for the sample-only EPContext data helpers in
// onnxruntime/test/autoep/library/ep_context_data_utils.h.

#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

#include <gsl/gsl>
#include <gtest/gtest.h>

#include "core/graph/model_editor_api_types.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_experimental_cxx_api.h"

#include "test/autoep/ep_context_data_callbacks.h"
#include "test/autoep/library/ep_context_data_utils.h"
#include "test/util/include/api_asserts.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {

namespace {

OrtStatus* ORT_API_CALL LoadInvalidEpContextDataCallback(void* state, const char* file_name,
                                                         OrtAllocator* /*allocator*/, void** buffer,
                                                         size_t* data_size) {
  auto* callback_state = static_cast<EpContextDataCallbackState*>(state);
  callback_state->read_called = true;
  callback_state->read_file_name = file_name;

  *buffer = nullptr;
  *data_size = 1;
  return nullptr;
}

void ExpectOrtStatusError(OrtStatus* status_ptr, OrtErrorCode expected_code, std::string_view expected_message) {
  Ort::Status status{status_ptr};
  ASSERT_NE(status_ptr, nullptr) << "Expected a failure status, but the API returned nullptr (OK).";
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.GetErrorCode(), expected_code);
  EXPECT_THAT(std::string{status.GetErrorMessage()}, ::testing::HasSubstr(std::string{expected_message}));
}

std::filesystem::path PrepareTempTestDir(std::string_view name) {
  std::filesystem::path test_dir = std::string{name};
  std::filesystem::remove_all(test_dir);
  std::filesystem::create_directories(test_dir);
  return test_dir;
}

}  // namespace

TEST(OrtEpLibrary, EpContextDataUtils_PathHelpersRoundTrip) {
  const auto& api = Ort::GetApi();
  const std::string file_name = "context_data.bin";

#ifdef _WIN32
  std::wstring wide_file_name;
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::Utf8ToWideString(api, file_name, wide_file_name));
  ASSERT_FALSE(wide_file_name.empty());
  std::string round_tripped_file_name;
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::WideToUtf8String(api, wide_file_name, round_tripped_file_name));
  EXPECT_EQ(round_tripped_file_name, file_name);

  const std::string invalid_utf8(1, static_cast<char>(0xff));
  std::wstring invalid_wide;
  ExpectOrtStatusError(ep_context_data_utils::Utf8ToWideString(api, invalid_utf8, invalid_wide),
                       ORT_INVALID_ARGUMENT, "not valid UTF-8");
  EXPECT_TRUE(invalid_wide.empty());
#endif

  std::filesystem::path file_path;
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::Utf8Path(api, file_name.c_str(), file_path));
  ASSERT_FALSE(file_path.empty());
  std::string round_tripped_path;
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::PathToUtf8String(api, file_path, round_tripped_path));
  EXPECT_EQ(round_tripped_path, file_name);

  std::filesystem::path empty_path;
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::Utf8Path(api, nullptr, empty_path));
  EXPECT_TRUE(empty_path.empty());
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::Utf8Path(api, "", empty_path));
  EXPECT_TRUE(empty_path.empty());
}

TEST(OrtEpLibrary, EpContextDataUtils_ResolvePathAndInvalidArguments) {
  const auto& api = Ort::GetApi();
  std::filesystem::path data_path;

  data_path = "stale.ctx";
  ExpectOrtStatusError(ep_context_data_utils::ResolveEpContextDataPath(api, nullptr, nullptr, data_path),
                       ORT_INVALID_ARGUMENT, "EPContext data file name must not be empty");
  EXPECT_TRUE(data_path.empty());

  data_path = "stale.ctx";
  ExpectOrtStatusError(ep_context_data_utils::ResolveEpContextDataPath(api, "", nullptr, data_path),
                       ORT_INVALID_ARGUMENT, "EPContext data file name must not be empty");
  EXPECT_TRUE(data_path.empty());

  ASSERT_ORTSTATUS_OK(ep_context_data_utils::ResolveEpContextDataPath(api, "relative.ctx", nullptr, data_path));
  std::string resolved_data_path;
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::PathToUtf8String(api, data_path, resolved_data_path));
  EXPECT_EQ(resolved_data_path, "relative.ctx");

  ExpectOrtStatusError(ep_context_data_utils::WriteEpContextDataToFile(api, "unused.ctx", nullptr, nullptr, 1),
                       ORT_INVALID_ARGUMENT, "EPContext data buffer must not be null for non-empty data");
  ExpectOrtStatusError(ep_context_data_utils::WriteEpContextDataWithFileFallback(api, nullptr, "unused.ctx", nullptr,
                                                                                 nullptr, 1),
                       ORT_INVALID_ARGUMENT, "EPContext data buffer must not be null for non-empty data");

  std::vector<char> data;
  ExpectOrtStatusError(ep_context_data_utils::ReadEpContextDataWithFileFallback(api, nullptr, "", nullptr, data),
                       ORT_INVALID_ARGUMENT, "EPContext data file name must not be empty");
  ExpectOrtStatusError(ep_context_data_utils::WriteEpContextDataWithFileFallback(api, nullptr, "", nullptr, nullptr, 0),
                       ORT_INVALID_ARGUMENT, "EPContext data file name must not be empty");
  ExpectOrtStatusError(ep_context_data_utils::WriteEpContextDataWithFileFallback(
                           api, nullptr, "logical_context_data.bin", "", nullptr, nullptr, 0),
                       ORT_INVALID_ARGUMENT, "EPContext data fallback file name must not be empty");
}

TEST(OrtEpLibrary, EpContextDataUtils_ResolvePathRejectsUnsafeNames) {
  const auto& api = Ort::GetApi();
  std::filesystem::path data_path;

  ExpectOrtStatusError(ep_context_data_utils::ResolveEpContextDataPath(api, "../escape.ctx", nullptr, data_path),
                       ORT_INVALID_ARGUMENT, "EPContext data file name must not contain path traversal");
  EXPECT_TRUE(data_path.empty());

#ifdef _WIN32
  const char* absolute_file_name = "C:\\temp\\escape.ctx";
  const char* drive_relative_file_name = "C:escape.ctx";
  const char* root_relative_file_name = "\\escape.ctx";
#else
  const char* absolute_file_name = "/tmp/escape.ctx";
#endif
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::ResolveEpContextDataPath(api, absolute_file_name, nullptr, data_path));
  EXPECT_TRUE(data_path.is_absolute());

#ifdef _WIN32
  ExpectOrtStatusError(ep_context_data_utils::WriteEpContextDataWithFileFallback(
                           api, nullptr, drive_relative_file_name, "unused.ctx", nullptr, nullptr, 0),
                       ORT_INVALID_ARGUMENT, "EPContext data file name must not be absolute or rooted");
  ExpectOrtStatusError(ep_context_data_utils::WriteEpContextDataWithFileFallback(
                           api, nullptr, root_relative_file_name, "unused.ctx", nullptr, nullptr, 0),
                       ORT_INVALID_ARGUMENT, "EPContext data file name must not be absolute or rooted");
#endif

  std::vector<char> data;
  ExpectOrtStatusError(ep_context_data_utils::ReadEpContextDataFromFile(api, "../escape.ctx", nullptr, data),
                       ORT_INVALID_ARGUMENT, "EPContext data file name must not contain path traversal");
  ExpectOrtStatusError(ep_context_data_utils::WriteEpContextDataWithFileFallback(
                           api, nullptr, absolute_file_name, "unused.ctx", nullptr, nullptr, 0),
                       ORT_INVALID_ARGUMENT, "EPContext data file name must not be absolute or rooted");

  ModelEditorGraph empty_model_path_graph;
  ExpectOrtStatusError(ep_context_data_utils::ResolveEpContextDataPath(api, "../escape.ctx",
                                                                       empty_model_path_graph.ToExternal(), data_path),
                       ORT_INVALID_ARGUMENT, "requires a model path");

  // A model-derived name that designates a directory ("." or a trailing separator with an empty filename) is
  // rejected up front, rather than resolving to a directory and failing later with a confusing I/O error.
  ExpectOrtStatusError(ep_context_data_utils::ResolveEpContextDataPath(api, ".", empty_model_path_graph.ToExternal(),
                                                                       data_path),
                       ORT_INVALID_ARGUMENT, "must refer to a file");
  ExpectOrtStatusError(ep_context_data_utils::WriteEpContextDataWithFileFallback(
                           api, nullptr, ".", "unused.ctx", nullptr, nullptr, 0),
                       ORT_INVALID_ARGUMENT, "must refer to a file");
  ExpectOrtStatusError(ep_context_data_utils::WriteEpContextDataWithFileFallback(
                           api, nullptr, "sub/", "unused.ctx", nullptr, nullptr, 0),
                       ORT_INVALID_ARGUMENT, "must refer to a file");
}

TEST(OrtEpLibrary, EpContextDataUtils_ResolvePathRejectsSymlinkEscape) {
  const auto& api = Ort::GetApi();
  const std::filesystem::path test_dir = PrepareTempTestDir("ort_ep_context_data_utils_symlink_escape_test");
  auto cleanup = gsl::finally([&]() { std::filesystem::remove_all(test_dir); });

  const std::filesystem::path model_dir = test_dir / "model_dir";
  const std::filesystem::path outside_dir = test_dir / "outside_dir";
  ASSERT_TRUE(std::filesystem::create_directories(model_dir));
  ASSERT_TRUE(std::filesystem::create_directories(outside_dir));

  const std::filesystem::path symlink_path = model_dir / "linked_outside";
  // Relative symlink targets are resolved by the OS relative to the link's own directory, not the test's working
  // directory. Point to the sibling outside_dir using a link-relative target; using the test_dir-relative
  // `outside_dir` path here would create a dangling link under model_dir, and weakly_canonical() would not traverse it.
  const std::filesystem::path symlink_target = std::filesystem::path{".."} / outside_dir.filename();
  std::error_code symlink_error;
  std::filesystem::create_directory_symlink(symlink_target, symlink_path, symlink_error);
  if (symlink_error) {
    GTEST_SKIP() << "Unable to create directory symlink for containment test: " << symlink_error.message();
  }

  ModelEditorGraph graph;
  graph.model_path = model_dir / "model.onnx";

  std::filesystem::path data_path;
  ExpectOrtStatusError(ep_context_data_utils::ResolveEpContextDataPath(api, "linked_outside/escape.ctx",
                                                                       graph.ToExternal(), data_path),
                       ORT_INVALID_ARGUMENT, "resolve to a path within the model directory");
  EXPECT_TRUE(data_path.empty());
}

TEST(OrtEpLibrary, EpContextDataUtils_FileFallbackReadsAndWrites) {
  const auto& api = Ort::GetApi();
  const std::filesystem::path test_dir = PrepareTempTestDir("ort_ep_context_data_utils_file_fallback_test");
  auto cleanup = gsl::finally([&]() { std::filesystem::remove_all(test_dir); });

  const std::string payload = "file fallback payload";
  const std::filesystem::path data_path = test_dir / "context_data.bin";
  std::string data_file_name;
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::PathToUtf8String(api, data_path, data_file_name));

  ASSERT_ORTSTATUS_OK(ep_context_data_utils::WriteEpContextDataToFile(api, data_file_name.c_str(), nullptr,
                                                                      payload.data(), payload.size()));

  std::vector<char> data;
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::ReadEpContextDataFromFile(api, data_file_name.c_str(), nullptr, data));
  EXPECT_EQ(std::string(data.begin(), data.end()), payload);

  const std::filesystem::path wrapper_data_path = test_dir / "wrapper_context_data.bin";
  std::string wrapper_data_file_name;
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::PathToUtf8String(api, wrapper_data_path, wrapper_data_file_name));
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::WriteEpContextDataWithFileFallback(
      api, nullptr, wrapper_data_file_name.c_str(), nullptr, payload.data(), payload.size()));

  data.clear();
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::ReadEpContextDataWithFileFallback(
      api, nullptr, wrapper_data_file_name.c_str(), nullptr, data));
  EXPECT_EQ(std::string(data.begin(), data.end()), payload);

  const std::filesystem::path fallback_data_path = test_dir / "fallback_context_data.bin";
  std::string fallback_data_file_name;
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::PathToUtf8String(api, fallback_data_path, fallback_data_file_name));
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::WriteEpContextDataWithFileFallback(
      api, nullptr, "logical_context_data.bin", fallback_data_file_name.c_str(), nullptr, payload.data(),
      payload.size()));

  const std::filesystem::path unsafe_logical_fallback_path = test_dir / "unsafe_logical_context_data.bin";
  std::string unsafe_logical_fallback_file_name;
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::PathToUtf8String(api, unsafe_logical_fallback_path,
                                                              unsafe_logical_fallback_file_name));
  ExpectOrtStatusError(ep_context_data_utils::WriteEpContextDataWithFileFallback(
                           api, nullptr, "../logical_context_data.bin",
                           unsafe_logical_fallback_file_name.c_str(), nullptr, payload.data(), payload.size()),
                       ORT_INVALID_ARGUMENT, "EPContext data file name must not contain path traversal");

  data.clear();
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::ReadEpContextDataFromFile(api, fallback_data_file_name.c_str(), nullptr,
                                                                       data));
  EXPECT_EQ(std::string(data.begin(), data.end()), payload);

  const std::filesystem::path empty_data_path = test_dir / "empty_context_data.bin";
  std::string empty_data_file_name;
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::PathToUtf8String(api, empty_data_path, empty_data_file_name));
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::WriteEpContextDataWithFileFallback(
      api, nullptr, empty_data_file_name.c_str(), nullptr, nullptr, 0));

  data.assign({'s', 't', 'a', 'l', 'e'});
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::ReadEpContextDataWithFileFallback(
      api, nullptr, empty_data_file_name.c_str(), nullptr, data));
  EXPECT_TRUE(data.empty());

  const std::filesystem::path missing_data_path = test_dir / "missing_context_data.bin";
  std::string missing_data_file_name;
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::PathToUtf8String(api, missing_data_path, missing_data_file_name));
  ExpectOrtStatusError(ep_context_data_utils::ReadEpContextDataFromFile(api, missing_data_file_name.c_str(), nullptr,
                                                                        data),
                       ORT_FAIL, "Failed to open EPContext data file for read");
}

TEST(OrtEpLibrary, EpContextDataUtils_CallbackFallbackUsesCallbacks) {
  const auto& api = Ort::GetApi();

  EpContextDataCallbackState read_callback_state;
  read_callback_state.payload = {'c', 'a', 'l', 'l', 'b', 'a', 'c', 'k'};
  EpContextDataCallbackState write_callback_state;

  std::vector<char> data;
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::ReadEpContextDataWithFileFallback(
      api, LoadEpContextDataCallback, &read_callback_state, "callback_context.bin", nullptr, data));
  ASSERT_TRUE(read_callback_state.read_called);
  EXPECT_EQ(read_callback_state.read_file_name, "callback_context.bin");
  EXPECT_EQ(data, read_callback_state.payload);

  const std::string payload = "callback write payload";
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::WriteEpContextDataWithFileFallback(
      api, StoreEpContextDataCallback, &write_callback_state, "callback_write_context.bin",
      "callback_write_context.bin", nullptr, payload.data(), payload.size()));
  ASSERT_TRUE(write_callback_state.write_called);
  EXPECT_EQ(write_callback_state.write_file_name, "callback_write_context.bin");
  EXPECT_EQ(std::string(write_callback_state.payload.begin(), write_callback_state.payload.end()), payload);

  write_callback_state = {};
  const std::string payload_with_unused_fallback = "callback write payload with unused fallback";
  // With a callback present the file fallback is never used, so the empty fallback name is accepted (not validated).
  ASSERT_ORTSTATUS_OK(ep_context_data_utils::WriteEpContextDataWithFileFallback(
      api, StoreEpContextDataCallback, &write_callback_state,
      "callback_write_context_unused_fallback.bin", "", nullptr,
      payload_with_unused_fallback.data(), payload_with_unused_fallback.size()));
  ASSERT_TRUE(write_callback_state.write_called);
  EXPECT_EQ(write_callback_state.write_file_name, "callback_write_context_unused_fallback.bin");
  EXPECT_EQ(std::string(write_callback_state.payload.begin(), write_callback_state.payload.end()),
            payload_with_unused_fallback);
}

TEST(OrtEpLibrary, EpContextDataUtils_ReadCallbackRejectsNullBufferForNonEmptyPayload) {
  const auto& api = Ort::GetApi();

  EpContextDataCallbackState read_callback_state;

  std::vector<char> data;
  ExpectOrtStatusError(ep_context_data_utils::ReadEpContextDataWithFileFallback(
                           api, LoadInvalidEpContextDataCallback, &read_callback_state,
                           "invalid_callback_context.bin", nullptr, data),
                       ORT_FAIL, "OrtReadNamedBufferFunc returned a null buffer for non-empty EPContext data");
  ASSERT_TRUE(read_callback_state.read_called);
  EXPECT_EQ(read_callback_state.read_file_name, "invalid_callback_context.bin");
}

}  // namespace test
}  // namespace onnxruntime
