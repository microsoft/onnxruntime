// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <filesystem>
#include <string_view>

#include "gtest/gtest.h"

#include "core/common/path_string.h"

#include "test/shared_lib/runtime_path_test_shared_library/runtime_path_test_shared_library.h"
#include "test/util/include/file_util.h"

namespace onnxruntime::test {

namespace {
bool IsDirectorySeparator(PathChar c) {
  constexpr std::array dir_separators{ORT_TSTR('/'), std::filesystem::path::preferred_separator};
  return std::find(dir_separators.begin(), dir_separators.end(), c) != dir_separators.end();
}
}  // namespace

#if !defined(_AIX)
TEST(GetRuntimePathFromSharedLibraryTest, Basic) {
#else
TEST(GetRuntimePathFromSharedLibraryTest, DISABLED_Basic) {
#endif
  const auto* runtime_path_cstr = OrtTestGetSharedLibraryRuntimePath();
  ASSERT_NE(runtime_path_cstr, nullptr);

  const auto runtime_path_str = std::basic_string_view<PathChar>{runtime_path_cstr};
  ASSERT_FALSE(runtime_path_str.empty());
  ASSERT_TRUE(IsDirectorySeparator(runtime_path_str.back()));

  const auto runtime_path = std::filesystem::path{runtime_path_str};
  ASSERT_TRUE(runtime_path.is_absolute());

  // Check that the runtime path contains the shared library file.
  const auto shared_library_file_name =
      GetSharedLibraryFileName(ORT_TSTR("onnxruntime_runtime_path_test_shared_library"));

  const auto shared_library_path = runtime_path / shared_library_file_name;

  // Get canonical path to ensure it exists and resolve any symlinks.
  std::error_code ec{};
  const auto canonical_shared_library_path = std::filesystem::canonical(shared_library_path, ec);
  ASSERT_FALSE(ec) << "Failed to get canonical path to shared library file '" << shared_library_path
                   << "'. Error: " << ec.message();

  ASSERT_TRUE(std::filesystem::is_regular_file(canonical_shared_library_path));
}

}  // namespace onnxruntime::test
