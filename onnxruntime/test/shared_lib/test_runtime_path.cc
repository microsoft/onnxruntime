// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>

#include "gtest/gtest.h"

#include "test/shared_lib/runtime_path_test_shared_library/runtime_path_test_shared_library.h"
#include "test/util/include/file_util.h"

namespace onnxruntime::test {

TEST(GetRuntimePathFromSharedLibraryTest, Basic) {
  const auto* runtime_path_cstr = OrtTestGetSharedLibraryRuntimePath();
  ASSERT_NE(runtime_path_cstr, nullptr);

  const auto runtime_path = std::filesystem::path{runtime_path_cstr};
  ASSERT_FALSE(runtime_path.empty());

  ASSERT_TRUE(runtime_path.is_absolute());

  const auto shared_library_file_name = GetSharedLibraryFileName("onnxruntime_runtime_path_test_shared_library");
  const auto canonical_shared_library_path = std::filesystem::canonical(runtime_path / shared_library_file_name);

  ASSERT_TRUE(std::filesystem::is_regular_file(canonical_shared_library_path));
}

}  // namespace onnxruntime::test
