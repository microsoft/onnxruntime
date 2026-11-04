// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

#include "gtest/gtest.h"

int OrtTestMain(int argc, char** argv);

int main(int argc, char** argv) {
  const char* runfiles = std::getenv("TEST_SRCDIR");
  const char* workspace = std::getenv("TEST_WORKSPACE");
  const char* temporary = std::getenv("TEST_TMPDIR");
  if (!runfiles || !workspace || !temporary) {
    std::cerr << "Missing Bazel test environment; run this executable with bazel test.\n";
    return 1;
  }
  try {
    bool death_test_child = false;
    for (int i = 1; i < argc; ++i) {
      if (std::string_view(argv[i]).find("--gtest_internal_run_death_test=") == 0) {
        death_test_child = true;
      }
    }
    // Existing tests use relative fixture paths and may write models alongside them.
    // Give each test an isolated writable copy rather than modifying source runfiles.
    auto source = std::filesystem::path(runfiles) / workspace / "onnxruntime" / "test" / "testdata";
    auto destination = std::filesystem::path(temporary) / "testdata";
#ifdef _WIN32
    // Bazel's runfiles prefix can put existing fixture names beyond MAX_PATH.
    source = L"\\\\?\\" + std::filesystem::absolute(source).make_preferred().native();
    destination = L"\\\\?\\" + std::filesystem::absolute(destination).make_preferred().native();
#endif
    // A re-executed death-test child shares its parent's prepared fixtures.
    if (!death_test_child) {
      std::filesystem::copy(source, destination, std::filesystem::copy_options::recursive);
    }
    int test_argc = 1;
    for (int i = 1; i < argc; ++i) {
      constexpr std::string_view library_arg = "--ort_test_library=";
      const std::string_view argument(argv[i]);
      if (argument.substr(0, library_arg.size()) == library_arg) {
        const auto library = std::filesystem::absolute(std::string(argument.substr(library_arg.size())));
        if (!death_test_child) {
          std::filesystem::copy_file(library, std::filesystem::path(temporary) / library.filename());
        }
      } else {
        argv[test_argc++] = argv[i];
      }
    }
    argc = test_argc;
    argv[argc] = nullptr;
    std::filesystem::current_path(temporary);
  } catch (const std::filesystem::filesystem_error& error) {
    std::cerr << "Cannot prepare writable Bazel test fixtures: " << error.what()
              << ". Check TEST_SRCDIR fixtures and available space in TEST_TMPDIR.\n";
    return 1;
  }
  const int result = OrtTestMain(argc, argv);
  if (result == 0 && !::testing::GTEST_FLAG(list_tests) &&
      ::testing::UnitTest::GetInstance()->successful_test_count() == 0) {
    std::cerr << "No selected tests passed. Choose an existing --test_filter and enable its required features.\n";
    return 1;
  }
  return result;
}
