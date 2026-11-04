// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdlib>
#include <filesystem>
#include <iostream>

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
    // Existing tests use relative fixture paths and may write models alongside them.
    // Give each test an isolated writable copy rather than modifying source runfiles.
    auto source = std::filesystem::path(runfiles) / workspace / "onnxruntime" / "test" / "testdata";
    auto destination = std::filesystem::path(temporary) / "testdata";
#ifdef _WIN32
    // Bazel's runfiles prefix can put existing fixture names beyond MAX_PATH.
    source = L"\\\\?\\" + std::filesystem::absolute(source).make_preferred().native();
    destination = L"\\\\?\\" + std::filesystem::absolute(destination).make_preferred().native();
#endif
    std::filesystem::copy(source, destination, std::filesystem::copy_options::recursive);
    std::filesystem::current_path(temporary);
  } catch (const std::filesystem::filesystem_error& error) {
    std::cerr << "Cannot prepare writable Bazel test fixtures: " << error.what()
              << ". Check TEST_SRCDIR fixtures and available space in TEST_TMPDIR.\n";
    return 1;
  }
  return OrtTestMain(argc, argv);
}
