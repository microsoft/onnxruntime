// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(__MACH__)
#else

#include <experimental/filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>

using namespace std::experimental::filesystem::v1;

namespace onnxruntime {
namespace test {

std::set<path> RetrieveAllHeaders(const path& include_folder_root) {
  std::set<path> headers;
  std::vector<path> paths{include_folder_root};
  while (!paths.empty()) {
    path node_data_root_path = paths.back();
    paths.pop_back();
    for (directory_iterator file_entry(node_data_root_path), end; file_entry != end; ++file_entry) {
      if (is_directory(*file_entry)) {
        paths.push_back(file_entry->path());
        continue;
      }

      if (!file_entry->path().has_extension()) continue;
      if (file_entry->path().extension() != ".h") continue;
      headers.insert(file_entry->path());
    }
  }
  return headers;
}

std::vector<path> RetrieveHeaderDependencies(const path& header, const path& include_folder_root) {
  std::vector<path> header_dependencies;
  std::ifstream header_stream(header);
  const std::size_t pos_off = 10;  // length of "#include \""
  if (!header_stream.good()) return header_dependencies;
  std::string line;
  while (std::getline(header_stream, line)) {
    line.erase(line.find_last_not_of(" \r\n\t") + 1);
    std::size_t pos = line.find("#include \"core/");
    if (pos != std::string::npos) {
      header_dependencies.push_back(include_folder_root / line.substr(pos + pos_off, line.length() - pos - pos_off - 1));
    } else {
      pos = line.find("#include <core/");
      if (pos != std::string::npos) {
        header_dependencies.push_back(include_folder_root / line.substr(pos + pos_off, line.length() - pos - pos_off - 1));
      }
    }
  }
  return header_dependencies;
}

TEST(HeaderFiles, EnsureAllPublicHeadersInIncludeFolder) {
  path include_folder_path(path(__FILE__).parent_path().parent_path().parent_path().parent_path() / "include" / "onnxruntime");
  std::set<path> headers(RetrieveAllHeaders(include_folder_path));

  for (auto header_iterator = headers.begin(); header_iterator != headers.end(); header_iterator++) {
    std::vector<path> header_dependencies(RetrieveHeaderDependencies(*header_iterator, include_folder_path));
    for (auto dependency_iterator = header_dependencies.begin(); dependency_iterator != header_dependencies.end(); dependency_iterator++) {
      EXPECT_TRUE(headers.find(*dependency_iterator) != headers.end()) << *header_iterator << " depends on " << *dependency_iterator << " that is not in include folder";
    }
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif
