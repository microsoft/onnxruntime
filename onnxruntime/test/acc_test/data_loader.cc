// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "data_loader.h"
#include <algorithm>
#include <cassert>
#include <filesystem>
#include <iostream>

namespace acctest {

bool LoadIODataFromDisk(const std::vector<std::filesystem::path>& dataset_paths,
                        const std::vector<IOInfo>& io_infos,
                        const char* data_file_prefix,
                        std::vector<std::unique_ptr<char[]>>& dataset_data) {
  size_t total_data_size = 0;
  for (const auto& io_info : io_infos) {
    total_data_size += io_info.total_data_size;
  }

  dataset_data.clear();
  dataset_data.reserve(dataset_paths.size());

  for (const auto& dataset_path : dataset_paths) {
    dataset_data.emplace_back(std::make_unique<char[]>(total_data_size));

    size_t num_files_loaded = 0;

    for (const auto& data_file_entry : std::filesystem::directory_iterator{dataset_path}) {
      const std::filesystem::path& data_file_path = data_file_entry.path();

      if (!std::filesystem::is_regular_file(data_file_path)) {
        continue;
      }

      std::string data_filename_wo_ext = data_file_path.stem().string();
      if (data_filename_wo_ext.rfind(data_file_prefix, 0) != 0) {
        continue;
      }

      const int64_t io_index = GetFileIndexSuffix(data_filename_wo_ext, data_file_prefix);
      if (io_index < 0) {
        std::cerr << "[ERROR]: The file " << data_file_path << " does not have a properly formatted name"
                  << " (e.g., " << data_file_prefix << "0.raw)" << std::endl;
        return false;
      }

      if (io_index >= static_cast<int64_t>(io_infos.size())) {
        std::cerr << "[ERROR]: The input (or output) file index for file " << data_file_path
                  << " exceeds the number of inputs (or outputs) in the model ("
                  << io_infos.size() << ")" << std::endl;
        return false;
      }

      size_t offset = 0;
      for (int64_t i = 0; i < io_index; i++) {
        offset += io_infos[i].total_data_size;
      }
      assert(offset < total_data_size);

      Span<char> span_to_fill(dataset_data.back().get() + offset, io_infos[io_index].total_data_size);
      if (!FillBytesFromBinaryFile(span_to_fill, data_file_path.string())) {
        std::cerr << "[ERROR]: Unable to read raw data from file " << data_file_path << std::endl;
        return false;
      }

      num_files_loaded += 1;
    }

    if (num_files_loaded != io_infos.size()) {
      std::cerr << "[ERROR]: " << dataset_path << " does not have the expected number of "
                << data_file_prefix << "<i>.raw files. Loaded " << num_files_loaded << "files, but expected "
                << io_infos.size() << "files." << std::endl;
      return false;
    }
  }

  return true;
}
}  // namespace acctest
