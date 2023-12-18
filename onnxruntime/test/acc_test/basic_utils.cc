// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "basic_utils.h"
#include <fstream>
#include <string>

bool FillBytesFromBinaryFile(Span<char> array, const std::string& binary_filepath) {
  std::ifstream input_ifs(binary_filepath, std::ifstream::binary);

  if (!input_ifs.is_open()) {
    return false;
  }

  input_ifs.seekg(0, input_ifs.end);
  auto file_byte_size = input_ifs.tellg();
  input_ifs.seekg(0, input_ifs.beg);

  if (static_cast<size_t>(file_byte_size) != array.size()) {
    return false;
  }

  input_ifs.read(array.data(), file_byte_size);
  return static_cast<bool>(input_ifs);
}

int32_t GetFileIndexSuffix(const std::string& filename_wo_ext, const char* prefix) {
  int32_t index = -1;
  const char* str = filename_wo_ext.c_str();

  // Move past the prefix.
  while (*str && *prefix && *str == *prefix) {
    str++;
    prefix++;
  }

  if (*prefix) {
    return -1;  // File doesn't start with the prefix.
  }

  // Parse the input index from file name.
  index = 0;
  while (*str) {
    int32_t c = *str;
    if (!(c >= '0' && c <= '9')) {
      return -1;  // Not a number.
    }

    index *= 10;
    index += (c - '0');
    str++;
  }

  return index;
}
