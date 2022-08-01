// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TVM_UTILS_H
#define TVM_UTILS_H

#include <fstream>
#include <streambuf>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "tvm_utils.h"  // NOLINT(build/include_subdir)


namespace onnxruntime {
namespace tvm {

bool IsFileExists(const std::string& filename) {
#ifdef _WIN32
    return _access(filename.c_str(), 0) == 0;
#else
    return access(filename.c_str(), F_OK) == 0;
#endif
}

std::string readFromFile(const std::string& file_path) {
  std::string str;

  std::ifstream t(file_path);
  t.seekg(0, std::ios::end);
  str.reserve(t.tellg());
  t.seekg(0, std::ios::beg);

  str.assign((std::istreambuf_iterator<char>(t)),
              std::istreambuf_iterator<char>());
  return str;
}

}   // namespace tvm
}   // namespace onnxruntime

#endif  // TVM_UTILS_H
