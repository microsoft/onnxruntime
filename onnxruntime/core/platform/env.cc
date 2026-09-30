/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Portions Copyright (c) Microsoft Corporation

#include "core/platform/env.h"

namespace onnxruntime {

std::ostream& operator<<(std::ostream& os, const LogicalProcessors& aff) {
  os << "{";
  std::copy(aff.cbegin(), aff.cend(), std::ostream_iterator<int>(os, ", "));
  return os << "}";
}

std::ostream& operator<<(std::ostream& os, gsl::span<const LogicalProcessors> affs) {
  os << "{";
  for (const auto& aff : affs) {
    os << aff;
  }
  return os << "}";
}

Env::Env() = default;

common::Status Env::CaptureModelPath(const std::filesystem::path& path, ModelPath& model_path,
                                     bool allow_model_symlink) const {
  ORT_RETURN_IF(path.native().find(ORTCHAR_T{}) != PathString::npos, "Model path contains a null character.");
  ModelPath result{path};
#if defined(__wasm__)
  std::error_code current_path_error;
  std::filesystem::current_path(current_path_error);
  if (current_path_error) {
    model_path = std::move(result);
    return common::Status::OK();
  }
#endif
  auto directories = std::make_shared<ModelPath::ExternalDataDirectories>();
  PathString canonical;
  const auto parent = path.parent_path().empty() ? std::filesystem::path{"."} : path.parent_path();
  ORT_RETURN_IF_ERROR(GetWeaklyCanonicalPath(parent.native(), canonical));
  directories->apparent = canonical;
  if (allow_model_symlink && !path.empty()) {
    std::error_code error;
    const bool is_symlink = std::filesystem::is_symlink(path, error);
    ORT_RETURN_IF(error && error != std::errc::no_such_file_or_directory,
                  "Failed to inspect model path: ", path, ": ", error.message());
    if (is_symlink) {
      ORT_RETURN_IF_ERROR(GetWeaklyCanonicalPath(path.native(), canonical));
      directories->model_target = std::filesystem::path(canonical).parent_path();
    }
  }
  result.directories_ = std::move(directories);
  model_path = std::move(result);
  return common::Status::OK();
}

common::Status Env::OpenModelFile(const std::filesystem::path& path, std::unique_ptr<RandomAccessFile>& file,
                                  ModelPath& model_path) const {
  ModelPath result;
  ORT_RETURN_IF_ERROR(CaptureModelPath(path, result, false));
  ORT_RETURN_IF_NOT(result.directories_, "Model file opening requires a filesystem.");
  const auto& apparent = result.directories_->apparent;
  PathString canonical;
  ORT_RETURN_IF_ERROR(GetWeaklyCanonicalPath((apparent / path.filename()).native(), canonical));
  std::unique_ptr<RandomAccessFile> opened_file;
  auto status = OpenCanonicalFile(canonical.c_str(), opened_file);
  if (!status.IsOK()) {
    std::error_code error;
    const bool exists = std::filesystem::exists(path, error);
    if (!exists && (!error || error == std::errc::no_such_file_or_directory)) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NO_SUCHFILE, "Load model ", path, " failed. File doesn't exist");
    }
    return status;
  }
  result.directories_ = std::make_shared<ModelPath::ExternalDataDirectories>(
      ModelPath::ExternalDataDirectories{apparent, std::filesystem::path(canonical).parent_path()});
  file = std::move(opened_file);
  model_path = std::move(result);
  return common::Status::OK();
}

std::pair<int, std::string> GetErrnoInfo() {
  auto err = errno;
  std::string msg;

  if (err != 0) {
    char buf[512];

#if defined(_WIN32)
    auto ret = strerror_s(buf, sizeof(buf), err);
    msg = ret == 0 ? buf : "Failed to get error message";  // buf is guaranteed to be null terminated by strerror_s
#else
    // strerror_r return type differs by platform.
    auto ret = strerror_r(err, buf, sizeof(buf));
    if constexpr (std::is_same_v<decltype(ret), int>) {  // POSIX returns int
      msg = ret == 0 ? buf : "Failed to get error message";
    } else {
      // GNU returns char*
      msg = ret;
    }
#endif
  }

  return {err, msg};
}

}  // namespace onnxruntime
