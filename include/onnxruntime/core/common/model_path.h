// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <filesystem>
#include <memory>
#include <ostream>
#include <type_traits>
#include <utility>

namespace onnxruntime {

class Env;

// A model pathname together with the external-data directories captured when the model was opened.
// Copies share the immutable directory context. Converting back to a filesystem path is explicit.
class ModelPath {
 public:
  struct ExternalDataDirectories {
    std::filesystem::path apparent;
    std::filesystem::path model_target;
  };

  ModelPath() = default;

  template <typename Source, std::enable_if_t<std::is_constructible_v<std::filesystem::path, Source>, int> = 0>
  ModelPath(Source&& source) : path_(std::forward<Source>(source)) {}

  const std::filesystem::path& Path() const noexcept { return path_; }
  const auto& native() const noexcept { return path_.native(); }
  const auto* c_str() const noexcept { return path_.c_str(); }
  auto string() const { return path_.string(); }
  auto u8string() const { return path_.u8string(); }
  auto parent_path() const { return path_.parent_path(); }
  auto filename() const { return path_.filename(); }
  auto extension() const { return path_.extension(); }
  bool has_filename() const { return path_.has_filename(); }
  bool empty() const noexcept { return path_.empty(); }

  const ExternalDataDirectories* GetExternalDataDirectories() const noexcept { return directories_.get(); }

  friend bool operator==(const ModelPath& left, const ModelPath& right) { return left.path_ == right.path_; }
  friend bool operator!=(const ModelPath& left, const ModelPath& right) { return !(left == right); }
  friend std::ostream& operator<<(std::ostream& os, const ModelPath& path) { return os << path.path_; }

 private:
  friend class Env;
  std::filesystem::path path_;
  std::shared_ptr<const ExternalDataDirectories> directories_;
};

}  // namespace onnxruntime
