// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/framework/checkpoint_registry.h"

#include <algorithm>
#include <sstream>

#include "re2/re2.h"

#include "core/platform/path_lib.h"
#include "core/platform/env.h"

namespace onnxruntime {
namespace training {

namespace {

// checkpoint file names look like this: "checkpoint_<id>"

PathString MakeCheckpointPath(
    const PathString& base_directory_path, CheckpointRegistry::CheckpointId id) {
  std::basic_stringstream<PathChar> checkpoint_path{};
  checkpoint_path << base_directory_path << GetPathSep<PathChar>()
                  << ORT_TSTR("checkpoint_") << id;
  return checkpoint_path.str();
}

bool ParseCheckpointFileName(
    const PathString& checkpoint_file_name, CheckpointRegistry::CheckpointId& id) {
  static RE2 re = {R"(^checkpoint_(\d+)$)"};
  return RE2::FullMatch(ToMBString(checkpoint_file_name), re, &id);
}

}  // namespace

CheckpointRegistry::CheckpointRegistry(
    const PathString& checkpoints_directory_path, size_t max_num_checkpoints)
    : checkpoints_directory_path_{checkpoints_directory_path},
      checkpoints_{GetCheckpointsFromDirectory(checkpoints_directory_path_)},
      max_num_checkpoints_{
          std::max({static_cast<size_t>(1), max_num_checkpoints, checkpoints_.size()})} {
}

Status CheckpointRegistry::AddCheckpoint(
    CheckpointId id,
    PathString& new_checkpoint_path,
    bool& should_remove_old_checkpoint, PathString& old_checkpoint_path) {
  const PathString checkpoint_path{MakeCheckpointPath(checkpoints_directory_path_, id)};

  const auto existing_checkpoint_it = checkpoints_.find(id);
  if (existing_checkpoint_it != checkpoints_.end()) {
    // exists, replace it
    should_remove_old_checkpoint = true;
    old_checkpoint_path = existing_checkpoint_it->second;
    new_checkpoint_path = existing_checkpoint_it->second = checkpoint_path;
  } else {
    // doesn't exist, add it
    should_remove_old_checkpoint = checkpoints_.size() == max_num_checkpoints_;
    if (should_remove_old_checkpoint) {
      const auto oldest_checkpoint_it = checkpoints_.begin();
      old_checkpoint_path = oldest_checkpoint_it->second;
      checkpoints_.erase(oldest_checkpoint_it);
    }

    checkpoints_.emplace(id, checkpoint_path);
    new_checkpoint_path = checkpoint_path;
  }

  return Status::OK();
}

bool CheckpointRegistry::TryGetLatestCheckpoint(PathString& latest_checkpoint_path) const {
  if (checkpoints_.empty()) return false;
  latest_checkpoint_path = checkpoints_.rbegin()->second;
  return true;
}

CheckpointRegistry::CheckpointIdToPathMap CheckpointRegistry::GetCheckpointsFromDirectory(
    const PathString& checkpoints_directory_path) {
  CheckpointIdToPathMap checkpoints{};
  if (Env::Default().FolderExists(checkpoints_directory_path)) {
    LoopDir(
        checkpoints_directory_path,
        [&checkpoints_directory_path, &checkpoints](
            const PathString& file_name, OrtFileType /*file_type*/) {
          CheckpointId id;
          if (ParseCheckpointFileName(file_name, id)) {
            checkpoints.emplace(
                id, ConcatPathComponent(checkpoints_directory_path, file_name));
          }
          return true;
        });
  }

  return checkpoints;
}

}  // namespace training
}  // namespace onnxruntime
