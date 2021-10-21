// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <tuple>

#include "core/common/common.h"
#include "core/common/path_string.h"

namespace onnxruntime {
namespace training {

/**
 * This class keeps track of known checkpoints.
 * It does not handle the actual creation or deletion of checkpoints, but it
 * provides the corresponding file paths to use.
 *
 * Note: The checkpoint order is based on the checkpoint file names.
 */
class CheckpointRegistry {
 public:
  using CheckpointId = size_t;

  /**
   * Constructor.
   *
   * @param checkpoints_directory_path The directory containing the
   *        checkpoints.
   * @param max_num_checkpoints The maximum number of checkpoints to maintain.
   *        The maximum of 1, this value, and the number of existing
   *        checkpoints is the actual limit.
   */
  CheckpointRegistry(const PathString& checkpoints_directory_path, size_t max_num_checkpoints);

  /**
   * Registers a new checkpoint.
   *
   * Callers are responsible for deletion of the returned old checkpoint path
   * and creation of the returned new checkpoint path. It is possible for the
   * old and new checkpoint paths to be the same. That means the checkpoint
   * should be replaced.
   *
   * @param id The checkpoint ID.
   * @param[out] new_checkpoint_path The path to use for writing the new
   *             checkpoint.
   * @param[out] should_remove_old_checkpoint Whether an old checkpoint should
   *             be removed.
   * @param[out] old_checkpoint_path The path to the checkpoint to be removed.
   *             This is only meaningful if should_remove_old_checkpoint is
   *             true.
   * @return The status of the operation.
   */
  Status AddCheckpoint(
      CheckpointId id,
      PathString& new_checkpoint_path,
      bool& should_remove_old_checkpoint, PathString& old_checkpoint_path);

  /**
   * Attempts to get the latest checkpoint.
   *
   * @param[out] latest_checkpoint_path The path to the latest checkpoint.
   * @return True if there is a latest checkpoint, false otherwise.
   */
  bool TryGetLatestCheckpoint(PathString& latest_checkpoint_path) const;

 private:
  using CheckpointIdToPathMap = std::map<CheckpointId, PathString>;

  static CheckpointIdToPathMap GetCheckpointsFromDirectory(
      const PathString& checkpoints_directory_path);

  const PathString checkpoints_directory_path_;
  CheckpointIdToPathMap checkpoints_;
  const size_t max_num_checkpoints_;
};

}  // namespace training
}  // namespace onnxruntime
