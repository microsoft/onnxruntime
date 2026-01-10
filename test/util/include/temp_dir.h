// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/common/path_string.h"

namespace onnxruntime {
namespace test {

/**
 * Creates a temporary directory on construction and deletes it on destruction.
 */
class TemporaryDirectory {
 public:
  /**
   * Constructor. Creates the temporary directory.
   *
   * Currently, the provided path is used directly as the temporary directory.
   *
   * @param path The temporary directory path.
   * @param delete_if_exists If true and the temporary directory exists, delete and re-create it.
   *                         If false, fail if the directory exists.
   */
  explicit TemporaryDirectory(const PathString& path, bool delete_if_exists = true);

  /**
   * Destructor. Deletes the temporary directory.
   */
  ~TemporaryDirectory();

  /**
   * Gets the temporary directory path.
   */
  const PathString& Path() const { return path_; }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TemporaryDirectory);

  const PathString path_;
};

}  // namespace test
}  // namespace onnxruntime
