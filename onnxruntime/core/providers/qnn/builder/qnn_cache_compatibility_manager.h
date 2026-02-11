// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "QnnTypes.h"

#include "core/providers/qnn/builder/qnn_backend_manager.h"
#include "core/providers/qnn/ort_api.h"

namespace onnxruntime {
namespace qnn {

// Extended Qnn_Version_t to support comparing between two versions.
struct QnnVersion : Qnn_Version_t {
  QnnVersion& operator=(const Qnn_Version_t& other) {
    major = other.major;
    minor = other.minor;
    patch = other.patch;
    return *this;
  }

  bool operator==(const QnnVersion& other) const {
    return major == other.major && minor == other.minor;
  }

  bool operator<(const QnnVersion& other) const {
    return major != other.major ? major < other.major : minor < other.minor;
  }

  bool operator<=(const QnnVersion& other) const {
    return major != other.major ? major <= other.major : minor <= other.minor;
  }

  bool operator>(const QnnVersion& other) const {
    return major != other.major ? major > other.major : minor > other.minor;
  }

  bool operator>=(const QnnVersion& other) const {
    return major != other.major ? major >= other.major : minor >= other.minor;
  }
};

struct QnnCompatibilityInfo {
  uint32_t backend_id = 0;
  QnnVersion sdk_version = QNN_VERSION_INIT;
  QnnVersion backend_api_version = QNN_VERSION_INIT;
  QnnVersion context_blob_version = QNN_VERSION_INIT;
  uint32_t htp_arch = 0;
  bool is_htp_usr_drv = false;
};

class QnnCacheCompatibilityManager {
 public:
  QnnCacheCompatibilityManager(QnnBackendManager* qnn_backend_manager)
      : qnn_backend_manager_(qnn_backend_manager) {}

  Ort::Status GetCompatibilityInfo(unsigned char* context_buffer,
                                   uint64_t context_buffer_size,
                                   QnnCompatibilityInfo& info);

  Ort::Status ValidateCompatibilityInfo(const QnnCompatibilityInfo& info, OrtCompiledModelCompatibility& compatibility);

 private:
  Ort::Status CreateFakeContextBinary(std::unique_ptr<unsigned char[]>& context_buffer, uint64_t& context_buffer_size);

 private:
  QnnBackendManager* qnn_backend_manager_;
};

}  // namespace qnn
}  // namespace onnxruntime
