// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef USE_GDS

#include "core/framework/external_data_loader.h"

namespace onnxruntime {
namespace cuda {

// IExternalDataLoader implementation that uses NVIDIA cuFile (GPUDirect Storage)
// to load external initializer data directly from disk to GPU memory,
// bypassing CPU memory staging.
//
// Requirements:
//   - Linux only (cuFile is not available on Windows)
//   - NVIDIA GPU with GPUDirect Storage support
//   - nvidia-fs kernel module loaded
//   - NVMe SSD with ext4/XFS filesystem
//   - CUDA 11.0+, driver >= 450
class GpuExternalDataLoader : public IExternalDataLoader {
 public:
  explicit GpuExternalDataLoader(int device_id);
  ~GpuExternalDataLoader();

  bool CanLoad(const OrtMemoryInfo& target_memory_info) const override;

  common::Status LoadTensor(const Env& env,
                            const std::filesystem::path& data_file_path,
                            FileOffsetType data_offset,
                            SafeInt<size_t> data_length,
                            Tensor& tensor) const override;

  // Check if GDS/cuFile is available on the current system.
  static bool IsGdsAvailable();

 private:
  int device_id_;
  bool driver_open_{false};
};

}  // namespace cuda
}  // namespace onnxruntime

#endif  // USE_GDS
