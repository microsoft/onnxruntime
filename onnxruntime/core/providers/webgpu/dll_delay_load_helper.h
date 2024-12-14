// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace webgpu {

// The DLL delay load helper is a RAII style guard to ensure DLL loading is done correctly.
//
// - On Windows, the helper sets the DLL search path to the directory of the current DLL.
// - On other platforms, the helper does nothing.
//
struct DllDelayLoadHelper final {
  DllDelayLoadHelper();
  ~DllDelayLoadHelper();
};

}  // namespace webgpu
}  // namespace onnxruntime
