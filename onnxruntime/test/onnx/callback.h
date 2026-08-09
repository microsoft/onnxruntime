// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace onnxruntime {
namespace test {
struct OrtCallback {
  void (*f)(void* param) noexcept = nullptr;
  void* param = nullptr;
  void Run() noexcept;
};

}  // namespace test
}  // namespace onnxruntime
