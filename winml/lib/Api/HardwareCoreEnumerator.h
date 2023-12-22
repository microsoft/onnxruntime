// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace WINMLP {
struct HardwareCoreEnumerator {
  HardwareCoreEnumerator() = delete;
  static uint32_t DefaultIntraOpNumThreads();
};
}  // namespace WINMLP
