// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/x86_target_info.h"

namespace onnxruntime {

std::unique_ptr<CodeGenTarget> CodeGenTarget_AVX2() {
  return std::make_unique<CodeGenTargetX86>("llvm -mcpu=core-avx2", 256, 2);
}

std::unique_ptr<CodeGenTarget> CodeGenTarget_AVX512() {
  return std::make_unique<CodeGenTargetX86>("llvm -mcpu=skylake-avx512", 512, 2);
}

}  // namespace onnxruntime
