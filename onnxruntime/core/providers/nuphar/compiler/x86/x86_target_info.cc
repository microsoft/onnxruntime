// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/x86_target_info.h"
#include "core/common/common.h"

namespace onnxruntime {

const std::string CodeGenTargetX86::LLVM_TARGET_AVX = "llvm -mcpu=corei7-avx";
const std::string CodeGenTargetX86::LLVM_TARGET_AVX2 = "llvm -mcpu=core-avx2";
const std::string CodeGenTargetX86::LLVM_TARGET_AVX512 = "llvm -mcpu=skylake-avx512";

std::unique_ptr<CodeGenTarget> CodeGenTarget_AVX() {
  return std::make_unique<CodeGenTargetX86>(CodeGenTargetX86::LLVM_TARGET_AVX, 128, 2);
}

std::unique_ptr<CodeGenTarget> CodeGenTarget_AVX2() {
  return std::make_unique<CodeGenTargetX86>(CodeGenTargetX86::LLVM_TARGET_AVX2, 256, 2);
}

std::unique_ptr<CodeGenTarget> CodeGenTarget_AVX512() {
  return std::make_unique<CodeGenTargetX86>(CodeGenTargetX86::LLVM_TARGET_AVX512, 512, 2);
}

}  // namespace onnxruntime
