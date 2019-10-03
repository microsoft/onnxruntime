// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/compiler/x86/x86_target_info.h"
#include "core/common/common.h"

namespace onnxruntime {

const std::string CodeGenTargetX86::LLVM_TARGET_AVX = "llvm -mcpu=corei7-avx";
const std::string CodeGenTargetX86::LLVM_TARGET_AVX2 = "llvm -mcpu=core-avx2";
const std::string CodeGenTargetX86::LLVM_TARGET_AVX512 = "llvm -mcpu=skylake-avx512";

std::unique_ptr<CodeGenTarget> CodeGenTarget_AVX() {
  // Make it conservative by enabling basic avx support for sandy-bridge cpus
  // Also note that AVX-1 has 256-bit vector support only for float. For int,
  // AVX-1 still uses 128-vector. Let's be conservative as well by choosing
  // 128 as max_vector_bits. We can change the value to 256 if later we think
  // 256 is more appropriate. Moreover, the value only affects performance for avx-1,
  // which we care about less. I couldn't find the point in Intel's reference
  // where the number of vector units is mentioned explicitly, but got it
  // from some forum by google search.
  // TODO: we should refine vector_unit_num if it turned out not to be 2.
  // The current number is based on:
  // https://en.wikipedia.org/wiki/Sandy_Bridge
  // "Improved 3 integer ALU, 2 vector ALU and 2 AGU per core"
  // and
  // https://indico.cern.ch/event/625333/contributions/2587012/attachments/1492809/2330113/VectorParallelism.pdf
  // "Example: Intel Xeon E5-2670 v2 “Ivy Bridge” –10 cores, each with 256-bit AVX vector unit "
  return onnxruntime::make_unique<CodeGenTargetX86>(CodeGenTargetX86::LLVM_TARGET_AVX, 128, 2);
}

std::unique_ptr<CodeGenTarget> CodeGenTarget_AVX2() {
  return onnxruntime::make_unique<CodeGenTargetX86>(CodeGenTargetX86::LLVM_TARGET_AVX2, 256, 2);
}

std::unique_ptr<CodeGenTarget> CodeGenTarget_AVX512() {
  return onnxruntime::make_unique<CodeGenTargetX86>(CodeGenTargetX86::LLVM_TARGET_AVX512, 512, 2);
}

}  // namespace onnxruntime
