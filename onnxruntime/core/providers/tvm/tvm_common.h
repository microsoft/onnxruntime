// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef TVM_COMMON_H
#define TVM_COMMON_H

#include <vector>
#include <map>

#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/vm/vm.h>

namespace onnxruntime {
namespace tvm {

using TvmModule = ::tvm::runtime::Module;

}  // namespace tvm
}  // namespace onnxruntime

#endif  // TVM_COMMON_H
