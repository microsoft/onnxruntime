// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "js_kernel_lookup.h"

namespace onnxruntime {
namespace js {

const KernelCreateInfo* JsKernelLookup::LookUpKernel(const Node& node) const {
    // if (node.OpType() == "Clip") {
    //     node.
    // }

    return orig_.LookUpKernel(node);
}

}  // namespace js
}  // namespace onnxruntime
