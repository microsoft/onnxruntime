// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <vector>

#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {

class ORTInvoker {
 public:
  ORTInvoker(std::unique_ptr<IExecutionProvider> execution_provider);
  common::Status Invoke(const std::string& name,
                        //optional inputs / outputs?
                        const std::vector<Tensor>& inputs,
                        std::vector < std::unique_ptr<Tensor> > & outputs,
                        //attributes?
                        const std::string domain = common::kONNXDomain,
                        const int version = -1);
};

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}  // namespace onnxruntime
