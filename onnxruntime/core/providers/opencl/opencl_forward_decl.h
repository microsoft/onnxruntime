// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {

class OpenCLExecutionProvider;

namespace opencl {
class OpenCLDataTransfer;
class OpenCLKernelHolder;
class OpenCLProgramManager;
class OpenCLKernel;
class Image2DDesc;

template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

}  // namespace opencl
}  // namespace onnxruntime
