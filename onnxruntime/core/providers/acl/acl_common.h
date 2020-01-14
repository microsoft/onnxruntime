// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

// ACL
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"
#include "arm_compute/runtime/MemoryManagerOnDemand.h"

namespace onnxruntime {
namespace acl {

arm_compute::TensorShape ACLTensorShape(const TensorShape& tensorShape, unsigned int extDim = 0);
void ACLPrintTensorShape(const char*, arm_compute::Tensor& t);
std::shared_ptr<arm_compute::MemoryManagerOnDemand> ACLCreateMemoryManager();
arm_compute::Status ACLImportMemory(arm_compute::TensorAllocator* allocator, void* memory, size_t size);

}  // namespace acl
}  // namespace onnxruntime
