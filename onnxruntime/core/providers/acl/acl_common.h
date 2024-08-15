// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019-2020, NXP Semiconductor, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

// ACL
#include "arm_compute/core/experimental/Types.h"
#include "arm_compute/runtime/Tensor.h"
#include "arm_compute/runtime/TensorAllocator.h"

namespace onnxruntime {
namespace acl {

struct Workspace {
  std::vector<std::unique_ptr<arm_compute::Tensor>> temporary_tensors;
  std::vector<std::unique_ptr<arm_compute::Tensor>> prepare_tensors;
  std::vector<std::unique_ptr<arm_compute::Tensor>> persistent_tensors;
};

void PopulateWorkspace(const arm_compute::experimental::MemoryRequirements &reqs,
    Workspace &workspace, arm_compute::MemoryGroup &memory_group,
    arm_compute::ITensorPack &run_pack, arm_compute::ITensorPack &prep_pack);

arm_compute::TensorShape ACLTensorShape(const TensorShape& tensorShape, unsigned int extDim = 0);
Status GetArgShape(const NodeArg *tensor, TensorShape& outShape);
void ACLPrintTensorShape(const char*, arm_compute::Tensor& t);
arm_compute::DataType ACLDataType(const std::string &dtype);

int GetIntScalar(const Tensor *tensor);
Status LoadQuantizationInfo(const OpKernelInfo& info, arm_compute::Tensor* tensor,
    const int scaleIdx, const int zpIdx, bool flipZeroPoint);

void GetPackingInfo(std::vector<std::unique_ptr<arm_compute::Tensor>>& state, size_t& packedSize, size_t& alignment);
Status LoadPackedTensors(std::vector<std::unique_ptr<arm_compute::Tensor>>& state, void* packed,
        const size_t packedSize, const size_t alignment);

Status ACLImportMemory(arm_compute::TensorAllocator* allocator, void* memory, size_t size);
template <typename T>
void importDataToTensor(arm_compute::Tensor* tensor, const T* data);
template <typename T>
void importDataFromTensor(arm_compute::Tensor* tensor, T* data);

}  // namespace acl
}  // namespace onnxruntime
