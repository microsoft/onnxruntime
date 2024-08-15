// Copyright(C) 2018 Intel Corporation
// Copyright (c) 2019-2020, NXP Semiconductor, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "core/providers/acl/acl_common.h"

namespace onnxruntime {
namespace acl {

void PopulateWorkspace(const arm_compute::experimental::MemoryRequirements &reqs,
    Workspace &workspace, arm_compute::MemoryGroup &memory_group,
    arm_compute::ITensorPack &run_pack, arm_compute::ITensorPack &prep_pack) {

  for (const arm_compute::experimental::MemoryInfo &req : reqs) {
    if (req.size == 0) {
      continue;
    }

    arm_compute::Tensor *aux_tensor;
    if (req.lifetime == arm_compute::experimental::MemoryLifetime::Temporary) {
      workspace.temporary_tensors.emplace_back(std::make_unique<arm_compute::Tensor>());
      aux_tensor = workspace.temporary_tensors.back().get();

      memory_group.manage(aux_tensor);
    } else if (req.lifetime == arm_compute::experimental::MemoryLifetime::Prepare) {
      workspace.prepare_tensors.emplace_back(std::make_unique<arm_compute::Tensor>());
      aux_tensor = workspace.prepare_tensors.back().get();

      prep_pack.add_tensor(req.slot, aux_tensor);
    } else {
      workspace.persistent_tensors.emplace_back(std::make_unique<arm_compute::Tensor>());
      aux_tensor = workspace.persistent_tensors.back().get();

      prep_pack.add_tensor(req.slot, aux_tensor);
    }
    run_pack.add_tensor(req.slot, aux_tensor);

    const auto aux_info = arm_compute::TensorInfo{arm_compute::TensorShape(req.size), 1, arm_compute::DataType::U8};
    aux_tensor->allocator()->init(aux_info, req.alignment);
  }

  for (const std::unique_ptr<arm_compute::Tensor> &tensor : workspace.temporary_tensors) {
    tensor->allocator()->allocate();
  }
}

arm_compute::TensorShape ACLTensorShape(const TensorShape& tensorShape, unsigned int extDim) {
  arm_compute::TensorShape shape;
  unsigned int inDim = tensorShape.NumDimensions();
  unsigned int outDim = (extDim > inDim) ? extDim : inDim;

  // arm_compute tensors are in reversed order (width, height, channels, batch).
  for (unsigned int i = 0; i < inDim; i++)
    shape.set(outDim - i - 1, tensorShape.GetDims()[i], false);

  // extend dimensions
  for (unsigned int i = 0; i < outDim - inDim; i++)
    shape.set(i, 1);

  // prevent arm_compute issue where tensor is flattened to nothing
  if (shape.num_dimensions() == 0)
    shape.set_num_dimensions(1);

  return shape;
}

Status GetArgShape(const NodeArg *tensor, TensorShape& outShape)
{
  const auto& inShape = tensor->Shape();
  TensorShapeVector shapeVec;

  for (int i = 0; i < inShape->dim_size(); i++) {
    const auto& dim = inShape->dim(i);
    ORT_RETURN_IF_NOT(dim.has_dim_value(), "ACL does not support unknown tensor shapes: ", tensor->Name());
    shapeVec.push_back(dim.dim_value());
  }

  outShape = TensorShape(shapeVec);
  return Status::OK();
}

void ACLPrintTensorShape(const char* s, arm_compute::Tensor& t) {
  for (unsigned int i = 0; i < t.info()->tensor_shape().num_dimensions(); i++)
    LOGS_DEFAULT(VERBOSE) << "ACL " << s << " " << t.info()->tensor_shape()[i];
  LOGS_DEFAULT(VERBOSE) << std::endl;
}

arm_compute::DataType ACLDataType(const std::string &dtype) {
  if (dtype == "tensor(float)") {
    return arm_compute::DataType::F32;
  }
  if (dtype == "tensor(float16)") {
    return arm_compute::DataType::F16;
  }
  if (dtype == "tensor(bfloat16)") {
    return arm_compute::DataType::BFLOAT16;
  }
  if (dtype == "tensor(uint8)") {
    return arm_compute::DataType::QASYMM8;
  }
  if (dtype == "tensor(int8)") {
    return arm_compute::DataType::QASYMM8_SIGNED;
  }
  if (dtype == "tensor(int32)") {
    return arm_compute::DataType::S32;
  }
  ORT_THROW("ACL execution provider does not support data type ", dtype);
}

int GetIntScalar(const Tensor *tensor) {
  ORT_ENFORCE(tensor->Shape().Size() == 1, "Tensor is not a scalar");
  if (tensor->IsDataType<uint8_t>()) {
    return *tensor->Data<uint8_t>();
  }
  if (tensor->IsDataType<int8_t>()) {
    return *tensor->Data<int8_t>();
  }
  ORT_THROW("Unsupported int type: ", DataTypeImpl::ToString(tensor->DataType()));
}

Status LoadQuantizationInfo(const OpKernelInfo& info, arm_compute::Tensor* tensor,
    const int scaleIdx, const int zpIdx, bool flipZeroPoint) {

  const Tensor *scaleTensor = nullptr;
  ORT_RETURN_IF_NOT(info.TryGetConstantInput(scaleIdx, &scaleTensor), "Scale must be constant");

  const Tensor *zeroPointTensor = nullptr;
  ORT_RETURN_IF_NOT(info.TryGetConstantInput(zpIdx, &zeroPointTensor), "Zero point must be constant");

  const float *scale = scaleTensor->Data<float>();
  const int zeroPoint = GetIntScalar(zeroPointTensor);
  tensor->info()->set_quantization_info(arm_compute::QuantizationInfo(*scale, flipZeroPoint? -zeroPoint : zeroPoint));

  return Status::OK();
}

void GetPackingInfo(std::vector<std::unique_ptr<arm_compute::Tensor>>& state, size_t& packedSize, size_t& alignment) {
  alignment = 0;
  for (auto &tensor : state) {
    alignment = std::max(alignment, tensor->allocator()->alignment());
  }

  packedSize = 0;
  for (auto &tensor : state) {
    const size_t size = tensor->info()->total_size();
    packedSize += ((size - 1) / alignment + 1) * alignment;
  }
}

Status LoadPackedTensors(std::vector<std::unique_ptr<arm_compute::Tensor>>& state, void* packed,
        const size_t packedSize, const size_t alignment) {

  auto buffSize = packedSize + alignment;
  uint8_t *alignedPtr = (uint8_t *) (alignment == 0? packed : std::align(alignment, packedSize, packed, buffSize));

  uint8_t *currentPtr = alignedPtr;
  for (auto &tensor : state) {
    ORT_RETURN_IF_ERROR(ACLImportMemory(tensor->allocator(), currentPtr, 0));

    const size_t size = tensor->info()->total_size();
    currentPtr += ((size - 1) / alignment + 1) * alignment;
  }

  return Status::OK();
}

Status ACLImportMemory(arm_compute::TensorAllocator* allocator, void* memory, size_t size) {
  ORT_UNUSED_PARAMETER(size);
  arm_compute::Status status = allocator->import_memory(memory);

  if (status) {
    return Status::OK();
  } else {
    return Status(common::ONNXRUNTIME, common::FAIL, status.error_description());
  }
}

template <typename T>
void importDataToTensor(arm_compute::Tensor* tensor, const T* data) {
  arm_compute::Window aclInpuWindow;
  aclInpuWindow.use_tensor_dimensions(tensor->info()->tensor_shape());

  arm_compute::Iterator aclInputIt(tensor, aclInpuWindow);
  int index = 0;

  // copy input tensor into the larger buffer
  arm_compute::execute_window_loop(
      aclInpuWindow,
      [&](const arm_compute::Coordinates& co) {
        *reinterpret_cast<T*>(aclInputIt.ptr()) = data[index];
        index++;
      },
      aclInputIt);
}
template void importDataToTensor<float>(arm_compute::Tensor*, const float*);
template void importDataToTensor<MLFloat16>(arm_compute::Tensor*, const MLFloat16*);

template <typename T>
void importDataFromTensor(arm_compute::Tensor* tensor, T* data) {
  arm_compute::Window aclInpuWindow;
  aclInpuWindow.use_tensor_dimensions(tensor->info()->tensor_shape());

  arm_compute::Iterator aclInputIt(tensor, aclInpuWindow);
  int index = 0;
  // copy input tensor into the larger buffer
  arm_compute::execute_window_loop(
      aclInpuWindow,
      [&](const arm_compute::Coordinates& co) {
        data[index] = *reinterpret_cast<T*>(aclInputIt.ptr());
        index++;
      },
      aclInputIt);
}
template void importDataFromTensor<float>(arm_compute::Tensor*, float*);
template void importDataFromTensor<MLFloat16>(arm_compute::Tensor*, MLFloat16*);

}  // namespace acl
}  // namespace onnxruntime
