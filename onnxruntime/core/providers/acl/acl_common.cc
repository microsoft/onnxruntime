// Copyright(C) 2018 Intel Corporation
// Copyright (c) 2019, NXP Semiconductor, Inc. All rights reserved.
// Licensed under the MIT License

#ifdef _WIN32
#pragma warning(disable : 4244)
#endif

#include "core/providers/acl/acl_common.h"

#include "arm_compute/runtime/PoolManager.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"

#undef ACL_1902

namespace onnxruntime {
namespace acl {

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

void ACLPrintTensorShape(const char* s, arm_compute::Tensor& t) {
  for (unsigned int i = 0; i < t.info()->tensor_shape().num_dimensions(); i++)
    LOGS_DEFAULT(VERBOSE) << "ACL " << s << " " << t.info()->tensor_shape()[i];
  LOGS_DEFAULT(VERBOSE) << std::endl;
}

std::shared_ptr<arm_compute::MemoryManagerOnDemand> ACLCreateMemoryManager() {
  auto lifetime_mgr = std::make_shared<arm_compute::BlobLifetimeManager>();
  auto pool_mgr = std::make_shared<arm_compute::PoolManager>();
  auto mm = std::make_shared<arm_compute::MemoryManagerOnDemand>(lifetime_mgr, pool_mgr);

  return mm;
}

arm_compute::Status ACLImportMemory(arm_compute::TensorAllocator* allocator, void* memory, size_t size) {
#ifdef ACL_1902
  return allocator->import_memory(memory, size);
#else
  ORT_UNUSED_PARAMETER(size);
  return allocator->import_memory(memory);
#endif
}

}  // namespace acl
}  // namespace onnxruntime
