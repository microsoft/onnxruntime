// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/allocator.h"
#include "core/framework/allocatormgr.h"
#include <cstdlib>
#include <sstream>

namespace onnxruntime {

void* CPUAllocator::Alloc(size_t size) {
  if (size <= 0)
    return nullptr;
  //todo: we should pin the memory in some case
  void* p = malloc(size);
  return p;
}

void CPUAllocator::Free(void* p) {
  //todo: unpin the memory
  free(p);
}

const ONNXRuntimeAllocatorInfo& CPUAllocator::Info() const {
  static constexpr ONNXRuntimeAllocatorInfo cpuAllocatorInfo(CPU, ONNXRuntimeAllocatorType::ONNXRuntimeDeviceAllocator);
  return cpuAllocatorInfo;
}
}  // namespace onnxruntime

std::ostream& operator<<(std::ostream& out, const ONNXRuntimeAllocatorInfo& info) {
  return (out << info.ToString());
}

ONNXRUNTIME_API_STATUS_IMPL(ONNXRuntimeCreateAllocatorInfo, const char* name1, ONNXRuntimeAllocatorType type, int id1, ONNXRuntimeMemType mem_type1, ONNXRuntimeAllocatorInfo** out) {
  *out = new ONNXRuntimeAllocatorInfo(name1, type, id1, mem_type1);
  return nullptr;
}

ONNXRUNTIME_API(void, ReleaseONNXRuntimeAllocatorInfo, ONNXRuntimeAllocatorInfo* p) {
  delete p;
}

ONNXRUNTIME_API(const char*, ONNXRuntimeAllocatorInfoGetName, _In_ ONNXRuntimeAllocatorInfo* ptr) {
  return ptr->name;
}

ONNXRUNTIME_API(int, ONNXRuntimeAllocatorInfoGetId, _In_ ONNXRuntimeAllocatorInfo* ptr) {
  return ptr->id;
}

ONNXRUNTIME_API(ONNXRuntimeMemType, ONNXRuntimeAllocatorInfoGetMemType, _In_ ONNXRuntimeAllocatorInfo* ptr) {
  return ptr->mem_type;
}

ONNXRUNTIME_API(ONNXRuntimeAllocatorType, ONNXRuntimeAllocatorInfoGetType, _In_ ONNXRuntimeAllocatorInfo* ptr) {
  return ptr->type;
}

ONNXRUNTIME_API(int, ONNXRuntimeCompareAllocatorInfo, _In_ const ONNXRuntimeAllocatorInfo* info1, _In_ const ONNXRuntimeAllocatorInfo* info2) {
  if (*info1 == *info2) {
    return 0;
  }
  return -1;
}