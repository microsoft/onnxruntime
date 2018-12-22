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

const OrtAllocatorInfo& CPUAllocator::Info() const {
  static constexpr OrtAllocatorInfo cpuAllocatorInfo(CPU, OrtAllocatorType::OrtDeviceAllocator);
  return cpuAllocatorInfo;
}
}  // namespace onnxruntime

std::ostream& operator<<(std::ostream& out, const OrtAllocatorInfo& info) {
  return (out << info.ToString());
}

ORT_API_STATUS(OrtCreateAllocatorInfo, const char* name1, OrtAllocatorType type, int id1, OrtMemType mem_type1, OrtAllocatorInfo** out) {
  *out = new OrtAllocatorInfo(name1, type, id1, mem_type1);
  return nullptr;
}

ORT_API(void, OrtReleaseAllocatorInfo, OrtAllocatorInfo* p) {
  delete p;
}

ORT_API(const char*, OrtAllocatorInfoGetName, _In_ OrtAllocatorInfo* ptr) {
  return ptr->name;
}

ORT_API(int, OrtAllocatorInfoGetId, _In_ OrtAllocatorInfo* ptr) {
  return ptr->id;
}

ORT_API(OrtMemType, OrtAllocatorInfoGetMemType, _In_ OrtAllocatorInfo* ptr) {
  return ptr->mem_type;
}

ORT_API(OrtAllocatorType, OrtAllocatorInfoGetType, _In_ OrtAllocatorInfo* ptr) {
  return ptr->type;
}

ORT_API(int, OrtCompareAllocatorInfo, _In_ const OrtAllocatorInfo* info1, _In_ const OrtAllocatorInfo* info2) {
  if (*info1 == *info2) {
    return 0;
  }
  return -1;
}
