// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once


#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "gsl/pointers"

namespace onnxruntime {

using DeviceAllocatorFactory = std::function<std::unique_ptr<IDeviceAllocator>(int)>;

struct DeviceAllocatorRegistrationInfo {
  OrtMemType mem_type;
  DeviceAllocatorFactory factory;
  size_t max_mem;
};

AllocatorPtr CreateAllocator(DeviceAllocatorRegistrationInfo info, int device_id = 0);

using AllocatorMap = std::map<int, AllocatorPtr>;

class AllocatorManager {
 public:
  /**
     Get all IAllocators for <*this> execution provider.
  */
  const std::vector<gsl::not_null<const IAllocator*>>& GetAllocators() const;

  /**
   * Get an allocator with specified device id and MemType. Return nullptr if it doesn't exist
   */
  AllocatorPtr GetAllocator(const OrtDevice& device) const;

  OrtAllocatorInfo GetDefaultCpuAllocatorInfo() const {
    return GetAllocator(OrtDevice())->Info();
  }

  void InsertAllocator(AllocatorPtr allocator);

private:

  AllocatorMap allocators_;

  // convenience list of the allocators so GetAllocatorList doesn't have to build a new vector each time
  // contains the same instances as allocators_
  std::vector<gsl::not_null<const IAllocator*>> allocator_list_;
};

}  // namespace onnxruntime
