// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {

using ProviderType = const std::string&;

/*
  We use a simple fence mechanism for async compute. Assumptions in this fence mechanism:
  * Execution provider command queues, which execute in the same order of submit
  * No fence needed for kernels within one execution provider command queue
  * Fence is used to synchronize between command queues, and execution providers

  Fence usage:
  1. Fence object would be created by allocation planer for input/output when KernelDef::ExecQueueId() is not zero
  2. If fence object exists, executor would call BeforeUsingAs* prior to kernel::Compute(), and AfterUsedAs* afterwards
*/
class IFence {
 public:
  virtual ~IFence() = default;

  /**
     Called by executor before OrtValue is used as input in a compute kernel in provider_type and exec queue_id
     This should wait in the specified provider's exec queue for previous write to OrtValue to finish
  */
  virtual void BeforeUsingAsInput(onnxruntime::ProviderType provider_type, int queue_id) = 0;

  /**
     Called by executor before OrtValue is used as output in a compute kernel in provider_type and exec queue_id
     This should wait in the specified provider's exec queue for previous read to OrtValue to finish
  */
  virtual void BeforeUsingAsOutput(onnxruntime::ProviderType provider_type, int queue_id) = 0;

  /**
     Called by executor after OrtValue is used as input in a compute kernel in provider_type and exec queue_id
     This should update the read fence of the MLValue
  */
  virtual void AfterUsedAsInput(int queue_id) = 0;

  /**
     Called by executor after OrtValue is used as output in a compute kernel in provider_type and exec queue_id
     This should update the write fence of the MLValue
  */
  virtual void AfterUsedAsOutput(int queue_id) = 0;

  /**
  Called by executor before release OrtValue to see whether async data read is finished or not. This is non-blocking.
  */
  virtual bool CanRelease() = 0;
};
using Fence_t = IFence*;
using FencePtr = std::shared_ptr<IFence>;

}  // namespace onnxruntime
