// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_allocator.h"
#include "cuda_common.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/session_state.h"
#include "cuda_fence.h"
#include "gpu_data_transfer.h"

namespace onnxruntime {

static const GPUDataTransfer* GetGPUDataTransfer(const SessionState* session_state) {
  OrtDevice gpu_device(OrtDevice::GPU, OrtDevice::MemType::DEFAULT, 0);
  OrtDevice cpu_device;
  return dynamic_cast<const GPUDataTransfer*>(session_state->GetDataTransferMgr().GetDataTransfer(gpu_device, cpu_device));
}

void CUDAAllocator::CheckDevice() const {
#ifndef NDEBUG
  // check device to match at debug build
  // if it's expected to change, call cudaSetDevice instead of the check
  int current_device;
  CUDA_CALL_THROW(cudaGetDevice(&current_device));
  ORT_ENFORCE(current_device == info_.id);
#endif
}

void* CUDAAllocator::Alloc(size_t size) {
  CheckDevice();
  void* p = nullptr;
  if (size > 0) {
    CUDA_CALL_THROW(cudaMalloc((void**)&p, size));
  }
  return p;
}

void CUDAAllocator::Free(void* p) {
  CheckDevice();
  cudaFree(p);  // do not throw error since it's OK for cudaFree to fail during shutdown
}

const OrtAllocatorInfo& CUDAAllocator::Info() const {
  return info_;
}

FencePtr CUDAAllocator::CreateFence(const SessionState* session_state) {
  return std::make_shared<CUDAFence>(GetGPUDataTransfer(session_state));
}

void* CUDAPinnedAllocator::Alloc(size_t size) {
  void* p = nullptr;
  if (size > 0) {
    CUDA_CALL_THROW(cudaMallocHost((void**)&p, size));
  }
  return p;
}

void CUDAPinnedAllocator::Free(void* p) {
  CUDA_CALL_THROW(cudaFreeHost(p));
}

const OrtAllocatorInfo& CUDAPinnedAllocator::Info() const {
  return info_;
}

FencePtr CUDAPinnedAllocator::CreateFence(const SessionState* session_state) {
  return std::make_shared<CUDAFence>(GetGPUDataTransfer(session_state));
}

void RegisterCudaAllocator(AllocatorManager& allocator_mgr, int device_id) {
  DeviceAllocatorRegistrationInfo default_allocator_info(
      {OrtMemTypeDefault, [](int id) { return std::make_unique<CUDAAllocator>(id, CUDA); }, std::numeric_limits<size_t>::max()});
  allocator_mgr.InsertAllocator(CreateAllocator(default_allocator_info, device_id));

  DeviceAllocatorRegistrationInfo pinned_allocator_info(
      {OrtMemTypeCPUOutput, [](int) { return std::make_unique<CUDAPinnedAllocator>(0, CUDA_PINNED); }, std::numeric_limits<size_t>::max()});
  allocator_mgr.InsertAllocator(CreateAllocator(pinned_allocator_info, device_id));

  // This is actually used for the cuda kernels which explicitly ask for inputs from CPU.
  DeviceAllocatorRegistrationInfo cpu_allocator_info({OrtMemTypeCPUInput, [](int) { return std::make_unique<CPUAllocator>(std::make_unique<OrtAllocatorInfo>("CUDA_CPU", OrtAllocatorType::OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeCPUInput)); }, std::numeric_limits<size_t>::max()});
  allocator_mgr.InsertAllocator(CreateAllocator(cpu_allocator_info));
}

}  // namespace onnxruntime
