// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"
#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
namespace webgpu {
namespace ep {

// Plugin device allocators with a real device. Both paths use GpuBufferAllocator.
// The Session column describes config.device_allocator, also used for kernel scratch.
// Outer classes implement the plugin's OrtAllocator ABI, before any additional Core wrappers.
// AllocatorPtr is std::shared_ptr<IAllocator>, not a separate implementation class.
//
// | Aspect         | Env shared allocator                               | Session device allocator                    |
// |----------------|----------------------------------------------------|---------------------------------------------|
// | Outer class    | onnxruntime::ep::adapter::Allocator : OrtAllocator | WebGpuSessionAllocator : OrtAllocator       |
// | Impl class     | GpuBufferAllocator (held by AllocatorPtr)          | GpuBufferAllocator (held by AllocatorPtr)   |
// | C API exposure | Factory::CreateAllocatorImpl                       | Ep::CreateAllocatorImpl wraps existing impl |
// | Impl creation  | Lazy on first Alloc                                | Once in Factory::CreateEpImpl               |
// | App API use    | CreateTensor/Alloc without Session                 | CreateTensor/Alloc via a Session allocator  |
// | Internal use   | Not used for EP kernel scratch                     | Run input/intermediate/output and scratch   |
// | Buffer manager | Context-shared BufferManager                       | Context-shared BufferManager                |
// | Capture route  | Still context-shared default                       | Session-owned per-graph manager during Run  |
// | Recording      | Getter retains independent state                   | Getter borrows the owning EP's Recording()  |
// | Lifetime       | Getter retains Context; no Session                 | EP must outlive allocator use/tensor frees  |
// | Alloc          | Submit cached clear before return                  | Submit cached clear, even during Run        |
// | AllocOnStream  | Not exposed by the C API wrapper                   | Matching Session stream: defer cached clear |
// |                |                                                    | Null stream: same policy as plain Alloc     |
//
// Per-graph routing applies when graph capture is enabled and the Run's graph annotation ID is not -1.
// Either can supply tensors to other Sessions on the same WebGPU device/context. A shared buffer
// cache does not imply a shared recording; callers must order writes before another Session uses them.
// Alloc vs AllocOnStream is a stream-based distinction, not an external-vs-internal API distinction:
// BindInput can allocate on a Session stream before Run; streamless allocation during Run uses Alloc.
// Read-only initializers and writable prepacked weights use separate GpuBufferAllocator instances
// with InitializerBufferManager(). Read-only initializers skip clears; prepack and native callers
// can supply different plain-Alloc submission policies.

OrtAllocator* CreateWebGpuSessionAllocator(AllocatorPtr allocator);
bool TryReleaseWebGpuSessionAllocator(OrtAllocator* allocator);

}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
