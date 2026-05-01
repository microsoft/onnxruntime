// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Shadow stub for <cub/device/device_transform.cuh>. Resolved first via -I,
// ahead of the -isystem CUDA toolkit path.
//
// CUB 3.2.0 (CUDA 13.2) ships an invalid template specialisation:
//   struct ::cuda::proclaims_copyable_arguments<...> : ::cuda::std::true_type {};
// A globally-qualified class name in a specialisation is rejected by the compiler.
// We re-emit the parts Thrust needs internally with the fixed syntax (the
// specialisation is written inside the cuda namespace so the name is unqualified).
// cub::DeviceTransform itself is not used by ORT and is intentionally omitted.

#pragma once

#include <cub/version.cuh>

#if CUB_VERSION >= 300200

#include <cub/device/dispatch/dispatch_transform.cuh>  // cub::detail::transform::dispatch_t (Thrust)
#include <cuda/__functional/address_stability.h>       // cuda::proclaims_copyable_arguments primary

CUB_NAMESPACE_BEGIN
namespace detail
{
template <typename T>
struct __return_constant
{
  T value;
  template <typename... Args>
  _CCCL_HOST_DEVICE T operator()(Args&&...) const { return value; }
};
} // namespace detail
CUB_NAMESPACE_END

_CCCL_BEGIN_NAMESPACE_CUDA
template <typename T>
struct proclaims_copyable_arguments<CUB_NS_QUALIFIER::detail::__return_constant<T>>
    : ::cuda::std::true_type {};
_CCCL_END_NAMESPACE_CUDA

#endif  // CUB_VERSION >= 300200
