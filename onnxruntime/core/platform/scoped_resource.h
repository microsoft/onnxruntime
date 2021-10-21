// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"

namespace onnxruntime {

/**
 * Utility class to manage the cleanup of an arbitrary resource handle.
 * This is similar to std::unique_ptr<T> but it allows for invalid handle values that are not nullptr.
 *
 * The resource is specified by the TResourceTraits type.
 * The following definitions should be present and have the specified semantics:
 * - TResourceTraits::Handle
 *   - resource handle type, should be copyable
 * - static TResourceTraits::Handle TResourceTraits::GetInvalidHandleValue() noexcept
 *   - function returning an invalid handle value
 * - static void TResourceTraits::CleanUp(TResourceTraits::Handle handle) noexcept
 *   - function that performs the resource clean up
 */
template <typename TResourceTraits>
class ScopedResource {
 public:
  using Handle = typename TResourceTraits::Handle;
  using Traits = TResourceTraits;

  explicit ScopedResource(Handle handle = Traits::GetInvalidHandleValue()) noexcept
      : handle_{handle} {}

  ScopedResource(ScopedResource&& other) noexcept
      : handle_{other.Release()} {}

  ScopedResource& operator=(ScopedResource&& other) noexcept {
    Reset(other.Release());
    return *this;
  }

  ~ScopedResource() noexcept {
    Reset();
  }

  Handle Get() const noexcept {
    return handle_;
  }

  bool IsValid() const noexcept {
    return handle_ != Traits::GetInvalidHandleValue();
  }

  explicit operator bool() const noexcept {
    return IsValid();
  }

  // Cleans up the currently held resource, if any, and sets the held resource to a new value
  void Reset(Handle new_handle = Traits::GetInvalidHandleValue()) noexcept {
    if (IsValid()) {
      Traits::CleanUp(handle_);
    }
    handle_ = new_handle;
  }

  // Releases the held resource without cleaning it up
  Handle Release() noexcept {
    Handle result = handle_;
    handle_ = Traits::GetInvalidHandleValue();
    return result;
  }

 private:
  ORT_DISALLOW_COPY_AND_ASSIGNMENT(ScopedResource);

  Handle handle_;
};

}  // namespace onnxruntime
