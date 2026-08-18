// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <wrl/client.h>
#include <new>
#include <utility>

// Drop-in replacement for wil::MakeOrThrow that avoids an ASan false positive.
// WRL's MakeAllocator stores its buffer as char*, so if the constructor throws,
// ~MakeAllocator calls delete on a char* — passing sizeof(char)=1 to sized
// operator delete instead of sizeof(T). With the default MSVC allocator, this is
// benign (sized delete ignores the size), but ASan flags it as
// new-delete-type-mismatch. This helper uses placement new with correctly-sized
// cleanup to avoid the issue.
namespace Dml
{
    template <typename T, typename... TArgs>
    Microsoft::WRL::ComPtr<T> SafeMakeOrThrow(TArgs&&... args)
    {
        void* buffer = ::operator new(sizeof(T));
        T* raw = nullptr;
        try
        {
            raw = new (buffer) T(std::forward<TArgs>(args)...);
        }
        catch (...)
        {
            ::operator delete(buffer, sizeof(T));
            throw;
        }
        Microsoft::WRL::ComPtr<T> result;
        result.Attach(raw);
        return result;
    }
} // namespace Dml
