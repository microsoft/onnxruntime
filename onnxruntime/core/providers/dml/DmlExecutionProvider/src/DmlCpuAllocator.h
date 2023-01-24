// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

namespace Dml
{

class DmlCpuAllocator : public onnxruntime::IAllocator
{
public:
    explicit DmlCpuAllocator(OrtMemType memType);

    void* Alloc(size_t size) override;
    void Free(void* p) override;
};

} // namespace Dml
