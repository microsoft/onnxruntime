// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

namespace onnxruntime
{
constexpr const char* TRT = "Trt";

class TRTPinnedAllocator : public CPUAllocator
{
public:
    virtual const OrtAllocatorInfo& Info() const override
    {
        static OrtAllocatorInfo trt_cpu_allocator_info(TRT,
                OrtAllocatorType::OrtDeviceAllocator, 0,
                OrtMemType::OrtMemTypeCPU);
        return trt_cpu_allocator_info;
    }
};

class TRTAllocator : public CPUAllocator
{
public:
    virtual const OrtAllocatorInfo& Info() const override
    {
        static OrtAllocatorInfo trt_default_allocator_info(TRT,
                OrtAllocatorType::OrtDeviceAllocator, 0,
                OrtMemType::OrtMemTypeDefault);
        return trt_default_allocator_info;
    }
};
}  // namespace onnxruntime

