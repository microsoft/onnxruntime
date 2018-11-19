// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_allocator.h"

ONNXRuntimeAllocatorInteface MockedONNXRuntimeAllocator::table_ = {
    {MockedONNXRuntimeAllocator::AddRef_, MockedONNXRuntimeAllocator::Release_}, MockedONNXRuntimeAllocator::Alloc_, MockedONNXRuntimeAllocator::Free_, MockedONNXRuntimeAllocator::Info_};
