// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_allocator.h"

OrtAllocatorInterface MockedOrtAllocator::table_ = {
    {MockedOrtAllocator::AddRef_, MockedOrtAllocator::Release_}, MockedOrtAllocator::Alloc_, MockedOrtAllocator::Free_, MockedOrtAllocator::Info_};
