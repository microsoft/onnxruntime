// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_pch.h"
#include "core/platform/ort_mutex.h"
#include "core/graph/constants.h"
#include "core/framework/allocatormgr.h"
#include "core/framework/execution_provider.h"
#include "shared_inc/cuda_utils.h"
#include <deque>

#include "cuda_common.h"
#include "cuda_execution_provider.h"
#include "core/framework/memcpy.h"
#include "cuda_fence.h"
#include "cuda_allocator.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "contrib_ops/contrib_kernels.h"

namespace onnxruntime {

Status CopyTensorFromCudaToCudaPinned(const void* src_data, void* dst_data, size_t bytes, int exec_queue_id);
Status CopyTensorFromCudaToCpu(const void* src_data, void* dst_data, size_t bytes, int exec_queue_id);
Status CopyTensorFromCudaPinnedToCuda(const void* src_data, void* dst_data, size_t bytes, int exec_queue_id);
Status CopyTensorFromCudaToCuda(const void* src_data, void* dst_data, size_t bytes, int exec_queue_id);
Status CopyTensorFromCpuToCuda(const void* src_data, void* dst_data, size_t bytes, int exec_queue_id);

}  // namespace onnxruntime
