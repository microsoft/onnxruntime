// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <tvm/tvm.h>
#include <core/codegen/passes/scheduler/tvm_scheduler.h>

namespace onnxruntime {
namespace tvm_codegen {

// Check the schedule of tensor
// If it has no compute_root, Insert compute_root to tensor,
// and record it to ctx.scheduled_tensors
bool InsertRootSchedule(
    const tvm::Tensor& tensor,
    ScheduleContext& ctx);

// Check the schedule of tensor
// If it is not labeled as closure, lable it.
bool InsertClosure(
    const tvm::Tensor& tensor,
    ScheduleContext& ctx);

// Combination of InsertRootSchedule and InsertClosure
bool InsertRootScheduleAndClosure(
    const tvm::Tensor& tensor,
    ScheduleContext& ctx);

// Check precondition for vectorize schedule
bool ShouldTryVectorization(
    const tvm::Tensor& tensor,
    ScheduleContext& ctx);

// Check the schedule of tensor
// If it is not scheduled, try to vectorize it.
// Note TryVectorization has to use with compute_root.
// Therefore, there is a safty check of tensor's schedule
bool TryVectorization(
    const tvm::Tensor& tensor,
    int64_t natural_vector_size,
    ScheduleContext& ctx);

// Check the schedule of tensor
// If it is not scheduled, try to add compute_inline on it.
// Note TryInlineSchedule cannot be used with compute_root.
// Therefore, there is a safty check of tensor's schedule.
bool TryInlineSchedule(
    const tvm::Tensor& tensor,
    ScheduleContext& ctx);

// Check the schedule of tensor's inputs,
// and call InsertRootSchedule for each of them
bool InputRootSchedule(
    const tvm::Tensor& tensor,
    ScheduleContext& ctx);

// Check the schedule of tensor's inputs,
// and call InsertRootSchedule and TryVectorization for each of them
bool InputRootScheduleWithVectorization(
    const tvm::Tensor& tensor,
    int64_t natural_vector_size,
    ScheduleContext& ctx);

}  // namespace tvm_codegen
}  // namespace onnxruntime
