// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "core/common/common.h"
#include "scheduler.h"


using onnxruntime::concurrency::ThreadPool;

namespace onnxruntime {
namespace acl {

void ORTScheduler::set_num_threads(unsigned int num_threads) {
    ORT_THROW("Not supported");
}

unsigned int ORTScheduler::num_threads() const {
    // We can't check the size of the thread pool during kernel initialization,
    // as required by ACL. Therefore we have to choose a fixed thread count and
    // let some cores run multiple workloads if there are fewer than 32 cores.
    // This doesn't seem to cause performance issues with fewer cores in practice.
    return 32;
}

void ORTScheduler::schedule(arm_compute::ICPPKernel *kernel, const Hints &hints) {
    arm_compute::ITensorPack tensors;
    schedule_op(kernel, hints, kernel->window(), tensors);
}

void ORTScheduler::schedule_op(arm_compute::ICPPKernel *kernel, const Hints &hints,
        const arm_compute::Window &window, arm_compute::ITensorPack &tensors) {
    schedule_common(kernel, hints, window, tensors);
}

void ORTScheduler::run_workloads(std::vector<Workload> &workloads) {
    ThreadPool::TrySimpleParallelFor(_provider->GetThreadPool(), workloads.size(),
        [&](std::ptrdiff_t id) {
            const arm_compute::ThreadInfo info {
                (int) id, (int) workloads.size(), &cpu_info()
            };
            workloads[id](info);
        }
    );
}

}  // namespace acl
}  // namespace onnxruntime
