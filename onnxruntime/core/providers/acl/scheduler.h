// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// SPDX-License-Identifier: MIT

#include "acl_execution_provider.h"

#include "arm_compute/runtime/IScheduler.h"
#include "arm_compute/core/CPP/ICPPKernel.h"

namespace onnxruntime {
namespace acl {

class ORTScheduler: public arm_compute::IScheduler {
  public:
    ORTScheduler(ACLExecutionProvider *provider) : _provider(provider) {
    }

    void set_num_threads(unsigned int num_threads) override;

    unsigned int num_threads() const override;

    void schedule(arm_compute::ICPPKernel *kernel, const Hints &hints) override;

    void schedule_op(arm_compute::ICPPKernel *kernel, const Hints &hints,
        const arm_compute::Window &window, arm_compute::ITensorPack &tensors) override;

    void run_workloads(std::vector<Workload> &workloads) override;

  private:
    ACLExecutionProvider *_provider = nullptr;
};

}  // namespace acl
}  // namespace onnxruntime
