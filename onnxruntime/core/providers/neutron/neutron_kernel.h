// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#pragma once

#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/providers/neutron/neutron_allocator.h"
#include "core/providers/neutron/neutron_execution_provider.h"

namespace onnxruntime {
namespace neutron {

/*
  Remember: using AllocatorPtr = std::shared_ptr<IAllocator>;
*/

class NeutronKernel : public OpKernel {
 public:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NeutronKernel);

  explicit NeutronKernel(const OpKernelInfo& info)
      : OpKernel(info),
        provider_(const_cast<NeutronExecutionProvider*>(
            static_cast<const NeutronExecutionProvider*>(
                info.GetExecutionProvider()))) {
  }

 private:
  NeutronExecutionProvider* provider_;
};

}  // namespace neutron
}  // namespace onnxruntime
