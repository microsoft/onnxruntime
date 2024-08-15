// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2019-2020, NXP Semiconductor, Inc. All rights reserved.
// SPDX-FileCopyrightText: Copyright 2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
// Licensed under the MIT License.

#include "acl_execution_provider.h"
#include "core/framework/allocator.h"
#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/compute_capability.h"
#include "core/providers/acl/math/matmul.h"
#include "core/providers/acl/nn/conv.h"
#include "core/session/inference_session.h"
#include "contrib_ops/cpu/cpu_contrib_kernels.h"
#include "acl_fwd.h"
#include "scheduler.h"

#include "arm_compute/runtime/Scheduler.h"
#include "arm_compute/runtime/PoolManager.h"
#include "arm_compute/runtime/BlobLifetimeManager.h"
#include "arm_compute/runtime/Allocator.h"

namespace onnxruntime {

namespace acl {

// Forward declarations of op kernels
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 6, Relu);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 8, Gemm);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 9, 10, Gemm);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 11, Gemm);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 11, Conv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 11, MLFloat16, Conv);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 8, 11, float, MaxPool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 12, MaxPool);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 8, float, GlobalAveragePool);
class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 8, float, GlobalMaxPool);

class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 11, AveragePool);

class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, BatchNormalization);
class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 4, 10, Concat);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 11, Concat);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, float, FusedConv);
class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, NhwcConv);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 13, MatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 13, MLFloat16, MatMul);

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, FusedMatMul);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, MLFloat16, FusedMatMul);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, uint8_t, MatMulIntegerToFloat);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, int8_t, MatMulIntegerToFloat);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 10, uint8_t, QLinearConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 10, int8_t, QLinearConv);

class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, uint8_t, QLinearConv);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, int8_t, QLinearConv);

Status RegisterACLKernels(KernelRegistry& kernel_registry) {
  static const BuildKernelCreateInfoFn function_table[] = {
      // BuildKernelCreateInfo<void>,  //default entry to avoid the list become empty after ops-reducing
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 6, Relu)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 8, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 9, 10, Gemm)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 11, Gemm)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 11, Conv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 11, MLFloat16, Conv)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 7, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 8, 11, float, MaxPool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 12, MaxPool)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 8, float, GlobalAveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 1, 8, float, GlobalMaxPool)>,

      // Opset 10
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 10, 10, float, AveragePool)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 11, AveragePool)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 7, 9, BatchNormalization)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 4, 10, Concat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 11, Concat)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, float, FusedConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, NhwcConv)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 13, MatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 13, MLFloat16, MatMul)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, FusedMatMul)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, MLFloat16, FusedMatMul)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, uint8_t, MatMulIntegerToFloat)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, int8_t, MatMulIntegerToFloat)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 10, uint8_t, QLinearConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kOnnxDomain, 10, int8_t, QLinearConv)>,

      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, uint8_t, QLinearConv)>,
      BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kAclExecutionProvider, kMSDomain, 1, int8_t, QLinearConv)>,
  };

  for (auto& function_table_entry : function_table) {
    KernelCreateInfo info = function_table_entry();
    if (info.kernel_def != nullptr) {  // filter disabled entries where type is void
      ORT_RETURN_IF_ERROR(kernel_registry.Register(std::move(info)));
    }
  }

  return Status::OK();
}

std::shared_ptr<KernelRegistry> GetAclKernelRegistry() {
  std::shared_ptr<KernelRegistry> kernel_registry = std::make_shared<KernelRegistry>();
  ORT_THROW_IF_ERROR(RegisterACLKernels(*kernel_registry));
  return kernel_registry;
}

std::shared_ptr<arm_compute::MemoryManagerOnDemand> ACLCreateMemoryManager() {
  auto lifetime_mgr = std::make_shared<arm_compute::BlobLifetimeManager>();
  auto pool_mgr = std::make_shared<arm_compute::PoolManager>();
  auto mm = std::make_shared<arm_compute::MemoryManagerOnDemand>(lifetime_mgr, pool_mgr);

  return mm;
}

}  // namespace acl

ACLExecutionProvider::ACLExecutionProvider(const ACLExecutionProviderInfo &info)
    : IExecutionProvider{onnxruntime::kAclExecutionProvider},
    info(info), memory_manager(onnxruntime::acl::ACLCreateMemoryManager()) {

  arm_compute::Scheduler::set(std::make_shared<acl::ORTScheduler>(this));
}

ACLExecutionProvider::~ACLExecutionProvider() {}

std::shared_ptr<KernelRegistry> ACLExecutionProvider::GetKernelRegistry() const {
  static std::shared_ptr<KernelRegistry> kernel_registry = onnxruntime::acl::GetAclKernelRegistry();
  return kernel_registry;
}

std::vector<std::unique_ptr<ComputeCapability>>
ACLExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                  const IKernelLookup& kernel_lookup) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (const auto& node : graph.Nodes()) {
    if (const KernelCreateInfo* kernel_create_info = kernel_lookup.LookUpKernel(node);
        kernel_create_info != nullptr) {

      Status support_status = Status::OK();
      const std::string op_name = kernel_create_info->kernel_def->OpName();

      if (op_name == "Conv" || op_name == "NhwcConv" || op_name == "QLinearConv") {
        support_status = onnxruntime::acl::ValidateConv(node);
      }
      if (op_name == "MatMul" || op_name == "FusedMatMul" || op_name == "MatMulIntegerToFloat") {
        support_status = onnxruntime::acl::ValidateMatMul(node);
      }

      if (support_status.IsOK()) {
        std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
        sub_graph->nodes.push_back(node.Index());
        result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
      } else {
        LOGS_DEFAULT(WARNING) << "ACL supports operator " << op_name
          << ", but not with these parameters. Using fallback for node: " << node.Name()
          << " Reason: " << support_status.ErrorMessage();
      }
    }
  }

  return result;
}

Status ACLExecutionProvider::OnRunStart(const onnxruntime::RunOptions&) {
  arm_compute::Allocator alloc{};
  memory_manager->populate(alloc, 1);
  return Status::OK();
};

Status ACLExecutionProvider::OnRunEnd(bool, const onnxruntime::RunOptions&) {
  memory_manager->clear();
  return Status::OK();
};

}  // namespace onnxruntime
