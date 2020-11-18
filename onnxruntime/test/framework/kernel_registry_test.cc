// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <core/framework/kernel_registry.h>
#include <core/framework/op_kernel.h>
#include "core/framework/execution_providers.h"
#include "core/providers/cpu/cpu_execution_provider.h"
#include <core/framework/kernel_registry_manager.h>
#include "asserts.h"

using namespace onnxruntime;
static Status RegKernels(KernelRegistry& r, std::vector<std::unique_ptr<KernelDef> >& function_table, const KernelCreateFn& kernel_creator) {
  for (auto& function_table_entry : function_table) {
    ORT_RETURN_IF_ERROR(r.Register(KernelCreateInfo(std::move(function_table_entry), kernel_creator)));
  }
  return Status::OK();
}

class FakeKernel final : public OpKernel {
 public:
  FakeKernel(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext*) const override {
    return Status::OK();
  }
};

OpKernel* CreateFakeKernel(const OpKernelInfo& info) {
  return new FakeKernel(info);
}

// Test whether all onnx ops have kernel registrations for CPU EP. This is
// important because ORT promises compliance with onnx opsets.
TEST(KernelRegistryTests, kernels_registered_for_all_onnx_ops) {
  auto cpu_execution_provider = onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo(false));
  const auto registry = cpu_execution_provider->GetKernelRegistry();
  const auto& kernel_creator_fn_map = registry->GetKernelCreateMap();

  auto last_released_onnx_version = ONNX_NAMESPACE::OpSchemaRegistry::DomainToVersionRange().LastReleaseVersionMap().at(ONNX_NAMESPACE::ONNX_DOMAIN);
  const std::vector<ONNX_NAMESPACE::OpSchema> schemas = ONNX_NAMESPACE::OpSchemaRegistry::get_all_schemas();

  std::vector<std::string> expected_not_registered_ops = {"Hardmax",
                                                          "Loop",
                                                          "If",
                                                          "Constant",
                                                          "MemcpyToHost",
                                                          "MemcpyFromHost"};

  bool missing_kernel_registrations_for_op = false;
  std::string err_string = "Kernel registrations missing for following operators ";
  auto total_schemas = schemas.size();
  for (const auto& schema : schemas) {
    const std::string& op_type = schema.Name();
    const std::string& domain = schema.domain().empty() ? kOnnxDomainAlias : schema.domain();
    int op_since_version = schema.SinceVersion();

    // Only check for onnx domain, valid (non deprecated and not experimental) ops.
    if (domain != kOnnxDomainAlias || schema.Deprecated() ||
        schema.support_level() == ONNX_NAMESPACE::OpSchema::SupportType::EXPERIMENTAL) {
      total_schemas--;
      continue;
    }

    std::string key(op_type);
    key.append(1, ' ').append(domain).append(1, ' ').append(kCpuExecutionProvider);
    auto range = kernel_creator_fn_map.equal_range(key);
    bool valid_version = false;
    // walk through all registered versions for this op
    // untill we find the one which matches the version.
    for (auto i = range.first; i != range.second; ++i) {
      auto kernel_def = *i->second.kernel_def;
      int kernel_start_version;
      int kernel_end_version;
      kernel_def.SinceVersion(&kernel_start_version, &kernel_end_version);

      valid_version = kernel_start_version == op_since_version ||
                      (kernel_start_version < op_since_version && kernel_end_version != INT_MAX && kernel_end_version >= op_since_version);
      if (valid_version) {
        break;
      }
    }
    if (!valid_version) {
      if (!schema.HasFunction() && !schema.HasContextDependentFunction() &&
          schema.SinceVersion() <= last_released_onnx_version) {
        if (std::find(expected_not_registered_ops.begin(), expected_not_registered_ops.end(), op_type) == expected_not_registered_ops.end()) {
          missing_kernel_registrations_for_op = true;
          err_string.append(op_type).append(" version ").append(std::to_string(op_since_version).append(1, ' '));
        }
      }
    }
  }

  ASSERT_FALSE(missing_kernel_registrations_for_op) << err_string;
}

TEST(KernelRegistryTests, simple) {
  KernelRegistry r;
  std::vector<std::unique_ptr<KernelDef> > function_table;
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6).Provider(kCpuExecutionProvider).Build());

  Status st;
  ASSERT_TRUE((st = RegKernels(r, function_table, CreateFakeKernel)).IsOK()) << st.ErrorMessage();
}

TEST(KernelRegistryTests, dup_simple) {
  KernelRegistry r;
  std::vector<std::unique_ptr<KernelDef> > function_table;
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  Status st;
  ASSERT_FALSE((st = RegKernels(r, function_table, CreateFakeKernel)).IsOK()) << st.ErrorMessage();
}

//duplicated registration. One in default("") domain, another in "ai.onnx" domain
TEST(KernelRegistryTests, dup_simple2) {
  KernelRegistry r;
  std::vector<std::unique_ptr<KernelDef> > function_table;
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("ai.onnx").SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  Status st;
  ASSERT_FALSE((st = RegKernels(r, function_table, CreateFakeKernel)).IsOK()) << st.ErrorMessage();
}

//One in default("") domain, another in ms domain. Should be ok
TEST(KernelRegistryTests, one_op_name_in_two_domains) {
  KernelRegistry r;
  std::vector<std::unique_ptr<KernelDef> > function_table;
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain(kMSDomain).SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  Status st;
  ASSERT_TRUE((st = RegKernels(r, function_table, CreateFakeKernel)).IsOK()) << st.ErrorMessage();
}

//One op two versions
TEST(KernelRegistryTests, two_versions) {
  KernelRegistry r;
  std::vector<std::unique_ptr<KernelDef> > function_table;
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(1, 5).Provider(kCpuExecutionProvider).Build());
  Status st;
  ASSERT_TRUE((st = RegKernels(r, function_table, CreateFakeKernel)).IsOK()) << st.ErrorMessage();
}

//One op two versions
TEST(KernelRegistryTests, two_versions2) {
  KernelRegistry r;
  std::vector<std::unique_ptr<KernelDef> > function_table;
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(1, 6).Provider(kCpuExecutionProvider).Build());
  Status st;
  ASSERT_FALSE((st = RegKernels(r, function_table, CreateFakeKernel)).IsOK());
}

//One op two versions
TEST(KernelRegistryTests, two_versions3) {
  KernelRegistry r;
  std::vector<std::unique_ptr<KernelDef> > function_table;
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(1).Provider(kCpuExecutionProvider).Build());
  Status st;
  ASSERT_TRUE((st = RegKernels(r, function_table, CreateFakeKernel)).IsOK()) << st.ErrorMessage();
}

//One op two versions
TEST(KernelRegistryTests, two_versions4) {
  KernelRegistry r;
  std::vector<std::unique_ptr<KernelDef> > function_table;
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(5, 6).Provider(kCpuExecutionProvider).Build());
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6, 7).Provider(kCpuExecutionProvider).Build());
  Status st;
  ASSERT_FALSE((st = RegKernels(r, function_table, CreateFakeKernel)).IsOK());
}