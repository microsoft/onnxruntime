// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_registry.h"

#include <gtest/gtest.h>

#include "asserts.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime::test {

namespace {
Status RegKernels(KernelRegistry& r, std::vector<std::unique_ptr<KernelDef>>& function_table, const KernelCreateFn& kernel_creator) {
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

Status CreateFakeKernel(FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) {
  out = std::make_unique<FakeKernel>(info);
  return Status::OK();
}
}  // namespace

TEST(KernelRegistryTests, simple) {
  KernelRegistry r;
  std::vector<std::unique_ptr<KernelDef>> function_table;
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  ASSERT_STATUS_OK(RegKernels(r, function_table, CreateFakeKernel));
}

TEST(KernelRegistryTests, dup_simple) {
  KernelRegistry r;
  std::vector<std::unique_ptr<KernelDef>> function_table;
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  ASSERT_STATUS_NOT_OK(RegKernels(r, function_table, CreateFakeKernel));
}

//duplicated registration. One in default("") domain, another in "ai.onnx" domain
TEST(KernelRegistryTests, dup_simple2) {
  KernelRegistry r;
  std::vector<std::unique_ptr<KernelDef>> function_table;
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("ai.onnx").SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  ASSERT_STATUS_NOT_OK(RegKernels(r, function_table, CreateFakeKernel));
}

//One in default("") domain, another in ms domain. Should be ok
TEST(KernelRegistryTests, one_op_name_in_two_domains) {
  KernelRegistry r;
  std::vector<std::unique_ptr<KernelDef>> function_table;
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain(kMSDomain).SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  ASSERT_STATUS_OK(RegKernels(r, function_table, CreateFakeKernel));
}

//One op two versions
TEST(KernelRegistryTests, two_versions) {
  KernelRegistry r;
  std::vector<std::unique_ptr<KernelDef>> function_table;
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(1, 5).Provider(kCpuExecutionProvider).Build());
  ASSERT_STATUS_OK(RegKernels(r, function_table, CreateFakeKernel));
}

//One op two versions
TEST(KernelRegistryTests, two_versions2) {
  KernelRegistry r;
  std::vector<std::unique_ptr<KernelDef>> function_table;
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(1, 6).Provider(kCpuExecutionProvider).Build());
  ASSERT_STATUS_NOT_OK(RegKernels(r, function_table, CreateFakeKernel));
}

//One op two versions
TEST(KernelRegistryTests, two_versions3) {
  KernelRegistry r;
  std::vector<std::unique_ptr<KernelDef>> function_table;
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6).Provider(kCpuExecutionProvider).Build());
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(1).Provider(kCpuExecutionProvider).Build());
  ASSERT_STATUS_OK(RegKernels(r, function_table, CreateFakeKernel));
}

//One op two versions
TEST(KernelRegistryTests, two_versions4) {
  KernelRegistry r;
  std::vector<std::unique_ptr<KernelDef>> function_table;
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(5, 6).Provider(kCpuExecutionProvider).Build());
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6, 7).Provider(kCpuExecutionProvider).Build());
  ASSERT_STATUS_NOT_OK(RegKernels(r, function_table, CreateFakeKernel));
}

TEST(KernelRegistryTests, TryFindKernelByHash) {
  auto kernel_def =
      KernelDefBuilder()
          .MayInplace(0, 0)
          .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
          .SetName("Elu")
          .SetDomain("")
          .SinceVersion(6)
          .Provider(kCpuExecutionProvider)
          .Build();
  const auto kernel_def_hash = kernel_def->GetHash();
  std::vector<std::unique_ptr<KernelDef>> function_table{};
  function_table.emplace_back(std::move(kernel_def));
  KernelRegistry r{};
  ASSERT_STATUS_OK(RegKernels(r, function_table, CreateFakeKernel));

  const KernelCreateInfo* pkci = nullptr;
  ASSERT_TRUE(r.TryFindKernelByHash(kernel_def_hash, &pkci));

  const auto unregistered_kernel_def_hash =
      KernelDefBuilder()
          .MayInplace(0, 0)
          .TypeConstraint("T", DataTypeImpl::GetTensorType<float>())
          .SetName("Elu")
          .SetDomain("")
          .SinceVersion(1)  // different from registered kernel
          .Provider(kCpuExecutionProvider)
          .Build()
          ->GetHash();
  ASSERT_FALSE(r.TryFindKernelByHash(unregistered_kernel_def_hash, &pkci));
}

}  // namespace onnxruntime::test
