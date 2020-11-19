// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <gtest/gtest.h>
#include <core/framework/kernel_registry.h>
#include <core/framework/op_kernel.h>

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
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(5,6).Provider(kCpuExecutionProvider).Build());
  function_table.emplace_back(KernelDefBuilder().MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<float>()).SetName("Elu").SetDomain("").SinceVersion(6,7).Provider(kCpuExecutionProvider).Build());
  Status st;
  ASSERT_FALSE((st = RegKernels(r, function_table, CreateFakeKernel)).IsOK());
}