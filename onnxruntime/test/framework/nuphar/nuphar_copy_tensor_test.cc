// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/nuphar/nuphar_allocator.h"
#include "test/framework/test_utils.h"
#include "default_providers.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

template<typename T>
void TestNupharCopyTensorWithPrimitiveType(IExecutionProvider *nuphar_provider,
                                           AllocatorPtr nuphar_allocator,
                                           const std::vector<int64_t> &dims,
                                           const std::vector<T> &src_values) {
  MLValue src_ml_value, dst_ml_value;
  CreateMLValue<T>(nuphar_allocator, dims, src_values, &src_ml_value);
  CreateMLValue<T>(nuphar_allocator, dims, std::vector<T>(src_values.size()), &dst_ml_value);
  const Tensor &src_tensor = src_ml_value.Get<Tensor>();
  Tensor* dst_tensor = dst_ml_value.GetMutable<Tensor>();

  Status status = nuphar_provider->CopyTensor(src_tensor, *dst_tensor);
  EXPECT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(src_tensor.Shape(), dst_tensor->Shape());
  EXPECT_EQ(src_tensor.DataType(), dst_tensor->DataType());
  std::vector<T> dst_values(dst_tensor->template Data<T>(), dst_tensor->template Data<T>() + src_values.size());
  ASSERT_EQ(src_values, dst_values);
}

TEST(NupharCopyTensorTests, PrimitiveTypesTest) {
  auto nuphar_provider_holder = DefaultNupharExecutionProvider();
  onnxruntime::IExecutionProvider* nuphar_provider = nuphar_provider_holder.get();
  auto nuphar_allocator = nuphar_provider->GetAllocator(0, OrtMemTypeDefault);

  TestNupharCopyTensorWithPrimitiveType<int8_t>(nuphar_provider, nuphar_allocator, {2, 3}, {1, 3, 2, 6, 5, 4});
  TestNupharCopyTensorWithPrimitiveType<uint16_t>(nuphar_provider, nuphar_allocator, {3, 2}, {11, 32, 21, 61, 51, 41});
  TestNupharCopyTensorWithPrimitiveType<int32_t>(nuphar_provider, nuphar_allocator, {2, 2}, {2112, 345, 90, 10});
  TestNupharCopyTensorWithPrimitiveType<uint64_t>(nuphar_provider, nuphar_allocator, {1, 2}, {9021, 1010});
  TestNupharCopyTensorWithPrimitiveType<float>(nuphar_provider, nuphar_allocator, {2, 2}, {1.0, 4.0, 3.0, 2.0});
  TestNupharCopyTensorWithPrimitiveType<double>(nuphar_provider, nuphar_allocator, {3, 2}, {1.0, 4.0, 3.0, 2.0, 11.0, 12.0});
}

}  // namespace test
}  // namespace onnxruntime
