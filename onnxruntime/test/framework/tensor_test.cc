// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"
#include "test_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <absl/base/config.h>
#include <sstream>

namespace onnxruntime {
namespace test {
template <typename T>
void CPUTensorTest(std::vector<int64_t> dims, const int offset_elements = 0) {
  // create Tensor where we provide the buffer
  TensorShape shape(dims);  // this is the shape that will be available starting at the offset in the Tensor
  auto alloc = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  // alloc extra data if needed, as anything before the offset is not covered by the shape
  auto num_elements = shape.Size() + offset_elements;
  auto num_bytes = num_elements * sizeof(T);
  auto offset_bytes = offset_elements * sizeof(T);
  void* data = alloc->Alloc(num_bytes);
  const T* first_element = static_cast<const T*>(data) + offset_elements;

  Tensor t(DataTypeImpl::GetType<T>(), shape, data, alloc->Info(), offset_bytes);
  auto tensor_shape = t.Shape();
  EXPECT_EQ(shape.GetDims(), tensor_shape.GetDims());
  EXPECT_EQ(t.DataType(), DataTypeImpl::GetType<T>());
  auto& location = t.Location();
  EXPECT_STREQ(location.name, CPU);
  EXPECT_EQ(location.id, 0);

  const T* t_data = t.Data<T>();
  EXPECT_EQ(first_element, t_data);
  alloc->Free(data);

  // test when the Tensor allocates the buffer.
  // there's no point using an offset_elements here as you'd be allocating extra data prior to the buffer needed
  // by the Tensor instance.
  if (offset_elements == 0) {
    Tensor new_t(DataTypeImpl::GetType<T>(), shape, alloc);
    EXPECT_TRUE(new_t.OwnsBuffer());

    tensor_shape = new_t.Shape();
    EXPECT_EQ(shape.GetDims(), tensor_shape.GetDims());
    EXPECT_EQ(new_t.DataType(), DataTypeImpl::GetType<T>());
    auto& new_location = new_t.Location();
    ASSERT_STREQ(new_location.name, CPU);
    EXPECT_EQ(new_location.id, 0);
  }
}

TEST(TensorTest, CPUFloatTensorTest) {
  CPUTensorTest<float>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUInt32TensorTest) {
  CPUTensorTest<int32_t>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUUInt8TensorTest) {
  CPUTensorTest<uint8_t>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUUInt16TensorTest) {
  CPUTensorTest<uint16_t>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUInt16TensorTest) {
  CPUTensorTest<int16_t>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUInt64TensorTest) {
  CPUTensorTest<int64_t>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUDoubleTensorTest) {
  CPUTensorTest<double>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUUInt32TensorTest) {
  CPUTensorTest<uint32_t>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUUInt64TensorTest) {
  CPUTensorTest<uint64_t>(std::vector<int64_t>({3, 2, 4}));
}

TEST(TensorTest, CPUFloatTensorOffsetTest) {
  CPUTensorTest<float>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, CPUInt32TensorOffsetTest) {
  CPUTensorTest<int32_t>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, CPUUInt8TensorOffsetTest) {
  CPUTensorTest<uint8_t>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, CPUUInt16TensorOffsetTest) {
  CPUTensorTest<uint16_t>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, CPUInt16TensorOffsetTest) {
  CPUTensorTest<int16_t>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, CPUInt64TensorOffsetTest) {
  CPUTensorTest<int64_t>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, CPUDoubleTensorOffsetTest) {
  CPUTensorTest<double>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, CPUUInt32TensorOffsetTest) {
  CPUTensorTest<uint32_t>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, CPUUInt64TensorOffsetTest) {
  CPUTensorTest<uint64_t>(std::vector<int64_t>({3, 2, 4}), 5);
}

TEST(TensorTest, EmptyTensorTest) {
  auto type = DataTypeImpl::GetType<float>();
  Tensor t(type, TensorShape({1, 0}), nullptr, TestCPUExecutionProvider()->CreatePreferredAllocators()[0]->Info());
  auto& shape = t.Shape();
  EXPECT_EQ(shape.Size(), 0);
  EXPECT_EQ(t.DataType(), type);

  auto data = t.MutableData<float>();
  EXPECT_TRUE(!data);

  auto& location = t.Location();
  ASSERT_STREQ(location.name, CPU);
  EXPECT_EQ(location.id, 0);

  // arena is disabled for CPUExecutionProvider on x86 and JEMalloc
#if (defined(__amd64__) || defined(_M_AMD64) || defined(__aarch64__) || defined(_M_ARM64)) && !defined(USE_JEMALLOC) && !defined(USE_MIMALLOC) && !defined(ABSL_HAVE_ADDRESS_SANITIZER)
  EXPECT_EQ(location.alloc_type, OrtAllocatorType::OrtArenaAllocator);
#else
  EXPECT_EQ(location.alloc_type, OrtAllocatorType::OrtDeviceAllocator);
#endif
}

TEST(TensorTest, StringTensorTest) {
// add scope to explicitly delete tensor
#ifdef _MSC_VER
  std::string* string_ptr = nullptr;
#else
  std::string* string_ptr __attribute__((unused)) = nullptr;
#endif
  {
    TensorShape shape({2, 3});
    auto alloc = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
    Tensor t(DataTypeImpl::GetType<std::string>(), shape, alloc);

    auto& tensor_shape = t.Shape();
    EXPECT_EQ(shape, tensor_shape);
    EXPECT_EQ(t.DataType(), DataTypeImpl::GetType<std::string>());
    auto& location = t.Location();
    ASSERT_STREQ(location.name, CPU);
    EXPECT_EQ(location.id, 0);

    std::string* new_data = t.MutableData<std::string>();
    EXPECT_TRUE(new_data);
    new_data[0] = "a";
    new_data[1] = "b";

    auto tensor_data = t.Data<std::string>();
    EXPECT_EQ(tensor_data[0], "a");
    EXPECT_EQ(tensor_data[1], "b");
    string_ptr = new_data;
  }
}

TEST(TensorTest, ConvertToString) {
  TensorShape shape({2, 3, 4});

  EXPECT_EQ(shape.ToString(), "{2,3,4}");

  std::ostringstream ss;
  ss << shape;
  EXPECT_EQ(ss.str(), "{2,3,4}");
}

TEST(TensorTest, Int64PtrConstructor) {
  int64_t dimensions[] = {2, 3, 4};
  TensorShape shape(dimensions, 2);  // just use first 2
  EXPECT_EQ(shape.Size(), 6);
  EXPECT_EQ(shape.NumDimensions(), 2u);
  EXPECT_THAT(shape.GetDims(), testing::ElementsAre(2, 3));
}

TEST(TensorTest, SizeOverflow) {
  // shape overflow
  EXPECT_THROW(TensorShape({std::numeric_limits<int64_t>::max() / 2, 3}).Size(), OnnxRuntimeException);

  auto type = DataTypeImpl::GetType<float>();
  auto alloc = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];

  // total size overflow with 4 bytes per element
  TensorShape shape1({static_cast<int64_t>(std::numeric_limits<size_t>::max() / 3)});
  EXPECT_THROW(Tensor(type, shape1, alloc), OnnxRuntimeException);

  Tensor t(type, shape1, nullptr, alloc->Info());
  EXPECT_THROW(t.SizeInBytes(), OnnxRuntimeException);
}

#ifdef ENABLE_STRIDED_TENSORS
TEST(TensorTest, Strided) {
  TensorShape shape({2, 3, 4});
  auto alloc = TestCPUExecutionProvider()->CreatePreferredAllocators()[0];
  void* data = alloc->Alloc(shape.Size() * sizeof(float));

  Tensor t(DataTypeImpl::GetType<float>(), shape, data, alloc->Info());
  EXPECT_TRUE(t.IsContiguous());
  const TensorShapeVector strides{12, 4, 1};
  ASSERT_EQ(t.Shape(), shape);
  ASSERT_THAT(t.Strides(), testing::ContainerEq(gsl::make_span(strides)));
  ASSERT_EQ(t.SizeInBytes(), sizeof(float) * 24);
  TensorShape new_shape({4, 2, 3});
  const TensorShapeVector new_strides{1, 12, 4};
  t.SetShapeAndStrides(new_shape, new_strides);
  EXPECT_FALSE(t.IsContiguous());
  ASSERT_EQ(t.Shape(), new_shape);
  ASSERT_THAT(t.Strides(), testing::ContainerEq(gsl::make_span(new_strides)));
  ASSERT_EQ(t.SizeInBytes(), sizeof(float) * 24);

  Tensor t2(DataTypeImpl::GetType<float>(), new_shape, data, alloc->Info(), 0L, gsl::make_span(new_strides));
  EXPECT_FALSE(t2.IsContiguous());
  ASSERT_EQ(t2.Shape(), new_shape);
  ASSERT_THAT(t2.Strides(), testing::ContainerEq(gsl::make_span(new_strides)));
  ASSERT_EQ(t2.SizeInBytes(), sizeof(float) * 24);
  t2.SetShapeAndStrides(shape, strides);
  EXPECT_TRUE(t2.IsContiguous());
  ASSERT_EQ(t2.Shape(), shape);
  ASSERT_THAT(t2.Strides(), testing::ContainerEq(gsl::make_span(strides)));
  ASSERT_EQ(t2.SizeInBytes(), sizeof(float) * 24);

  alloc->Free(data);

  data = alloc->Alloc(sizeof(int64_t));
  const TensorShapeVector single_element_strides{0, 0, 0};
  Tensor t3(DataTypeImpl::GetType<int64_t>(), shape, data, alloc->Info(), 0L, gsl::make_span(single_element_strides));
  EXPECT_FALSE(t3.IsContiguous());
  ASSERT_EQ(t3.Shape(), shape);
  ASSERT_THAT(t3.Strides(), testing::ContainerEq(gsl::make_span(single_element_strides)));
  ASSERT_EQ(t3.SizeInBytes(), sizeof(int64_t));
  alloc->Free(data);

  const TensorShapeVector zero_strides{0, 0, 0};
  Tensor t4(DataTypeImpl::GetType<float>(), shape, alloc);
  t4.SetShapeAndStrides(shape, zero_strides);
  EXPECT_FALSE(t4.IsContiguous());
  EXPECT_EQ(t4.Shape(), shape);
  ASSERT_THAT(t4.Strides(), testing::ContainerEq(gsl::make_span(zero_strides)));
  ASSERT_EQ(t4.SizeInBytes(), sizeof(float));
}
#endif

}  // namespace test
}  // namespace onnxruntime
