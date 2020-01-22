// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/tensor.h"
#include "core/framework/allocatormgr.h"
#include "test_utils.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

#include <sstream>

namespace onnxruntime {
namespace test {
template <typename T>
void CPUTensorTest(std::vector<int64_t> dims, const int offset = 0) {
  //not own the buffer
  TensorShape shape(dims);
  auto alloc = TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault);
  auto data = alloc->Alloc(sizeof(T) * (shape.Size() + offset));
  EXPECT_TRUE(data);
  Tensor t(DataTypeImpl::GetType<T>(), shape, data, alloc->Info(), offset);
  auto tensor_shape = t.Shape();
  //Use reinterpret_cast to bypass a gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51213
  EXPECT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&shape), *reinterpret_cast<const std::vector<int64_t>*>(&tensor_shape));
  EXPECT_EQ(t.DataType(), DataTypeImpl::GetType<T>());
  auto& location = t.Location();
  EXPECT_STREQ(location.name, CPU);
  EXPECT_EQ(location.id, 0);

  auto t_data = t.template MutableData<T>();
  EXPECT_TRUE(t_data);
  memset(t_data, 0, sizeof(T) * shape.Size());
  EXPECT_EQ(*(T*)((char*)data + offset), (T)0);
  alloc->Free(data);

  Tensor new_t(DataTypeImpl::GetType<T>(), shape, alloc, offset);

  tensor_shape = new_t.Shape();
  //Use reinterpret_cast to bypass a gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51213
  EXPECT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&shape), *reinterpret_cast<const std::vector<int64_t>*>(&tensor_shape));
  EXPECT_EQ(new_t.DataType(), DataTypeImpl::GetType<T>());
  auto& new_location = new_t.Location();
  ASSERT_STREQ(new_location.name, CPU);
  EXPECT_EQ(new_location.id, 0);

  auto new_data = new_t.template MutableData<T>();
  EXPECT_TRUE(new_data);
  memset(new_data, 0, sizeof(T) * shape.Size());
  EXPECT_EQ(*(T*)((char*)new_data + offset), (T)0);
  //no free op as the tensor own the buffer
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
  Tensor t(type, TensorShape({1, 0}), nullptr, TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault)->Info());
  auto& shape = t.Shape();
  EXPECT_EQ(shape.Size(), 0);
  EXPECT_EQ(t.DataType(), type);

  auto data = t.template MutableData<float>();
  EXPECT_TRUE(!data);

  auto& location = t.Location();
  ASSERT_STREQ(location.name, CPU);
  EXPECT_EQ(location.id, 0);

  // arena is disabled for CPUExecutionProvider on x86 and JEMalloc
#if (defined(__amd64__) || defined(_M_AMD64)) && !defined(USE_JEMALLOC)
  EXPECT_EQ(location.alloc_type, OrtAllocatorType::OrtArenaAllocator);
#else
  EXPECT_EQ(location.alloc_type, OrtAllocatorType::OrtDeviceAllocator);
#endif
}

TEST(TensorTest, StringTensorTest) {
//add scope to explicitly delete tensor
#ifdef _MSC_VER
  std::string* string_ptr = nullptr;
#else
  std::string* string_ptr __attribute__((unused)) = nullptr;
#endif
  {
    TensorShape shape({2, 3});
    auto alloc = TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault);
    Tensor t(DataTypeImpl::GetType<std::string>(), shape, alloc);

    auto& tensor_shape = t.Shape();
    //Use reinterpret_cast to bypass a gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51213
    EXPECT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&shape), *reinterpret_cast<const std::vector<int64_t>*>(&tensor_shape));
    EXPECT_EQ(t.DataType(), DataTypeImpl::GetType<std::string>());
    auto& location = t.Location();
    ASSERT_STREQ(location.name, CPU);
    EXPECT_EQ(location.id, 0);

    std::string* new_data = t.template MutableData<std::string>();
    EXPECT_TRUE(new_data);
    new_data[0] = "a";
    new_data[1] = "b";

    auto tensor_data = t.template Data<std::string>();
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

}  // namespace test
}  // namespace onnxruntime
