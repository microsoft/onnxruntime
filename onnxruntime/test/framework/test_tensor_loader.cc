// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/framework/callback.h"
#include "core/framework/tensorprotoutils.h"
#include "test/util/include/file_util.h"
#include "test/util/include/asserts.h"

#ifdef _WIN32
#include <Windows.h>
#endif

const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);

namespace onnxruntime {
namespace test {

TEST(CApiTensorTest, load_simple_float_tensor_not_enough_space) {
  // construct a tensor proto
  onnx::TensorProto p;
  p.mutable_float_data()->Add(1.0f);
  p.mutable_float_data()->Add(2.2f);
  p.mutable_dims()->Add(2);
  p.set_data_type(onnx::TensorProto_DataType_FLOAT);
  std::string s;
  // save it to a buffer
  ASSERT_TRUE(p.SerializeToString(&s));
  // deserialize it
  std::vector<float> output(1);
  OrtValue value;
  OrtMemoryInfo cpu_memory_info(onnxruntime::CPU, OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeDefault);

  ASSERT_STATUS_NOT_OK(
      utils::TensorProtoToOrtValue(Env::Default(), nullptr, p,
                                   MemBuffer(output.data(), output.size() * sizeof(float), cpu_memory_info),
                                   value));
}

TEST(CApiTensorTest, load_simple_float_tensor_membuffer) {
  // construct a tensor proto
  onnx::TensorProto p;
  p.mutable_float_data()->Add(1.0f);
  p.mutable_float_data()->Add(2.2f);
  p.mutable_float_data()->Add(3.5f);
  p.mutable_dims()->Add(3);
  p.set_data_type(onnx::TensorProto_DataType_FLOAT);
  std::string s;
  // save it to a buffer
  ASSERT_TRUE(p.SerializeToString(&s));
  // deserialize it
  std::vector<float> output(3);
  OrtValue value;
  OrtMemoryInfo cpu_memory_info(onnxruntime::CPU, OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeDefault);
  ASSERT_STATUS_OK(
      utils::TensorProtoToOrtValue(Env::Default(), nullptr, p,
                                   MemBuffer(output.data(), output.size() * sizeof(float), cpu_memory_info),
                                   value));
  float* real_output;
  auto ort_st = g_ort->GetTensorMutableData(&value, (void**)&real_output);
  ASSERT_EQ(ort_st, nullptr) << g_ort->GetErrorMessage(ort_st);
  // check the result
  ASSERT_EQ(real_output[0], 1.0f);
  ASSERT_EQ(real_output[1], 2.2f);
  ASSERT_EQ(real_output[2], 3.5f);
  g_ort->ReleaseStatus(ort_st);
}

TEST(CApiTensorTest, load_simple_float_tensor_allocator) {
  // construct a tensor proto
  onnx::TensorProto p;
  p.mutable_float_data()->Add(1.0f);
  p.mutable_float_data()->Add(2.2f);
  p.mutable_float_data()->Add(3.5f);
  p.mutable_dims()->Add(3);
  p.set_data_type(onnx::TensorProto_DataType_FLOAT);
  std::string s;
  // save it to a buffer
  ASSERT_TRUE(p.SerializeToString(&s));
  // deserialize it
  AllocatorPtr tmp_allocator = std::make_shared<CPUAllocator>();
  OrtValue value;

  ASSERT_STATUS_OK(utils::TensorProtoToOrtValue(Env::Default(), nullptr, p, tmp_allocator, value));

  float* real_output;
  auto ort_st = g_ort->GetTensorMutableData(&value, (void**)&real_output);
  ASSERT_EQ(ort_st, nullptr) << g_ort->GetErrorMessage(ort_st);
  // check the result
  ASSERT_EQ(real_output[0], 1.0f);
  ASSERT_EQ(real_output[1], 2.2f);
  ASSERT_EQ(real_output[2], 3.5f);
  g_ort->ReleaseStatus(ort_st);
}

template <bool use_current_dir>
static void run_external_data_test() {
  FILE* fp;
  std::basic_string<ORTCHAR_T> filename(ORT_TSTR("tensor_XXXXXX"));
  CreateTestFile(fp, filename);
  std::unique_ptr<ORTCHAR_T, decltype(&DeleteFileFromDisk)> file_deleter(const_cast<ORTCHAR_T*>(filename.c_str()),
                                                                         DeleteFileFromDisk);
  float test_data[] = {1.0f, 2.2f, 3.5f};
  ASSERT_EQ(sizeof(test_data), fwrite(test_data, 1, sizeof(test_data), fp));
  ASSERT_EQ(0, fclose(fp));
  // construct a tensor proto
  onnx::TensorProto p;
  onnx::StringStringEntryProto* location = p.mutable_external_data()->Add();
  location->set_key("location");
  location->set_value(ToUTF8String(filename));
  p.mutable_dims()->Add(3);
  p.set_data_location(onnx::TensorProto_DataLocation_EXTERNAL);
  p.set_data_type(onnx::TensorProto_DataType_FLOAT);
  std::string s;
  // save it to a buffer
  ASSERT_TRUE(p.SerializeToString(&s));
  // deserialize it
  std::vector<float> output(3);
  std::basic_string<ORTCHAR_T> cwd;
  if (use_current_dir) {
#ifdef _WIN32
    DWORD len = GetCurrentDirectory(0, nullptr);
    ASSERT_NE(len, (DWORD)0);
    cwd.resize(static_cast<size_t>(len) - 1, '\0');
    len = GetCurrentDirectoryW(len, (ORTCHAR_T*)cwd.data());
    ASSERT_NE(len, (DWORD)0);
    cwd.append(ORT_TSTR("\\fake.onnx"));
#else
    char* p = getcwd(nullptr, 0);
    ASSERT_NE(p, nullptr);
    cwd = p;
    free(p);
    cwd.append(ORT_TSTR("/fake.onnx"));
#endif
  }
  OrtValue value;
  OrtMemoryInfo cpu_memory_info(onnxruntime::CPU, OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeDefault);
  ASSERT_STATUS_OK(utils::TensorProtoToOrtValue(
      Env::Default(), nullptr, p, MemBuffer(output.data(), output.size() * sizeof(float), cpu_memory_info), value));

  float* real_output;
  auto ort_st = g_ort->GetTensorMutableData(&value, (void**)&real_output);
  ASSERT_EQ(ort_st, nullptr) << g_ort->GetErrorMessage(ort_st);
  // check the result
  ASSERT_EQ(real_output[0], 1.0f);
  ASSERT_EQ(real_output[1], 2.2f);
  ASSERT_EQ(real_output[2], 3.5f);
  g_ort->ReleaseStatus(ort_st);
}

TEST(CApiTensorTest, load_float_tensor_with_external_data) {
  run_external_data_test<true>();
  run_external_data_test<false>();
}

#if defined(__amd64__) || defined(_M_X64)
#ifndef __ANDROID__
#ifdef NDEBUG
TEST(CApiTensorTest, load_huge_tensor_with_external_data) {
  FILE* fp;
  std::basic_string<ORTCHAR_T> filename(ORT_TSTR("tensor_XXXXXX"));
  CreateTestFile(fp, filename);
  std::unique_ptr<ORTCHAR_T, decltype(&DeleteFileFromDisk)> file_deleter(const_cast<ORTCHAR_T*>(filename.c_str()),
                                                                         DeleteFileFromDisk);
  std::vector<int> data(524288, 1);
  const size_t len = data.size() * sizeof(int);
  for (int i = 0; i != 1025; ++i) {
    ASSERT_EQ(len, fwrite(data.data(), 1, len, fp));
  }
  constexpr size_t total_ele_count = 524288 * 1025;
  ASSERT_EQ(0, fclose(fp));
  // construct a tensor proto
  onnx::TensorProto p;
  onnx::StringStringEntryProto* location = p.mutable_external_data()->Add();
  location->set_key("location");
  location->set_value(ToUTF8String(filename));
  p.mutable_dims()->Add(524288);
  p.mutable_dims()->Add(1025);
  p.set_data_location(onnx::TensorProto_DataLocation_EXTERNAL);
  p.set_data_type(onnx::TensorProto_DataType_INT32);
  std::string s;
  // save it to a buffer
  ASSERT_TRUE(p.SerializeToString(&s));
  // deserialize it
  std::vector<int> output(total_ele_count);
  OrtValue value;
  OrtMemoryInfo cpu_memory_info(onnxruntime::CPU, OrtDeviceAllocator, OrtDevice(), 0, OrtMemTypeDefault);
  ASSERT_STATUS_OK(
      utils::TensorProtoToOrtValue(Env::Default(), nullptr, p,
                                   MemBuffer(output.data(), output.size() * sizeof(int), cpu_memory_info), value));

  int* buffer;
  auto ort_st = g_ort->GetTensorMutableData(&value, (void**)&buffer);
  ASSERT_EQ(ort_st, nullptr) << "Error from OrtGetTensorMutableData: " << g_ort->GetErrorMessage(ort_st);
  for (size_t i = 0; i != total_ele_count; ++i) {
    ASSERT_EQ(1, buffer[i]);
  }
  g_ort->ReleaseStatus(ort_st);
}
#endif
#endif
#endif
}  // namespace test
}  // namespace onnxruntime
