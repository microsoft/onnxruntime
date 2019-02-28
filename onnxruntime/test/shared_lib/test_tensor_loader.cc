// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "onnx_protobuf.h"

#include "test_fixture.h"
#ifdef _WIN32
#include <Windows.h>
#endif
using namespace onnxruntime;

TEST_F(CApiTest, load_simple_float_tensor) {
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
  OrtValue* value;
  OrtDeleter* deleter;
  auto st = OrtTensorProtoToOrtValue(s.data(), static_cast<int>(s.size()), nullptr, output.data(),
                                     output.size() * sizeof(float), &value, &deleter);
  // check the result
  ASSERT_EQ(st, nullptr) << OrtGetErrorMessage(st);
  ASSERT_EQ(output[0], 1.0f);
  ASSERT_EQ(output[1], 2.2f);
  ASSERT_EQ(output[2], 3.5f);
  OrtReleaseValue(value);
}

static void CreateTestFile(FILE** out, std::basic_string<ORTCHAR_T>* final_filename) {
  ORTCHAR_T filename[] = ORT_TSTR("tensor_XXXXXX");
#ifdef _WIN32
  ASSERT_EQ(0, _wmktemp_s(filename, _countof(filename)));
  FILE* fp = nullptr;
  ASSERT_EQ(0, _wfopen_s(&fp, filename, ORT_TSTR("w")));
#else
  int fd = mkstemp(filename);
  ASSERT_TRUE(fd >= 0);
  FILE* fp = fdopen(fd, "w");
#endif
  *out = fp;
  *final_filename = filename;
}

TEST_F(CApiTest, load_float_tensor_with_external_data) {
  FILE* fp;
  std::basic_string<ORTCHAR_T> filename;
  CreateTestFile(&fp, &filename);
  float test_data[] = {1.0f, 2.2f, 3.5f};
  ASSERT_EQ(sizeof(test_data), fwrite(test_data, 1, sizeof(test_data), fp));
  ASSERT_EQ(0, fclose(fp));
  // construct a tensor proto
  onnx::TensorProto p;
  onnx::StringStringEntryProto* location = p.mutable_external_data()->Add();
  location->set_key("location");
  location->set_value(ToMBString(filename));
  p.mutable_dims()->Add(3);
  p.set_data_location(onnx::TensorProto_DataLocation_EXTERNAL);
  p.set_data_type(onnx::TensorProto_DataType_FLOAT);
  std::string s;
  // save it to a buffer
  ASSERT_TRUE(p.SerializeToString(&s));
  // deserialize it
  std::vector<float> output(3);
  OrtValue* value;
  OrtDeleter* deleter;
  auto st = OrtTensorProtoToOrtValue(s.data(), static_cast<int>(s.size()), nullptr, output.data(),
                                     output.size() * sizeof(float), &value, &deleter);
#ifdef _WIN32
  ASSERT_EQ(TRUE, DeleteFileW(filename.c_str()));
#else
  ASSERT_EQ(0, unlink(filename.c_str()));
#endif
  // check the result
  ASSERT_EQ(st, nullptr) << OrtGetErrorMessage(st);
  ASSERT_EQ(output[0], 1.0f);
  ASSERT_EQ(output[1], 2.2f);
  ASSERT_EQ(output[2], 3.5f);
  OrtReleaseValue(value);
}

#if defined(__amd64__) || defined(_M_X64)

TEST_F(CApiTest, load_huge_tensor_with_external_data) {
  FILE* fp;
  std::basic_string<ORTCHAR_T> filename;
  CreateTestFile(&fp, &filename);
  std::vector<int> data(524288, 1);
  const size_t len = data.size() * sizeof(int);
  for (int i = 0; i != 1025; ++i) {
    ASSERT_EQ(len, fwrite(data.data(), 1, len, fp));
  }
  const size_t total_ele_count = 524288 * 1025;
  ASSERT_EQ(0, fclose(fp));
  // construct a tensor proto
  onnx::TensorProto p;
  onnx::StringStringEntryProto* location = p.mutable_external_data()->Add();
  location->set_key("location");
  location->set_value(ToMBString(filename));
  p.mutable_dims()->Add(524288);
  p.mutable_dims()->Add(1025);
  p.set_data_location(onnx::TensorProto_DataLocation_EXTERNAL);
  p.set_data_type(onnx::TensorProto_DataType_INT32);
  std::string s;
  // save it to a buffer
  ASSERT_TRUE(p.SerializeToString(&s));
  // deserialize it
  std::vector<int> output(total_ele_count);
  OrtValue* value;
  OrtDeleter* deleter;
  auto st = OrtTensorProtoToOrtValue(s.data(), static_cast<int>(s.size()), nullptr, output.data(),
                                     output.size() * sizeof(int), &value, &deleter);
#ifdef _WIN32
  ASSERT_EQ(TRUE, DeleteFileW(filename.c_str()));
#else
  ASSERT_EQ(0, unlink(filename.c_str()));
#endif
  // check the result
  ASSERT_EQ(st, nullptr) << OrtGetErrorMessage(st);
  int* buffer;
  st = OrtGetTensorMutableData(value, (void**)&buffer);
  ASSERT_EQ(st, nullptr) << OrtGetErrorMessage(st);
  for (size_t i = 0; i != total_ele_count; ++i) {
    ASSERT_EQ(1, buffer[i]);
  }
  OrtReleaseValue(value);
}
#endif