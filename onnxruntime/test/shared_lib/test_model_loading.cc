// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/onnxruntime_cxx_api.h"
#include "core/platform/env.h"
#include "onnx_protobuf.h"
#include <google/protobuf/text_format.h>
#include "test_fixture.h"
#include "file_util.h"
namespace onnxruntime {
namespace test {
namespace {
void WriteStringToTempFile(const char* test_data, std::basic_string<ORTCHAR_T>& filename) {
  int fd;
  CreateTestFile(fd, filename);
  onnx::ModelProto mp;
  if (!google::protobuf::TextFormat::ParseFromString(test_data, &mp)) {
    throw std::runtime_error("protobuf parsing failed");
  }
  if (!mp.SerializeToFileDescriptor(fd))
    throw std::runtime_error("write file failed");
  auto st = Env::Default().FileClose(fd);
  if (!st.IsOK())
    throw std::runtime_error("close file failed");
}
}  // namespace

TEST_F(CApiTest, model_missing_data) {
  const char* test_data =
      "ir_version: 4\n"
      "graph {\n"
      "  node {\n"
      "    input: \"X\"\n"
      "    output: \"Y\"\n"
      "    op_type: \"Size\"\n"
      "  }\n"
      "  name: \"test-model\"\n"
      "  initializer {\n"
      "    dims: 100\n"
      "    dims: 3000\n"
      "    dims: 10\n"
      "    data_type: 1\n"
      "    name: \"X\"\n"
      "  }\n"
      "  input {\n"
      "    name: \"X\"\n"
      "    type {\n"
      "      tensor_type {\n"
      "        elem_type: 1\n"
      "        shape {\n"
      "          dim {\n"
      "            dim_value: 100\n"
      "          }\n"
      "          dim {\n"
      "            dim_value: 3000\n"
      "          }\n"
      "          dim {\n"
      "            dim_value: 10\n"
      "          }\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  }\n"
      "  output {\n"
      "    name: \"Y\"\n"
      "    type {\n"
      "      tensor_type {\n"
      "        elem_type: 7\n"
      "        shape {\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  }\n"
      "}\n"
      "opset_import {\n"
      "  domain: \"\"\n"
      "  version: 9\n"
      "}";
  std::basic_string<ORTCHAR_T> model_url(ORT_TSTR("model_XXXXXX"));
  WriteStringToTempFile(test_data, model_url);
  std::unique_ptr<ORTCHAR_T, decltype(&DeleteFileFromDisk)> file_deleter(const_cast<ORTCHAR_T*>(model_url.c_str()),
                                                                         DeleteFileFromDisk);
  std::unique_ptr<OrtSessionOptions> so(OrtCreateSessionOptions());
  OrtSession* ret;
  auto st = ::OrtCreateSession(env, model_url.c_str(), so.get(), &ret);
  ASSERT_NE(st, nullptr);
  OrtReleaseStatus(st);
}

TEST_F(CApiTest, model_with_external_data) {
  const char* test_data_begin =
      "ir_version: 4\n"
      "graph {\n"
      "  node {\n"
      "    input: \"X\"\n"
      "    output: \"Y\"\n"
      "    op_type: \"Size\"\n"
      "  }\n"
      "  name: \"test-model\"\n"
      "  initializer {\n"
      "    dims: 100\n"
      "    dims: 3000\n"
      "    dims: 10\n"
      "    data_type: 1\n"
      "    name: \"X\"\n"
      "    data_location: 1\n"
      "    external_data {\n"
      "      value: \"";

  const char* test_data_end =
      "\"\n"
      "      key:   \"location\"\n"
      "    }\n"
      "  }\n"
      "  input {\n"
      "    name: \"X\"\n"
      "    type {\n"
      "      tensor_type {\n"
      "        elem_type: 1\n"
      "        shape {\n"
      "          dim {\n"
      "            dim_value: 100\n"
      "          }\n"
      "          dim {\n"
      "            dim_value: 3000\n"
      "          }\n"
      "          dim {\n"
      "            dim_value: 10\n"
      "          }\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  }\n"
      "  output {\n"
      "    name: \"Y\"\n"
      "    type {\n"
      "      tensor_type {\n"
      "        elem_type: 7\n"
      "        shape {\n"
      "        }\n"
      "      }\n"
      "    }\n"
      "  }\n"
      "}\n"
      "opset_import {\n"
      "  domain: \"\"\n"
      "  version: 9\n"
      "}\n";
  std::basic_string<ORTCHAR_T> model_url(ORT_TSTR("model_XXXXXX"));
  std::basic_string<ORTCHAR_T> raw_data_url(ORT_TSTR("raw_data_XXXXXX"));
  FILE* fp;
  CreateTestFile(fp, raw_data_url);
  std::unique_ptr<ORTCHAR_T, decltype(&DeleteFileFromDisk)> file_deleter2(const_cast<ORTCHAR_T*>(raw_data_url.c_str()),
                                                                          DeleteFileFromDisk);
  float raw_data[3000];
  const size_t raw_data_len = sizeof(raw_data);
  for (int i = 0; i != 1000; ++i) {
    ASSERT_EQ(raw_data_len, fwrite(raw_data, 1, raw_data_len, fp));
  }
  ASSERT_EQ(0, fclose(fp));
  std::ostringstream oss;
  oss << test_data_begin << ToMBString(raw_data_url) << test_data_end;
  const std::string model_data = oss.str();
  WriteStringToTempFile(model_data.c_str(), model_url);
  std::unique_ptr<ORTCHAR_T, decltype(&DeleteFileFromDisk)> file_deleter(const_cast<ORTCHAR_T*>(model_url.c_str()),
                                                                         DeleteFileFromDisk);
  std::unique_ptr<OrtSessionOptions> so(OrtCreateSessionOptions());
  OrtSession* session;
  auto st = ::OrtCreateSession(env, model_url.c_str(), so.get(), &session);
  ASSERT_EQ(st, nullptr) << OrtGetErrorMessage(st);
  OrtReleaseStatus(st);
  ::OrtReleaseSession(session);
}
}  // namespace test
}  // namespace onnxruntime