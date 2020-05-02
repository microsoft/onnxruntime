// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/framework/protobuf_message_sequence.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "google/protobuf/io/zero_copy_stream_impl_lite.h"

#include "core/framework/data_types.h"

namespace onnxruntime {
namespace test {
TEST(ProtoMessageSequenceTest, Basic) {
  auto make_kvp_proto = [](std::string key, std::string value) {
    ONNX_NAMESPACE::StringStringEntryProto proto{};
    proto.set_key(std::move(key));
    proto.set_value(std::move(value));
    return proto;
  };

  const std::vector<ONNX_NAMESPACE::StringStringEntryProto> kvp_protos{
      make_kvp_proto("greeting", "hello world"),
      make_kvp_proto("answer", "42"),
      make_kvp_proto("color", "mauve"),
  };

  std::string buffer_str;

  google::protobuf::io::StringOutputStream output{&buffer_str};
  auto status = WriteProtoMessageSequence(kvp_protos, output);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  std::vector<ONNX_NAMESPACE::StringStringEntryProto> parsed_kvp_protos{};
  google::protobuf::io::ArrayInputStream input{
      buffer_str.data(), static_cast<int>(buffer_str.size())};
  status = ReadProtoMessageSequence(parsed_kvp_protos, input);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();

  auto kvp_eq = [](const ONNX_NAMESPACE::StringStringEntryProto& a,
                   const ONNX_NAMESPACE::StringStringEntryProto& b) {
    return a.key() == b.key() && a.value() == b.value();
  };

  ASSERT_EQ(kvp_protos.size(), parsed_kvp_protos.size());
  ASSERT_TRUE(std::equal(kvp_protos.begin(), kvp_protos.end(), parsed_kvp_protos.begin(), kvp_eq));
}
}  // namespace test
}  // namespace onnxruntime
