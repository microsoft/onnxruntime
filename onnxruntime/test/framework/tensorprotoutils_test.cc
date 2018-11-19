// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env.h"
#include "core/framework/tensor.h"
#include "core/graph/onnx_protobuf.h"
#include "core/framework/tensorprotoutils.h"
#include "gtest/gtest.h"
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <sstream>

namespace onnxruntime {
namespace test {
#ifdef ONNXRUNTIME_RUN_EXTERNAL_ONNX_TESTS
TEST(TensorProtoUtilsTest, test1) {
  const char* filename = "../models/opset8/test_resnet50/test_data_set_0/input_0.pb";
  int test_data_pb_fd;
  common::Status st = Env::Default().FileOpenRd(filename, test_data_pb_fd);
  ASSERT_TRUE(st.IsOK());
  google::protobuf::io::FileInputStream f(test_data_pb_fd);
  f.SetCloseOnDelete(true);
  ONNX_NAMESPACE::TensorProto proto;
  ASSERT_TRUE(proto.ParseFromZeroCopyStream(&f));
  std::unique_ptr<Tensor> tensor;
  ::onnxruntime::AllocatorPtr cpu_allocator = std::make_shared<::onnxruntime::CPUAllocator>();
  st = ::onnxruntime::utils::GetTensorFromTensorProto(proto, &tensor, cpu_allocator);
  ASSERT_TRUE(st.IsOK());
}
#endif
}  // namespace test
}  // namespace onnxruntime
