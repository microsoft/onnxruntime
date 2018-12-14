#include "gtest/gtest.h"
#include "google/protobuf/stubs/common.h"

#include "core/framework/environment.h"
#include "core/session/inference_session.h"

namespace onnxruntime {
namespace test {
TEST(InferenceSessionWithoutEnvironment, UninitializedEnvironment)
{
  EXPECT_FALSE(onnxruntime::Environment::IsInitialized());

  onnxruntime::SessionOptions session_options{};
  EXPECT_THROW(onnxruntime::InferenceSession{session_options},
               onnxruntime::OnnxRuntimeException);
}

// call protobuf shutdown to avoid memory leak
class TestEnvironment : public ::testing::Environment {
 public:
  void TearDown() override {
    ::google::protobuf::ShutdownProtobufLibrary();
  }
};
}  // namespace test
}  // namespace onnxruntime

GTEST_API_ int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  // the following call takes ownership of the test environment
  ::testing::AddGlobalTestEnvironment(new onnxruntime::test::TestEnvironment{});
  int status = RUN_ALL_TESTS();
  return status;
}
