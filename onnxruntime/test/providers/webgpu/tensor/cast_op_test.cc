// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#include "core/framework/to_tensor_proto_element_type.h"

namespace onnxruntime {
namespace test {

// bool -> uint8 Cast, run explicitly on the WebGPU EP (e.g. a terminal bool->uint8 cast at a graph
// output). WebGPU stores uint8 packed (4 per u32); the Cast writes it via the Uint8x4 SetByOffset
// packing. The 6-element shape deliberately isn't a multiple of 4, exercising the final partial
// packed word.
TEST(CastOpTest, BoolToUint8_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  const std::vector<int64_t> dims{2, 3};
  OpTester test("Cast", 13);
  test.AddAttribute<int64_t>("to", utils::ToTensorProtoElementType<uint8_t>());
  test.AddInput<bool>("input", dims, {true, false, true, false, true, true});
  test.AddOutput<uint8_t>("output", dims, {1, 0, 1, 0, 1, 1});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// uint8 -> int32 Cast, run explicitly on the WebGPU EP. WebGPU stores uint8 packed (4 per u32);
// reading it unpacks via the Uint8x4 GetByOffset. Values span the full byte range (0, 1, 128, 255)
// to prove each byte is unpacked correctly, and the 6-element shape isn't a multiple of 4 so the
// final partial packed word is exercised.
TEST(CastOpTest, Uint8ToInt32_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  const std::vector<int64_t> dims{2, 3};
  OpTester test("Cast", 13);
  test.AddAttribute<int64_t>("to", utils::ToTensorProtoElementType<int32_t>());
  test.AddInput<uint8_t>("input", dims, {0, 1, 128, 255, 42, 7});
  test.AddOutput<int32_t>("output", dims, {0, 1, 128, 255, 42, 7});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// uint8 -> float Cast on the WebGPU EP, exercising the uint8-input read on the float conversion path.
TEST(CastOpTest, Uint8ToFloat_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  const std::vector<int64_t> dims{2, 3};
  OpTester test("Cast", 13);
  test.AddAttribute<int64_t>("to", utils::ToTensorProtoElementType<float>());
  test.AddInput<uint8_t>("input", dims, {0, 1, 128, 255, 42, 7});
  test.AddOutput<float>("output", dims, {0.0f, 1.0f, 128.0f, 255.0f, 42.0f, 7.0f});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

namespace {
// Disables CPU fallback so a cast the WebGPU EP doesn't claim fails instead of silently running on CPU.
SessionOptions WebGpuOnlySessionOptions() {
  SessionOptions options;
  ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  return options;
}

// Takes initializer lists because std::vector<bool> cannot provide contiguous data.
template <typename From, typename To>
void RunCastOnWebGpu(std::initializer_list<From> input, std::initializer_list<To> expected) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }
  // 7 elements, so the final packed int8 word is partial.
  const std::vector<int64_t> dims{static_cast<int64_t>(input.size())};
  OpTester test("Cast", 13);
  test.AddAttribute<int64_t>("to", utils::ToTensorProtoElementType<To>());
  test.AddInput<From>("input", dims, input);
  test.AddOutput<To>("output", dims, expected);
  test.Config(WebGpuOnlySessionOptions()).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}
}  // namespace

// int8 is stored packed (4 per u32). Reading sign-extends via unpack4xI8; writing keeps each lane's
// low byte via the Int8x4 SetByOffset.
TEST(CastOpTest, Int8ToOtherTypes_WebGpu) {
  RunCastOnWebGpu<int8_t, float>({-128, -100, -1, 0, 1, 99, 127}, {-128.0f, -100.0f, -1.0f, 0.0f, 1.0f, 99.0f, 127.0f});
  RunCastOnWebGpu<int8_t, MLFloat16>({-128, -100, -1, 0, 1, 99, 127}, {MLFloat16(-128.0f), MLFloat16(-100.0f), MLFloat16(-1.0f),
                                                                       MLFloat16(0.0f), MLFloat16(1.0f), MLFloat16(99.0f),
                                                                       MLFloat16(127.0f)});
  RunCastOnWebGpu<int8_t, int32_t>({-128, -100, -1, 0, 1, 99, 127}, {-128, -100, -1, 0, 1, 99, 127});
  RunCastOnWebGpu<int8_t, int64_t>({-128, -100, -1, 0, 1, 99, 127}, {-128, -100, -1, 0, 1, 99, 127});
  RunCastOnWebGpu<int8_t, uint8_t>({-128, -100, -1, 0, 1, 99, 127}, {128, 156, 255, 0, 1, 99, 127});
  RunCastOnWebGpu<int8_t, bool>({-128, -100, -1, 0, 1, 99, 127}, {true, true, true, false, true, true, true});
  RunCastOnWebGpu<int8_t, int8_t>({-128, -100, -1, 0, 1, 99, 127}, {-128, -100, -1, 0, 1, 99, 127});
}

TEST(CastOpTest, OtherTypesToInt8_WebGpu) {
  RunCastOnWebGpu<float, int8_t>({-128.0f, -100.75f, -1.0f, 0.0f, 1.5f, 99.0f, 127.0f}, {-128, -100, -1, 0, 1, 99, 127});
  RunCastOnWebGpu<MLFloat16, int8_t>({MLFloat16(-128.0f), MLFloat16(-100.5f), MLFloat16(-1.0f), MLFloat16(0.0f),
                                      MLFloat16(1.5f), MLFloat16(99.0f), MLFloat16(127.0f)},
                                     {-128, -100, -1, 0, 1, 99, 127});
  RunCastOnWebGpu<int32_t, int8_t>({-128, -100, -1, 0, 1, 99, 127}, {-128, -100, -1, 0, 1, 99, 127});
  RunCastOnWebGpu<uint8_t, int8_t>({128, 156, 255, 0, 1, 99, 127}, {-128, -100, -1, 0, 1, 99, 127});
  RunCastOnWebGpu<bool, int8_t>({true, false, true, false, true, true, false}, {1, 0, 1, 0, 1, 1, 0});
}

// uint8 -> bool Cast on the WebGPU EP. Exercises the distinct Boolx4 output-packing path (vs. the
// direct write used for int32/float above) and the standard cast-to-bool semantics (any nonzero
// byte -> true) -- the natural inverse of the bool->uint8 case.
TEST(CastOpTest, Uint8ToBool_WebGpu) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available in this build.";
  }

  const std::vector<int64_t> dims{2, 3};
  OpTester test("Cast", 13);
  test.AddAttribute<int64_t>("to", utils::ToTensorProtoElementType<bool>());
  test.AddInput<uint8_t>("input", dims, {0, 1, 128, 255, 0, 7});
  test.AddOutput<bool>("output", dims, {false, true, true, true, false, true});

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(webgpu_ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace test
}  // namespace onnxruntime
