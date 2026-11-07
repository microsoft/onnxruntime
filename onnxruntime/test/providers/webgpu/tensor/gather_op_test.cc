// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

#ifdef USE_WEBGPU
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/providers/webgpu/webgpu_provider_options.h"
#endif

namespace onnxruntime {
namespace test {

#ifdef USE_WEBGPU
// Validates the WebGPU Gather kernel's packed-byte path for uint8 (bit-shift/mask read,
// OR-shift assembly write). Without this test, a non-WebGPU build would only exercise
// the CPU EP path and miss shader regressions.
TEST(GatherOpTest, Gather_axis1_indices2d_uint8_webgpu) {
  if (DefaultWebGpuExecutionProvider().get() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<uint8_t>("data", {3, 3},
                         {10, 20, 30,
                          40, 50, 60,
                          70, 80, 90});
  test.AddInput<int32_t>("indices", {2, 2},
                         {1, 0,
                          2, 1});
  test.AddOutput<uint8_t>("output", {3, 2, 2},
                          {20, 10, 30, 20,
                           50, 40, 60, 50,
                           80, 70, 90, 80});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Validates the WebGPU uint8 Gather kernel with output_size NOT a multiple of 4.
// output shape {3,3} = 9 elements: the last packed u32 word holds only 1 valid byte,
// with the remaining 3 bytes zero-padded (partial-thread boundary path).
TEST(GatherOpTest, Gather_axis0_uint8_non_multiple_of_4_webgpu) {
  if (DefaultWebGpuExecutionProvider().get() == nullptr) {
    GTEST_SKIP() << "WebGPU EP not available";
  }
  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 0LL);
  // 3×3 input with byte-diverse values
  test.AddInput<uint8_t>("data", {3, 3},
                         {10, 20, 30,
                          40, 50, 60,
                          70, 80, 90});
  // 1-D indices of length 3: gather rows 1, 2, 0
  test.AddInput<int32_t>("indices", {3}, {1, 2, 0});
  // output shape {3,3} = 9 bytes (not a multiple of 4)
  test.AddOutput<uint8_t>("output", {3, 3},
                          {40, 50, 60,
                           70, 80, 90,
                           10, 20, 30});
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultWebGpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// int64 Gather is gated behind the enableInt64 provider option. Disable CPU-EP fallback so the
// Gather node must run on the WebGPU kernel, and include values outside the int32 range (2^32,
// 2^33+1, a large negative, INT64_MAX) to prove the full 64-bit value is preserved via a raw
// vec2<u32> storage copy rather than truncated to i32.
TEST(GatherOpTest, Gather_int64_webgpu) {
  ConfigOptions provider_options{};
  ASSERT_STATUS_OK(provider_options.AddConfigEntry(webgpu::options::kEnableInt64, "1"));
  auto provider = WebGpuExecutionProviderWithOptions(provider_options);
  if (provider == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }

  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<int64_t>("data", {3, 2},
                         {1, 4294967296,                 // 1, 2^32
                          8589934593, -4294967297,       // 2^33+1, -(2^32+1)
                          100, 9223372036854775807LL});  // 100, INT64_MAX
  test.AddInput<int32_t>("indices", {2}, {2, 0});
  test.AddOutput<int64_t>("output", {2, 2},
                          {100, 9223372036854775807LL,
                           1, 4294967296});

  // Disable CPU-EP fallback so the Gather node must run on the WebGPU kernel.
  SessionOptions so;
  ASSERT_STATUS_OK(so.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  test.Config(so)
      .ConfigEp(std::move(provider))
      .RunWithConfig();
}

namespace {
// Disables CPU fallback so a Gather the WebGPU EP doesn't claim fails instead of silently running on CPU.
SessionOptions WebGpuOnlySessionOptions() {
  SessionOptions options;
  ORT_THROW_IF_ERROR(options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
  return options;
}

template <typename TIndex>
void RunInt8GatherOnWebGpu(int opset, int64_t axis,
                           const std::vector<int64_t>& data_dims, const std::vector<int8_t>& data,
                           const std::vector<int64_t>& indices_dims, const std::vector<TIndex>& indices,
                           const std::vector<int64_t>& output_dims, const std::vector<int8_t>& expected) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (webgpu_ep == nullptr) {
    GTEST_SKIP() << "WebGPU EP is not available";
  }
  OpTester test("Gather", opset);
  test.AddAttribute<int64_t>("axis", axis);
  test.AddInput<int8_t>("data", data_dims, data);
  test.AddInput<TIndex>("indices", indices_dims, indices);
  test.AddOutput<int8_t>("output", output_dims, expected);
  test.Config(WebGpuOnlySessionOptions()).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

// 3x3 int8 table with the extremes and other negative values, so a byte read with the wrong
// signedness or packed into the wrong lane changes the result.
const std::vector<int8_t> kInt8Table = {-128, -1, 0,
                                        127, 1, -2,
                                        100, -100, -64};

// Builds table -> Gather(indices_i) -> Cast(float) -> output_i for each indices input, mirroring
// WebNN's gather-cast-int8-table conformance graphs, including two Gathers sharing one table.
class Int8GatherCastTester final : public OpTester {
 public:
  Int8GatherCastTester() : OpTester("Gather", 13) {}

  void AddNodes(Graph& graph, std::vector<NodeArg*>& inputs, std::vector<NodeArg*>& outputs,
                std::vector<std::function<void(Node&)>>& add_attribute_funcs) override {
    ONNX_NAMESPACE::TypeProto gathered_type;
    gathered_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT8);
    for (size_t i = 1; i < inputs.size(); ++i) {
      const std::string suffix = std::to_string(i);
      auto& gathered = graph.GetOrCreateNodeArg("gathered_" + suffix, &gathered_type);
      auto& gather = graph.AddNode("gather_" + suffix, "Gather", "", {inputs[0], inputs[i]}, {&gathered});
      for (auto& add_attribute : add_attribute_funcs) {
        add_attribute(gather);
      }
      auto& cast = graph.AddNode("cast_" + suffix, "Cast", "", {&gathered}, {outputs[i - 1]});
      cast.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    }
  }
};
}  // namespace

// int8 shares the packed-byte (4 per u32) path with uint8 and is copied as raw bytes. The 9-element
// output leaves the last packed word partial, and the negative index exercises index wrapping.
TEST(GatherOpTest, Gather_int8_axis0_webgpu) {
  const std::vector<int8_t> expected = {100, -100, -64,
                                        -128, -1, 0,
                                        127, 1, -2};
  RunInt8GatherOnWebGpu<int32_t>(13, 0, {3, 3}, kInt8Table, {3}, {2, -3, 1}, {3, 3}, expected);
  RunInt8GatherOnWebGpu<int64_t>(13, 0, {3, 3}, kInt8Table, {3}, {2, -3, 1}, {3, 3}, expected);
}

TEST(GatherOpTest, Gather_int8_axis1_webgpu) {
  // 2-D indices; output {3, 1, 3} has 9 elements.
  const std::vector<int8_t> expected_partial = {0, -128, -1,
                                                -2, 127, 1,
                                                -64, 100, -100};
  RunInt8GatherOnWebGpu<int32_t>(13, 1, {3, 3}, kInt8Table, {1, 3}, {-1, 0, -2}, {3, 1, 3}, expected_partial);
  RunInt8GatherOnWebGpu<int64_t>(13, 1, {3, 3}, kInt8Table, {1, 3}, {-1, 0, -2}, {3, 1, 3}, expected_partial);

  // Output {3, 2, 2} fills exactly three packed words.
  const std::vector<int8_t> expected_full = {-1, -128, 0, -1,
                                             1, 127, -2, 1,
                                             -100, 100, -64, -100};
  RunInt8GatherOnWebGpu<int32_t>(13, 1, {3, 3}, kInt8Table, {2, 2}, {1, 0, -1, 1}, {3, 2, 2}, expected_full);
  RunInt8GatherOnWebGpu<int64_t>(13, 1, {3, 3}, kInt8Table, {2, 2}, {1, 0, -1, 1}, {3, 2, 2}, expected_full);
}

TEST(GatherOpTest, Gather_int8_scalar_indices_webgpu) {
  // 1-D table with a 0-D index produces a 0-D output.
  RunInt8GatherOnWebGpu<int64_t>(13, 0, {5}, {-128, -1, 0, 1, 127}, {}, {-5}, {}, {-128});
  RunInt8GatherOnWebGpu<int32_t>(13, 0, {5}, {-128, -1, 0, 1, 127}, {}, {4}, {}, {127});
  // 0-D index on axis 1 squeezes that axis.
  RunInt8GatherOnWebGpu<int64_t>(13, 1, {3, 3}, kInt8Table, {}, {-1}, {3}, {0, -2, -64});
  RunInt8GatherOnWebGpu<int32_t>(13, 0, {3, 3}, kInt8Table, {}, {1}, {3}, {127, 1, -2});
}

// Gather is registered for opsets 1-10, 11-12 and 13+; int8 must be claimed by each registration.
TEST(GatherOpTest, Gather_int8_all_opsets_webgpu) {
  for (int opset : {7, 11, 13}) {
    SCOPED_TRACE(opset);
    RunInt8GatherOnWebGpu<int64_t>(opset, 0, {3, 3}, kInt8Table, {2}, {2, 0}, {2, 3},
                                   {100, -100, -64, -128, -1, 0});
  }
}

// Reads every int8 value through the packed path. 257 negative indices walk the table backwards and
// wrap once, so the output ends on a partial packed word.
TEST(GatherOpTest, Gather_int8_all_byte_values_webgpu) {
  std::vector<int8_t> data(256);
  for (int i = 0; i < 256; ++i) {
    data[i] = static_cast<int8_t>(i - 128);
  }
  std::vector<int32_t> indices(257);
  std::vector<int8_t> expected(257);
  for (int k = 0; k < 257; ++k) {
    indices[k] = -1 - (k % 256);
    expected[k] = data[255 - (k % 256)];
  }
  RunInt8GatherOnWebGpu<int32_t>(13, 0, {256}, data, {257}, indices, {257}, expected);
}

// Mirrors WebNN's gather-cast-int8-table conformance tests: an int8 table, as an initializer or a
// runtime input, feeds Gather (2-D and 0-D indices, two Gathers on one table) followed by a Cast to
// float. With CPU fallback disabled the whole graph must run on WebGPU, and the float output catches
// a table read as unsigned (-128 would become 128).
TEST(GatherOpTest, Gather_int8_then_cast_to_float_webgpu) {
  for (bool table_is_initializer : {false, true}) {
    SCOPED_TRACE(table_is_initializer ? "initializer table" : "runtime table");
    auto webgpu_ep = DefaultWebGpuExecutionProvider();
    if (webgpu_ep == nullptr) {
      GTEST_SKIP() << "WebGPU EP is not available";
    }
    Int8GatherCastTester test;
    test.AddAttribute<int64_t>("axis", 0LL);
    test.AddInput<int8_t>("table", {5}, {-128, -1, 0, 1, 127}, table_is_initializer);
    test.AddInput<int64_t>("indices", {2, 3}, {0, 4, 1, -1, 2, -5});
    test.AddInput<int32_t>("scalar_index", {}, {-5});
    test.AddOutput<float>("output", {2, 3}, {-128.0f, 127.0f, -1.0f, 127.0f, 0.0f, -128.0f});
    test.AddOutput<float>("scalar_output", {}, {-128.0f});
    test.Config(WebGpuOnlySessionOptions()).ConfigEp(std::move(webgpu_ep)).RunWithConfig();
  }
}
#endif  // USE_WEBGPU

}  // namespace test
}  // namespace onnxruntime
