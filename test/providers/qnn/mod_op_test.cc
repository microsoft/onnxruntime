// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <cassert>
#include <string>

#include "test/providers/qnn/qnn_test_utils.h"
#include "core/graph/node_attr_utils.h"

#include "core/graph/onnx_protobuf.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

// Runs a model with a Mod operator on the QNN CPU backend. Checks the graph node assignment
// and that inference outputs for QNN EP and CPU EP match.
template <typename DataType>
static void RunModTest(const std::vector<TestInputDef<DataType>>& input_defs,
                       const std::vector<ONNX_NAMESPACE::AttributeProto>& attrs,
                       ExpectedEPNodeAssignment expected_ep_assignment,
                       const std::string& backend_name = "cpu",
                       int opset = 13) {
  ProviderOptions provider_options;

  provider_options["backend_type"] = backend_name;
  provider_options["offload_graph_io_quantization"] = "0";

  RunQnnModelTest(BuildOpTestCase<DataType>("Mod", input_defs, {}, attrs),
                  provider_options,
                  opset,
                  expected_ep_assignment);
}

//
// CPU tests:
//

// Test that Mod with dynamic divisor.
TEST_F(QnnCPUBackendTests, Mod_dynamic_Divisor) {
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> dividend_shape{1, 4, 5};
  auto dividend = rand_gen_.Uniform<int32_t>(dividend_shape, -100.0f, 100.0f);

  const std::vector<int64_t> divisor_shape{1, 5};
  auto divisor = rand_gen_.Uniform<int32_t>(divisor_shape, 1.0f, 10.0f);
  RunModTest<int32_t>({TestInputDef<int32_t>({1, 4, 5}, false, dividend),
                       TestInputDef<int32_t>({1, 5}, false, divisor)},
                      {},
                      ExpectedEPNodeAssignment::All);

  // Test negative divisor
  auto neg_divisor = rand_gen_.Uniform<int32_t>(divisor_shape, -10.0f, -1.0f);
  RunModTest<int32_t>({TestInputDef<int32_t>({1, 4, 5}, false, dividend),
                       TestInputDef<int32_t>({1, 5}, false, neg_divisor)},
                      {},
                      ExpectedEPNodeAssignment::All);
}

// Test that Mod with static divisor.
TEST_F(QnnCPUBackendTests, Mod_static_Divisor) {
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> dividend_shape{1, 4, 5};
  auto dividend = rand_gen_.Uniform<int32_t>(dividend_shape, -100.0f, 100.0f);

  const std::vector<int64_t> divisor_shape{1, 5};
  auto divisor = rand_gen_.Uniform<int32_t>(divisor_shape, 1.0f, 10.0f);

  RunModTest<int32_t>({TestInputDef<int32_t>({1, 4, 5}, false, dividend),
                       TestInputDef<int32_t>({1, 5}, true, divisor)},
                      {},
                      ExpectedEPNodeAssignment::All);

  // Test negative divisor
  auto neg_divisor = rand_gen_.Uniform<int32_t>(divisor_shape, -10.0f, -1.0f);
  RunModTest<int32_t>({TestInputDef<int32_t>({1, 4, 5}, false, dividend),
                       TestInputDef<int32_t>({1, 5}, true, neg_divisor)},
                      {},
                      ExpectedEPNodeAssignment::All);
}
#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
//
// HTP tests:
//

// Test that Mod with dynamic divisor.
TEST_F(QnnHTPBackendTests, Mod_dynamic_Divisor) {
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> dividend_shape{1, 4, 5};
  auto dividend = rand_gen_.Uniform<int64_t>(dividend_shape, -100.0f, 100.0f);

  const std::vector<int64_t> divisor_shape{1, 5};
  auto divisor = rand_gen_.Uniform<int64_t>(divisor_shape, 1.0f, 10.0f);

  RunModTest<int64_t>({TestInputDef<int64_t>({1, 4, 5}, false, dividend),
                       TestInputDef<int64_t>({1, 5}, false, divisor)},
                      {},
                      ExpectedEPNodeAssignment::All,
                      "htp");

  // Test negative divisor
  auto neg_divisor = rand_gen_.Uniform<int64_t>(divisor_shape, -10.0f, -1.0f);
  RunModTest<int64_t>({TestInputDef<int64_t>({1, 4, 5}, false, dividend),
                       TestInputDef<int64_t>({1, 5}, false, neg_divisor)},
                      {},
                      ExpectedEPNodeAssignment::All,
                      "htp");
}

// Test that Mod with static divisor.
TEST_F(QnnHTPBackendTests, Mod_static_Divisor) {
  RandomValueGenerator rand_gen_{optional<RandomValueGenerator::RandomSeedType>{2345}};
  const std::vector<int64_t> dividend_shape{1, 4, 5};
  auto dividend = rand_gen_.Uniform<int64_t>(dividend_shape, -100.0f, 100.0f);

  const std::vector<int64_t> divisor_shape{1, 5};
  auto divisor = rand_gen_.Uniform<int64_t>(divisor_shape, 1.0f, 10.0f);

  RunModTest<int64_t>({TestInputDef<int64_t>({1, 4, 5}, false, dividend),
                       TestInputDef<int64_t>({1, 5}, true, divisor)},
                      {},
                      ExpectedEPNodeAssignment::All,
                      "htp");

  // Test negative divisor
  auto neg_divisor = rand_gen_.Uniform<int64_t>(divisor_shape, -10.0f, -1.0f);
  RunModTest<int64_t>({TestInputDef<int64_t>({1, 4, 5}, false, dividend),
                       TestInputDef<int64_t>({1, 5}, true, neg_divisor)},
                      {},
                      ExpectedEPNodeAssignment::All,
                      "htp");
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime
#endif  // !defined(ORT_MINIMAL_BUILD)
