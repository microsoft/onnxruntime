// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <cmath>
#include <vector>
#include "gtest/gtest.h"
#include "core/common/logging/logging.h"
#include "core/framework/kernel_registry.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {
enum class TensorType {
  kFloat,
  kFloat16
};

// Reference implementation for CausalConvWithState
// Performs depthwise causal 1D convolution with optional state, bias, and activation.
//
// Input: (B, D, L) channels-first
// Weight: (D, 1, K) depthwise
// Bias: (D,) optional
// past_state: (B, D, K-1) optional carry state
//
// Output: (B, D, L) convolution output (with optional activation)
// present_state: (B, D, K-1) updated carry state
void CausalConvWithStateReference(
    const std::vector<float>& input,
    const std::vector<float>& weight,
    const std::vector<float>* bias,
    const std::vector<float>* conv_state,
    std::vector<float>& output,
    std::vector<float>& present_state,
    int batch_size,
    int channels,
    int input_length,
    int kernel_size,
    const std::string& activation) {
  int state_length = kernel_size - 1;
  int total_virtual_length = state_length + input_length;

  output.resize(batch_size * channels * input_length);
  present_state.resize(batch_size * channels * state_length);

  for (int b = 0; b < batch_size; ++b) {
    for (int d = 0; d < channels; ++d) {
      int bd = b * channels + d;

      // Build virtual input: [conv_state, input]
      std::vector<float> virtual_input(total_virtual_length, 0.0f);
      if (conv_state != nullptr) {
        for (int s = 0; s < state_length; ++s) {
          virtual_input[s] = (*conv_state)[bd * state_length + s];
        }
      }
      for (int l = 0; l < input_length; ++l) {
        virtual_input[state_length + l] = input[bd * input_length + l];
      }

      // Compute depthwise convolution
      for (int pos = 0; pos < input_length; ++pos) {
        float acc = 0.0f;
        for (int j = 0; j < kernel_size; ++j) {
          float val = virtual_input[pos + j];
          float w = weight[d * kernel_size + j];
          acc += val * w;
        }
        // Add bias
        if (bias != nullptr) {
          acc += (*bias)[d];
        }
        // Apply activation
        if (activation == "silu" || activation == "swish") {
          acc = acc / (1.0f + std::exp(-acc));
        }
        output[bd * input_length + pos] = acc;
      }

      // Compute present_state: last state_length values from virtual input
      for (int s = 0; s < state_length; ++s) {
        present_state[bd * state_length + s] =
            virtual_input[input_length + s];
      }
    }
  }
}

// Returns a WebGPU EP if it is available and has the CausalConvWithState kernel registered,
// or nullptr otherwise.
std::unique_ptr<IExecutionProvider> TryGetEpWithCausalConvWithState() {
  auto ep = DefaultWebGpuExecutionProvider();
  if (!ep) {
    ep = DefaultCpuExecutionProvider();
  }

  auto kernel_registry = ep->GetKernelRegistry();
  if (kernel_registry) {
    const KernelCreateInfo* info = nullptr;
    KernelRegistry::TypeConstraintMap type_constraints;
    auto status = kernel_registry->TryFindKernel(
        ep->Type(), "CausalConvWithState", kMSDomain, 1,
        type_constraints, DefaultLoggingManager().DefaultLogger(), &info);
    if (!status.IsOK()) return nullptr;
  }
  return ep;
}

}  // anonymous namespace

static void RunCausalConvWithStateTest(
    const std::vector<float>& input_data,
    const std::vector<float>& weight_data,
    const std::vector<float>* bias_data,
    const std::vector<float>* conv_state_data,
    const std::vector<float>& expected_output,
    const std::vector<float>& expected_state,
    int batch_size,
    int channels,
    int input_length,
    int kernel_size,
    const std::string& activation,
    TensorType tensor_type) {
  auto ep = TryGetEpWithCausalConvWithState();
  if (!ep) {
    GTEST_SKIP() << "CausalConvWithState kernel not registered";
    return;
  }

  int state_length = kernel_size - 1;

  std::vector<int64_t> input_shape = {batch_size, channels, input_length};
  std::vector<int64_t> weight_shape = {channels, 1, kernel_size};
  std::vector<int64_t> bias_shape = {channels};
  std::vector<int64_t> state_shape = {batch_size, channels, state_length};
  std::vector<int64_t> output_shape = {batch_size, channels, input_length};

  {
    OpTester test("CausalConvWithState", 1, onnxruntime::kMSDomain);
    test.AddAttribute<std::string>("activation", activation);

    if (tensor_type == TensorType::kFloat) {
      test.AddInput<float>("input", input_shape, input_data);
      test.AddInput<float>("weight", weight_shape, weight_data);

      if (bias_data != nullptr) {
        test.AddInput<float>("bias", bias_shape, *bias_data);
      } else {
        test.AddOptionalInputEdge<float>();
      }

      if (conv_state_data != nullptr) {
        test.AddInput<float>("past_state", state_shape, *conv_state_data);
      } else {
        test.AddOptionalInputEdge<float>();
      }

      test.AddOutput<float>("output", output_shape, expected_output);
      test.AddOutput<float>("present_state", state_shape, expected_state);
    } else {
      test.AddInput<MLFloat16>("input", input_shape, ToFloat16(input_data));
      test.AddInput<MLFloat16>("weight", weight_shape, ToFloat16(weight_data));

      if (bias_data != nullptr) {
        test.AddInput<MLFloat16>("bias", bias_shape, ToFloat16(*bias_data));
      } else {
        test.AddOptionalInputEdge<MLFloat16>();
      }

      if (conv_state_data != nullptr) {
        test.AddInput<MLFloat16>("past_state", state_shape, ToFloat16(*conv_state_data));
      } else {
        test.AddOptionalInputEdge<MLFloat16>();
      }

      test.AddOutput<MLFloat16>("output", output_shape, ToFloat16(expected_output));
      test.AddOutput<MLFloat16>("present_state", state_shape, ToFloat16(expected_state));
    }

    test.SetOutputAbsErr("output", 0.01f);
    test.SetOutputAbsErr("present_state", 0.01f);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(std::move(ep));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

static void RunCausalConvWithStateTests(
    const std::vector<float>& input_data,
    const std::vector<float>& weight_data,
    const std::vector<float>* bias_data,
    const std::vector<float>* conv_state_data,
    int batch_size,
    int channels,
    int input_length,
    int kernel_size,
    const std::string& activation = "silu") {
  // Compute expected output using reference implementation
  std::vector<float> expected_output;
  std::vector<float> expected_state;
  CausalConvWithStateReference(
      input_data, weight_data, bias_data, conv_state_data,
      expected_output, expected_state,
      batch_size, channels, input_length, kernel_size, activation);

  // FP32 test
  RunCausalConvWithStateTest(
      input_data, weight_data, bias_data, conv_state_data,
      expected_output, expected_state,
      batch_size, channels, input_length, kernel_size, activation,
      TensorType::kFloat);

  // FP16 test
  RunCausalConvWithStateTest(
      input_data, weight_data, bias_data, conv_state_data,
      expected_output, expected_state,
      batch_size, channels, input_length, kernel_size, activation,
      TensorType::kFloat16);
}

// =============================================================================
// Basic tests - simple cases
// =============================================================================

TEST(CausalConvWithStateTest, BasicNoStateNoBias) {
  // B=1, D=2, L=4, K=3, activation=none
  int batch_size = 1, channels = 2, input_length = 4, kernel_size = 3;

  // Input: (1, 2, 4)
  std::vector<float> input_data = {
      1.0f, 2.0f, 3.0f, 4.0f,   // channel 0
      0.5f, 1.5f, 2.5f, 3.5f};  // channel 1

  // Weight: (2, 1, 3)
  std::vector<float> weight_data = {
      0.1f, 0.2f, 0.3f,   // channel 0 kernel
      0.4f, 0.5f, 0.6f};  // channel 1 kernel

  RunCausalConvWithStateTests(
      input_data, weight_data, nullptr, nullptr,
      batch_size, channels, input_length, kernel_size, "none");
}

TEST(CausalConvWithStateTest, BasicWithBias) {
  // B=1, D=2, L=4, K=3, activation=none
  int batch_size = 1, channels = 2, input_length = 4, kernel_size = 3;

  std::vector<float> input_data = {
      1.0f, 2.0f, 3.0f, 4.0f,
      0.5f, 1.5f, 2.5f, 3.5f};
  std::vector<float> weight_data = {
      0.1f, 0.2f, 0.3f,
      0.4f, 0.5f, 0.6f};
  std::vector<float> bias_data = {0.1f, -0.2f};

  RunCausalConvWithStateTests(
      input_data, weight_data, &bias_data, nullptr,
      batch_size, channels, input_length, kernel_size, "none");
}

TEST(CausalConvWithStateTest, BasicWithState) {
  // B=1, D=2, L=3, K=3, activation=none
  int batch_size = 1, channels = 2, input_length = 3, kernel_size = 3;

  std::vector<float> input_data = {
      1.0f, 2.0f, 3.0f,
      0.5f, 1.5f, 2.5f};
  std::vector<float> weight_data = {
      0.1f, 0.2f, 0.3f,
      0.4f, 0.5f, 0.6f};
  // State: (1, 2, 2) - kernel_size - 1 = 2
  std::vector<float> conv_state_data = {
      -1.0f, 0.5f,   // channel 0 state
      0.3f, -0.7f};  // channel 1 state

  RunCausalConvWithStateTests(
      input_data, weight_data, nullptr, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "none");
}

TEST(CausalConvWithStateTest, WithStateAndBias) {
  // B=1, D=2, L=3, K=3, activation=none
  int batch_size = 1, channels = 2, input_length = 3, kernel_size = 3;

  std::vector<float> input_data = {
      1.0f, 2.0f, 3.0f,
      0.5f, 1.5f, 2.5f};
  std::vector<float> weight_data = {
      0.1f, 0.2f, 0.3f,
      0.4f, 0.5f, 0.6f};
  std::vector<float> bias_data = {0.1f, -0.2f};
  std::vector<float> conv_state_data = {
      -1.0f, 0.5f,
      0.3f, -0.7f};

  RunCausalConvWithStateTests(
      input_data, weight_data, &bias_data, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "none");
}

// =============================================================================
// SiLU activation tests
// =============================================================================

TEST(CausalConvWithStateTest, SiluActivationNoState) {
  int batch_size = 1, channels = 2, input_length = 4, kernel_size = 3;

  std::vector<float> input_data = {
      1.0f, 2.0f, 3.0f, 4.0f,
      0.5f, 1.5f, 2.5f, 3.5f};
  std::vector<float> weight_data = {
      0.1f, 0.2f, 0.3f,
      0.4f, 0.5f, 0.6f};

  RunCausalConvWithStateTests(
      input_data, weight_data, nullptr, nullptr,
      batch_size, channels, input_length, kernel_size, "silu");
}

TEST(CausalConvWithStateTest, SiluActivationWithState) {
  int batch_size = 1, channels = 2, input_length = 3, kernel_size = 3;

  std::vector<float> input_data = {
      1.0f, 2.0f, 3.0f,
      0.5f, 1.5f, 2.5f};
  std::vector<float> weight_data = {
      0.1f, 0.2f, 0.3f,
      0.4f, 0.5f, 0.6f};
  std::vector<float> conv_state_data = {
      -1.0f, 0.5f,
      0.3f, -0.7f};

  RunCausalConvWithStateTests(
      input_data, weight_data, nullptr, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "silu");
}

TEST(CausalConvWithStateTest, SiluActivationWithBiasAndState) {
  int batch_size = 1, channels = 2, input_length = 4, kernel_size = 3;

  std::vector<float> input_data = {
      1.0f, 2.0f, 3.0f, 4.0f,
      0.5f, 1.5f, 2.5f, 3.5f};
  std::vector<float> weight_data = {
      0.1f, 0.2f, 0.3f,
      0.4f, 0.5f, 0.6f};
  std::vector<float> bias_data = {0.1f, -0.2f};
  std::vector<float> conv_state_data = {
      -1.0f, 0.5f,
      0.3f, -0.7f};

  RunCausalConvWithStateTests(
      input_data, weight_data, &bias_data, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "silu");
}

// =============================================================================
// Kernel size variations
// =============================================================================

TEST(CausalConvWithStateTest, KernelSize2) {
  int batch_size = 1, channels = 2, input_length = 4, kernel_size = 2;

  std::vector<float> input_data = {
      1.0f, 2.0f, 3.0f, 4.0f,
      0.5f, 1.5f, 2.5f, 3.5f};
  std::vector<float> weight_data = {
      0.3f, 0.7f,
      0.4f, 0.6f};
  // State: (1, 2, 1) - kernel_size - 1 = 1
  std::vector<float> conv_state_data = {0.5f, -0.3f};

  RunCausalConvWithStateTests(
      input_data, weight_data, nullptr, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "silu");
}

TEST(CausalConvWithStateTest, KernelSize4) {
  int batch_size = 1, channels = 1, input_length = 5, kernel_size = 4;

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> weight_data = {0.1f, 0.2f, 0.3f, 0.4f};
  // State: (1, 1, 3)
  std::vector<float> conv_state_data = {-1.0f, 0.0f, 0.5f};

  RunCausalConvWithStateTests(
      input_data, weight_data, nullptr, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "none");
}

// =============================================================================
// Batch size > 1
// =============================================================================

TEST(CausalConvWithStateTest, MultiBatch) {
  int batch_size = 2, channels = 2, input_length = 3, kernel_size = 3;

  // Input: (2, 2, 3)
  std::vector<float> input_data = {
      // Batch 0
      1.0f, 2.0f, 3.0f,  // ch 0
      0.5f, 1.5f, 2.5f,  // ch 1
      // Batch 1
      -1.0f, 0.0f, 1.0f,  // ch 0
      0.2f, 0.4f, 0.6f};  // ch 1

  std::vector<float> weight_data = {
      0.1f, 0.2f, 0.3f,
      0.4f, 0.5f, 0.6f};

  std::vector<float> bias_data = {0.1f, -0.1f};

  // State: (2, 2, 2)
  std::vector<float> conv_state_data = {
      // Batch 0
      -0.5f, 0.5f,  // ch 0
      0.3f, -0.3f,  // ch 1
      // Batch 1
      0.1f, -0.1f,  // ch 0
      0.7f, 0.8f};  // ch 1

  RunCausalConvWithStateTests(
      input_data, weight_data, &bias_data, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "silu");
}

// =============================================================================
// Single token decode (L=1) - the primary use case for incremental decoding
// =============================================================================

TEST(CausalConvWithStateTest, SingleTokenDecode) {
  int batch_size = 1, channels = 4, input_length = 1, kernel_size = 4;

  // Input: (1, 4, 1)
  std::vector<float> input_data = {0.5f, -0.3f, 1.2f, 0.8f};

  // Weight: (4, 1, 4)
  std::vector<float> weight_data = {
      0.1f, 0.2f, 0.3f, 0.4f,
      0.5f, 0.6f, 0.7f, 0.8f,
      -0.1f, -0.2f, 0.1f, 0.2f,
      0.3f, 0.3f, 0.3f, 0.3f};

  std::vector<float> bias_data = {0.0f, 0.1f, -0.1f, 0.0f};

  // State: (1, 4, 3) - carrying the last 3 values per channel
  std::vector<float> conv_state_data = {
      1.0f, 2.0f, 3.0f,     // ch 0
      -1.0f, 0.0f, 1.0f,    // ch 1
      0.5f, 0.5f, 0.5f,     // ch 2
      -0.2f, 0.4f, -0.6f};  // ch 3

  RunCausalConvWithStateTests(
      input_data, weight_data, &bias_data, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "silu");
}

TEST(CausalConvWithStateTest, SingleTokenDecodeMultiBatch) {
  int batch_size = 2, channels = 2, input_length = 1, kernel_size = 3;

  // Input: (2, 2, 1)
  std::vector<float> input_data = {
      0.5f,   // B0, ch 0
      -0.3f,  // B0, ch 1
      1.2f,   // B1, ch 0
      0.8f};  // B1, ch 1

  std::vector<float> weight_data = {
      0.1f, 0.2f, 0.3f,
      0.4f, 0.5f, 0.6f};

  // State: (2, 2, 2)
  std::vector<float> conv_state_data = {
      1.0f, 2.0f,    // B0, ch 0
      -1.0f, 0.0f,   // B0, ch 1
      0.5f, 0.5f,    // B1, ch 0
      -0.2f, 0.4f};  // B1, ch 1

  RunCausalConvWithStateTests(
      input_data, weight_data, nullptr, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "silu");
}

// =============================================================================
// State continuity test: verify that present_state from one call can be used
// as conv_state for the next call (simulating autoregressive decode)
// =============================================================================

TEST(CausalConvWithStateTest, StateContinuity) {
  // Process a sequence of single tokens and verify state propagation
  int batch_size = 1, channels = 1, kernel_size = 3;
  int input_length = 1;

  std::vector<float> weight_data = {0.2f, 0.3f, 0.5f};
  std::vector<float> bias_data = {0.1f};

  // Initial state: zeros
  std::vector<float> conv_state = {0.0f, 0.0f};

  // First token
  std::vector<float> input1 = {1.0f};
  std::vector<float> expected_output1;
  std::vector<float> expected_state1;
  CausalConvWithStateReference(input1, weight_data, &bias_data, &conv_state,
                               expected_output1, expected_state1,
                               batch_size, channels, input_length, kernel_size, "none");

  RunCausalConvWithStateTest(input1, weight_data, &bias_data, &conv_state,
                             expected_output1, expected_state1,
                             batch_size, channels, input_length, kernel_size, "none",
                             TensorType::kFloat);

  // Second token, using present_state from first as conv_state
  std::vector<float> input2 = {2.0f};
  std::vector<float> expected_output2;
  std::vector<float> expected_state2;
  CausalConvWithStateReference(input2, weight_data, &bias_data, &expected_state1,
                               expected_output2, expected_state2,
                               batch_size, channels, input_length, kernel_size, "none");

  RunCausalConvWithStateTest(input2, weight_data, &bias_data, &expected_state1,
                             expected_output2, expected_state2,
                             batch_size, channels, input_length, kernel_size, "none",
                             TensorType::kFloat);

  // Third token
  std::vector<float> input3 = {3.0f};
  std::vector<float> expected_output3;
  std::vector<float> expected_state3;
  CausalConvWithStateReference(input3, weight_data, &bias_data, &expected_state2,
                               expected_output3, expected_state3,
                               batch_size, channels, input_length, kernel_size, "none");

  RunCausalConvWithStateTest(input3, weight_data, &bias_data, &expected_state2,
                             expected_output3, expected_state3,
                             batch_size, channels, input_length, kernel_size, "none",
                             TensorType::kFloat);

  // The present_state after processing [1, 2, 3] should be [2, 3]
  EXPECT_NEAR(expected_state3[0], 2.0f, 1e-5f);
  EXPECT_NEAR(expected_state3[1], 3.0f, 1e-5f);
}

// =============================================================================
// Equivalence test: sequence processing should match token-by-token with state
// =============================================================================

TEST(CausalConvWithStateTest, SequenceVsTokenByToken) {
  int batch_size = 1, channels = 2, kernel_size = 3;

  std::vector<float> weight_data = {
      0.1f, 0.2f, 0.3f,
      0.4f, 0.5f, 0.6f};
  std::vector<float> bias_data = {0.05f, -0.05f};

  // Initial state: zeros
  std::vector<float> conv_state = {0.0f, 0.0f, 0.0f, 0.0f};  // (1, 2, 2)

  // Full sequence: length 4
  std::vector<float> full_input = {
      1.0f, 2.0f, 3.0f, 4.0f,   // ch 0
      0.5f, 1.5f, 2.5f, 3.5f};  // ch 1

  // Process full sequence at once
  std::vector<float> full_output;
  std::vector<float> full_final_state;
  CausalConvWithStateReference(full_input, weight_data, &bias_data, &conv_state,
                               full_output, full_final_state,
                               batch_size, channels, 4, kernel_size, "none");

  // Process token by token
  std::vector<float> current_state = conv_state;
  std::vector<float> token_outputs;

  for (int t = 0; t < 4; ++t) {
    // Extract single token: (1, 2, 1)
    std::vector<float> token_input = {
        full_input[0 * 4 + t],   // ch 0
        full_input[1 * 4 + t]};  // ch 1

    std::vector<float> token_output;
    std::vector<float> next_state;
    CausalConvWithStateReference(token_input, weight_data, &bias_data, &current_state,
                                 token_output, next_state,
                                 batch_size, channels, 1, kernel_size, "none");

    // Collect outputs
    for (int d = 0; d < channels; ++d) {
      token_outputs.push_back(token_output[d]);
    }
    current_state = next_state;
  }

  // Rearrange token_outputs from (T, D) to (D, T) layout for comparison
  std::vector<float> token_outputs_dlt(channels * 4);
  for (int t = 0; t < 4; ++t) {
    for (int d = 0; d < channels; ++d) {
      token_outputs_dlt[d * 4 + t] = token_outputs[t * channels + d];
    }
  }

  // Compare outputs
  for (int i = 0; i < channels * 4; ++i) {
    EXPECT_NEAR(full_output[i], token_outputs_dlt[i], 1e-5f)
        << "Mismatch at index " << i;
  }

  // Compare final states
  for (int i = 0; i < channels * 2; ++i) {
    EXPECT_NEAR(full_final_state[i], current_state[i], 1e-5f)
        << "State mismatch at index " << i;
  }
}

// =============================================================================
// Larger dimension test with realistic sizes
// =============================================================================

TEST(CausalConvWithStateTest, LargerDimensions) {
  int batch_size = 2, channels = 8, input_length = 16, kernel_size = 4;

  // Generate test data with a simple pattern
  std::vector<float> input_data(batch_size * channels * input_length);
  for (int i = 0; i < static_cast<int>(input_data.size()); ++i) {
    input_data[i] = std::sin(static_cast<float>(i) * 0.1f);
  }

  std::vector<float> weight_data(channels * kernel_size);
  for (int i = 0; i < static_cast<int>(weight_data.size()); ++i) {
    weight_data[i] = std::cos(static_cast<float>(i) * 0.2f) * 0.5f;
  }

  std::vector<float> bias_data(channels);
  for (int i = 0; i < channels; ++i) {
    bias_data[i] = 0.01f * static_cast<float>(i);
  }

  int state_length = kernel_size - 1;
  std::vector<float> conv_state_data(batch_size * channels * state_length);
  for (int i = 0; i < static_cast<int>(conv_state_data.size()); ++i) {
    conv_state_data[i] = std::sin(static_cast<float>(i) * 0.3f) * 0.5f;
  }

  RunCausalConvWithStateTests(
      input_data, weight_data, &bias_data, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "silu");
}

// The state tensors grow linearly with state_window, so the schema caps it at 8.
TEST(CausalConvWithStateTest, StateWindowAboveMaxIsRejected) {
  OpTester test("CausalConvWithState", 1, onnxruntime::kMSDomain);
  test.AddAttribute<std::string>("activation", "none");
  test.AddAttribute<int64_t>("state_window", 9);
  test.AddInput<float>("input", {1, 1, 2}, {1.0f, 2.0f});
  test.AddInput<float>("weight", {1, 1, 2}, {0.5f, 0.25f});
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();
  test.AddOutput<float>("output", {1, 1, 2}, {0.5f, 1.25f});
  test.AddOutput<float>("present_state", {9, 1, 1, 1}, std::vector<float>(9, 0.0f));
  test.Run(OpTester::ExpectResult::kExpectFailure, "state_window must be in [0, 8]");
}

#ifdef USE_CUDA
TEST(CausalConvWithStateTest, StateWindowRejectsEmptySequence) {
  auto ep = DefaultCudaExecutionProvider();
  if (!ep) {
    GTEST_SKIP() << "CUDA execution provider not available";
    return;
  }

  OpTester test("CausalConvWithState", 1, onnxruntime::kMSDomain);
  test.AddAttribute<std::string>("activation", "none");
  test.AddAttribute<int64_t>("state_window", 1);
  test.AddInput<float>("input", {1, 3, 0}, {});
  test.AddInput<float>("weight", {3, 1, 2}, std::vector<float>(6, 1.0f));
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();
  test.AddOutput<float>("output", {1, 3, 0}, {});
  test.AddOutput<float>("present_state", {1, 1, 3, 1}, std::vector<float>(3, 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(ep));
  test.Run(OpTester::ExpectResult::kExpectFailure, "input length must be positive", {}, nullptr,
           &execution_providers);
}

// The state_window attribute is only implemented by the CUDA kernel. past_state / present_state
// then hold the last W per-position carry states, right-aligned; slot W-1 is the state after the
// last position (what the unwindowed op produces) and is the slot past_state is read from.
//
// The CUDA kernel has four families (fixed-K decode, generic-K decode, batched prefill,
// single-channel prefill) and each computes the slot offsets itself, so every family gets its own
// shape below. When `window` > `input_length` the slots below W - L hold no position from this
// call and are zero-filled, with or without a past_state.
static void RunCausalConvStateWindowTest(int batch_size, int channels, int input_length,
                                         int kernel_size, int window, bool with_past_state) {
  auto ep = DefaultCudaExecutionProvider();
  if (!ep) {
    GTEST_SKIP() << "CUDA execution provider not available";
    return;
  }

  const int state_length = kernel_size - 1;
  const std::string activation = "silu";

  std::vector<float> input_data(static_cast<size_t>(batch_size) * channels * input_length);
  for (int i = 0; i < static_cast<int>(input_data.size()); ++i) {
    input_data[i] = 0.5f * std::sin(static_cast<float>(i) * 0.37f);
  }
  std::vector<float> weight_data(static_cast<size_t>(channels) * kernel_size);
  for (int i = 0; i < static_cast<int>(weight_data.size()); ++i) {
    weight_data[i] = 0.25f * std::cos(static_cast<float>(i) * 0.21f);
  }
  std::vector<float> bias_data(channels);
  for (int i = 0; i < channels; ++i) {
    bias_data[i] = 0.01f * static_cast<float>(i);
  }
  std::vector<float> conv_state_data(static_cast<size_t>(batch_size) * channels * state_length);
  for (int i = 0; i < static_cast<int>(conv_state_data.size()); ++i) {
    conv_state_data[i] = with_past_state ? 0.5f * std::sin(static_cast<float>(i) * 0.3f) : 0.0f;
  }
  const std::vector<float>* past = with_past_state ? &conv_state_data : nullptr;

  // Keep only the first `prefix` positions of a (B, C, L) tensor.
  auto slice_prefix = [&](const std::vector<float>& src, int prefix) {
    std::vector<float> dst(static_cast<size_t>(batch_size) * channels * prefix);
    for (int bc = 0; bc < batch_size * channels; ++bc) {
      for (int l = 0; l < prefix; ++l) {
        dst[static_cast<size_t>(bc) * prefix + l] = src[static_cast<size_t>(bc) * input_length + l];
      }
    }
    return dst;
  };

  std::vector<float> expected_output, expected_state;
  CausalConvWithStateReference(
      input_data, weight_data, &bias_data, past,
      expected_output, expected_state,
      batch_size, channels, input_length, kernel_size, activation);

  // Slot j holds the state after the first (input_length - window + j + 1) positions; slots for
  // non-positive prefixes are never computed by the kernel and stay zero. The window axis leads
  // the batch axis, so a slot is exactly one contiguous (B, C, K-1) reference block.
  const size_t slot_elems = static_cast<size_t>(channels) * state_length;
  const size_t batch_slot_elems = static_cast<size_t>(batch_size) * slot_elems;
  std::vector<float> expected_state_window(static_cast<size_t>(window) * batch_slot_elems, 0.0f);
  for (int j = 0; j < window; ++j) {
    const int prefix = input_length - window + j + 1;
    if (prefix <= 0) continue;
    std::vector<float> prefix_state;
    if (prefix == input_length) {
      prefix_state = expected_state;
    } else {
      std::vector<float> prefix_output;
      CausalConvWithStateReference(
          slice_prefix(input_data, prefix), weight_data, &bias_data, past,
          prefix_output, prefix_state,
          batch_size, channels, prefix, kernel_size, activation);
    }
    std::copy_n(prefix_state.begin(), batch_slot_elems,
                expected_state_window.begin() + static_cast<size_t>(j) * batch_slot_elems);
  }

  OpTester test("CausalConvWithState", 1, onnxruntime::kMSDomain);
  test.AddAttribute<std::string>("activation", activation);
  test.AddAttribute<int64_t>("state_window", static_cast<int64_t>(window));

  test.AddInput<float>("input", {batch_size, channels, input_length}, input_data);
  test.AddInput<float>("weight", {channels, 1, kernel_size}, weight_data);
  test.AddInput<float>("bias", {channels}, bias_data);
  if (with_past_state) {
    // past_state is windowed too, and only slot W-1 is read. Poison the earlier slots to prove it.
    std::vector<float> past_state_window(static_cast<size_t>(window) * batch_slot_elems, -1e4f);
    std::copy_n(conv_state_data.begin(), batch_slot_elems,
                past_state_window.begin() + static_cast<size_t>(window - 1) * batch_slot_elems);
    test.AddInput<float>("past_state", {window, batch_size, channels, state_length}, past_state_window);
  } else {
    test.AddOptionalInputEdge<float>();
  }

  test.AddOutput<float>("output", {batch_size, channels, input_length}, expected_output);
  test.AddOutput<float>("present_state", {window, batch_size, channels, state_length},
                        expected_state_window);
  test.SetOutputAbsErr("output", 0.01f);
  test.SetOutputAbsErr("present_state", 0.01f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// channels < 4 forces the single-channel prefill kernel.
TEST(CausalConvWithStateTest, StateWindow) {
  RunCausalConvStateWindowTest(/*batch_size=*/1, /*channels=*/3, /*input_length=*/5,
                               /*kernel_size=*/4, /*window=*/3, /*with_past_state=*/true);
}

// 2 <= L <= 128 with channels >= 4 selects the batched prefill kernel; B = 2 exercises the batch
// stride of the window axis.
TEST(CausalConvWithStateTest, StateWindow_BatchedPrefill) {
  RunCausalConvStateWindowTest(/*batch_size=*/2, /*channels=*/8, /*input_length=*/6,
                               /*kernel_size=*/4, /*window=*/3, /*with_past_state=*/true);
}

// L > 128 falls back to the single-channel prefill kernel even for wide inputs.
TEST(CausalConvWithStateTest, StateWindow_LongPrefill) {
  RunCausalConvStateWindowTest(/*batch_size=*/2, /*channels=*/8, /*input_length=*/140,
                               /*kernel_size=*/4, /*window=*/5, /*with_past_state=*/true);
}

// L == 1 with kernel_size in [2, 5] selects the compile-time specialized decode kernel. The window
// is wider than the sequence here, which is the shape genai actually runs during MTP decode.
TEST(CausalConvWithStateTest, StateWindow_DecodeFixedK) {
  RunCausalConvStateWindowTest(/*batch_size=*/2, /*channels=*/8, /*input_length=*/1,
                               /*kernel_size=*/4, /*window=*/3, /*with_past_state=*/false);
}

// kernel_size > 5 falls back to the generic decode kernel.
TEST(CausalConvWithStateTest, StateWindow_DecodeGenericK) {
  RunCausalConvStateWindowTest(/*batch_size=*/2, /*channels=*/8, /*input_length=*/1,
                               /*kernel_size=*/7, /*window=*/3, /*with_past_state=*/false);
}

// W > L with a past_state: the kernel writes only slot W-1, so the leading W-1 slots must come
// back zeroed rather than as uninitialized device memory.
TEST(CausalConvWithStateTest, StateWindow_DecodeFixedKWithPastState) {
  RunCausalConvStateWindowTest(/*batch_size=*/2, /*channels=*/8, /*input_length=*/1,
                               /*kernel_size=*/4, /*window=*/3, /*with_past_state=*/true);
}

TEST(CausalConvWithStateTest, StateWindow_DecodeGenericKWithPastState) {
  RunCausalConvStateWindowTest(/*batch_size=*/2, /*channels=*/8, /*input_length=*/1,
                               /*kernel_size=*/7, /*window=*/3, /*with_past_state=*/true);
}

// BFloat16 is CUDA-only for this op (CPU/WebGPU only register float/float16).
TEST(CausalConvWithStateTest, BFloat16_Cuda) {
  auto ep = DefaultCudaExecutionProvider();
  if (!ep) {
    GTEST_SKIP() << "CUDA execution provider not available";
    return;
  }
  if (!CudaHasBF16Support()) {
    GTEST_SKIP() << "CUDA device does not support BFloat16.";
    return;
  }

  const int batch_size = 2;
  const int channels = 4;
  const int input_length = 6;
  const int kernel_size = 3;
  const int state_length = kernel_size - 1;
  const std::string activation = "silu";

  std::vector<float> input_data(static_cast<size_t>(batch_size) * channels * input_length);
  for (size_t i = 0; i < input_data.size(); ++i) {
    input_data[i] = 0.3f * std::sin(static_cast<float>(i) * 0.41f);
  }
  std::vector<float> weight_data(static_cast<size_t>(channels) * kernel_size);
  for (size_t i = 0; i < weight_data.size(); ++i) {
    weight_data[i] = 0.2f * std::cos(static_cast<float>(i) * 0.17f);
  }
  std::vector<float> bias_data(channels);
  for (int i = 0; i < channels; ++i) {
    bias_data[i] = 0.01f * static_cast<float>(i);
  }
  std::vector<float> conv_state_data(static_cast<size_t>(batch_size) * channels * state_length);
  for (size_t i = 0; i < conv_state_data.size(); ++i) {
    conv_state_data[i] = 0.25f * std::sin(static_cast<float>(i) * 0.29f);
  }

  std::vector<float> expected_output, expected_state;
  CausalConvWithStateReference(
      input_data, weight_data, &bias_data, &conv_state_data,
      expected_output, expected_state,
      batch_size, channels, input_length, kernel_size, activation);

  std::vector<int64_t> input_shape = {batch_size, channels, input_length};
  std::vector<int64_t> weight_shape = {channels, 1, kernel_size};
  std::vector<int64_t> bias_shape = {channels};
  std::vector<int64_t> state_shape = {batch_size, channels, state_length};
  std::vector<int64_t> output_shape = {batch_size, channels, input_length};

  OpTester test("CausalConvWithState", 1, onnxruntime::kMSDomain);
  test.AddAttribute<std::string>("activation", activation);
  test.AddInput<BFloat16>("input", input_shape, ToBFloat16(input_data));
  test.AddInput<BFloat16>("weight", weight_shape, ToBFloat16(weight_data));
  test.AddInput<BFloat16>("bias", bias_shape, ToBFloat16(bias_data));
  test.AddInput<BFloat16>("past_state", state_shape, ToBFloat16(conv_state_data));
  test.AddOutput<BFloat16>("output", output_shape, ToBFloat16(expected_output), false, 0.02f, 0.0f);
  test.AddOutput<BFloat16>("present_state", state_shape, ToBFloat16(expected_state), false, 0.02f, 0.0f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(ep));
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif  // USE_CUDA

// =============================================================================
// VarlenCausalConvWithState: packed/ragged token-major batches (CUDA only)
// =============================================================================

TEST(ContribOpVarlenCausalConvWithStateTest, SchemaResolution) {
  const auto* schema = ONNX_NAMESPACE::OpSchemaRegistry::Schema("VarlenCausalConvWithState", 1, kMSDomain);
  ASSERT_NE(schema, nullptr);
  EXPECT_EQ(schema->inputs().size(), 5u);
  EXPECT_EQ(schema->outputs().size(), 2u);
  EXPECT_GT(schema->attributes().count("activation"), 0u);
  EXPECT_GT(schema->attributes().count("ndim"), 0u);
  EXPECT_GT(schema->attributes().count("state_window"), 0u);

  const auto bfloat16_type = ONNX_NAMESPACE::Utils::DataTypeUtils::ToType("tensor(bfloat16)");
  EXPECT_EQ(schema->inputs()[0].GetTypes().count(bfloat16_type), 0u);
  EXPECT_EQ(schema->inputs()[4].GetTypes().count(bfloat16_type), 0u);
  EXPECT_EQ(schema->outputs()[1].GetTypes().count(bfloat16_type), 0u);
}

#ifdef USE_CUDA
namespace {

// Returns a CUDA EP with the VarlenCausalConvWithState kernel registered, or nullptr. Unlike the
// dense op (also servable from WebGPU/CPU via TryGetEpWithCausalConvWithState), Varlen* ops are
// CUDA-only, so tests skip outright instead of falling back to another EP.
std::unique_ptr<IExecutionProvider> TryGetCudaEpWithVarlenCausalConvWithState() {
  auto ep = DefaultCudaExecutionProvider();
  if (!ep) {
    return nullptr;
  }
  auto kernel_registry = ep->GetKernelRegistry();
  if (kernel_registry) {
    const KernelCreateInfo* info = nullptr;
    KernelRegistry::TypeConstraintMap type_constraints;
    auto status = kernel_registry->TryFindKernel(
        ep->Type(), "VarlenCausalConvWithState", kMSDomain, 1,
        type_constraints, DefaultLoggingManager().DefaultLogger(), &info);
    if (!status.IsOK()) {
      return nullptr;
    }
  }
  return ep;
}

// Transpose a single request's (channels, length) reference block to the token-major
// (length, channels) layout the packed op expects.
std::vector<float> TransposeDL_to_LD(const std::vector<float>& data, int D, int L) {
  std::vector<float> out(static_cast<size_t>(D) * L);
  for (int d = 0; d < D; d++) {
    for (int l = 0; l < L; l++) {
      out[static_cast<size_t>(l) * D + d] = data[static_cast<size_t>(d) * L + l];
    }
  }
  return out;
}

// Keep only the first `prefix` positions of a single request's (D, L) reference block.
std::vector<float> SliceCausalConvPrefix(const std::vector<float>& src, int D, int L, int prefix) {
  std::vector<float> dst(static_cast<size_t>(D) * prefix);
  for (int d = 0; d < D; d++) {
    for (int l = 0; l < prefix; l++) {
      dst[static_cast<size_t>(d) * prefix + l] = src[static_cast<size_t>(d) * L + l];
    }
  }
  return dst;
}

// A packed/ragged test case. Every request's input/past_state is generated and run through the
// existing dense reference independently, with batch_size=1, and only then are the per-request
// packed rows and state blocks concatenated into one token-major batch. No request's reference
// computation ever sees another request's data, so packing cannot hide a cross-request leak that
// the op itself might introduce.
struct VarlenCausalConvCase {
  std::vector<int> seq_lens;
  int channels = 4;
  int kernel_size = 3;
  std::string activation = "silu";
  bool with_bias = true;
  bool with_past_state = false;
  int state_window = 0;
  bool use_fp16 = false;
  // When true, every request is filled with one large constant value of alternating sign instead
  // of a smooth per-request waveform, so any accidental cross-request boundary read produces an
  // unmistakably large mismatch instead of a subtle one.
  bool adversarial_constants = false;
};

void RunVarlenCausalConvCase(const VarlenCausalConvCase& c) {
  auto ep = TryGetCudaEpWithVarlenCausalConvWithState();
  if (!ep) {
    GTEST_SKIP() << "VarlenCausalConvWithState kernel not registered";
    return;
  }

  const int B = static_cast<int>(c.seq_lens.size());
  const int D = c.channels;
  const int K = c.kernel_size;
  const int pad = K - 1;
  const int W = c.state_window;
  const size_t slot_elems = static_cast<size_t>(D) * pad;

  std::vector<int32_t> cu_seqlens(static_cast<size_t>(B) + 1, 0);
  for (int i = 0; i < B; i++) {
    cu_seqlens[i + 1] = cu_seqlens[i] + c.seq_lens[i];
  }
  const int total_tokens = cu_seqlens[B];

  std::vector<float> weight(static_cast<size_t>(D) * K);
  for (size_t idx = 0; idx < weight.size(); idx++) {
    weight[idx] = 0.25f * std::cos(static_cast<float>(idx) * 0.21f);
  }
  std::vector<float> bias(D, 0.0f);
  if (c.with_bias) {
    for (int d = 0; d < D; d++) bias[d] = 0.01f * static_cast<float>(d) - 0.02f;
  }
  const std::vector<float>* bias_ptr = c.with_bias ? &bias : nullptr;

  std::vector<float> packed_input, packed_output;
  std::vector<float> packed_past_state(static_cast<size_t>(B) * slot_elems, 0.0f);
  std::vector<float> packed_present_state(static_cast<size_t>(B) * slot_elems, 0.0f);
  const size_t window_slots = static_cast<size_t>(std::max(W, 1));
  std::vector<float> packed_past_state_window(window_slots * B * slot_elems, -1e4f);
  std::vector<float> packed_present_state_window(window_slots * B * slot_elems, 0.0f);

  int seed = 0;
  for (int i = 0; i < B; i++) {
    const int L = c.seq_lens[i];
    std::vector<float> input(static_cast<size_t>(D) * L);
    if (c.adversarial_constants) {
      const float value = (i % 2 == 0) ? 1000.0f : -1000.0f;
      std::fill(input.begin(), input.end(), value);
    } else {
      for (size_t idx = 0; idx < input.size(); idx++) {
        input[idx] = 0.5f * std::sin(static_cast<float>(idx + seed) * 0.37f);
      }
    }
    seed += 53;

    std::vector<float> initial_state(slot_elems, 0.0f);
    if (c.with_past_state) {
      for (size_t idx = 0; idx < initial_state.size(); idx++) {
        initial_state[idx] = 0.1f * std::cos(static_cast<float>(idx + i * 11) * 0.3f);
      }
    }
    const std::vector<float>* past = c.with_past_state ? &initial_state : nullptr;

    std::vector<float> output_i, final_state_i;
    CausalConvWithStateReference(input, weight, bias_ptr, past, output_i, final_state_i,
                                 1, D, L, K, c.activation);

    std::vector<float> input_td = TransposeDL_to_LD(input, D, L);
    std::vector<float> output_td = TransposeDL_to_LD(output_i, D, L);
    packed_input.insert(packed_input.end(), input_td.begin(), input_td.end());
    packed_output.insert(packed_output.end(), output_td.begin(), output_td.end());

    std::copy(initial_state.begin(), initial_state.end(),
              packed_past_state.begin() + static_cast<size_t>(i) * slot_elems);
    std::copy(final_state_i.begin(), final_state_i.end(),
              packed_present_state.begin() + static_cast<size_t>(i) * slot_elems);

    if (W > 0) {
      // past_state is windowed too, and only slot W-1 is read: poison the earlier slots so a
      // stray read of them would fail the comparison below.
      std::copy(initial_state.begin(), initial_state.end(),
                packed_past_state_window.begin() + (static_cast<size_t>(W - 1) * B + i) * slot_elems);
      // Slot j holds the state after this request's own first (L - W + j + 1) positions; slots
      // for non-positive prefixes are never computed by the kernel and stay zero. Each request's
      // own L decides which slots are non-empty, so unequal lengths in one call produce different
      // non-zero slot ranges per request -- this is the "state_window unequal per sequence" case.
      for (int j = 0; j < W; j++) {
        const int prefix = L - W + j + 1;
        if (prefix <= 0) continue;
        std::vector<float> slot_state;
        if (prefix == L) {
          slot_state = final_state_i;
        } else {
          std::vector<float> input_prefix = SliceCausalConvPrefix(input, D, L, prefix);
          std::vector<float> slot_output;
          CausalConvWithStateReference(input_prefix, weight, bias_ptr, past, slot_output, slot_state,
                                       1, D, prefix, K, c.activation);
        }
        std::copy(slot_state.begin(), slot_state.end(),
                  packed_present_state_window.begin() + (static_cast<size_t>(j) * B + i) * slot_elems);
      }
    }
  }

  OpTester tester("VarlenCausalConvWithState", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<std::string>("activation", c.activation);
  if (W > 0) {
    tester.AddAttribute<int64_t>("state_window", static_cast<int64_t>(W));
  }

  const std::vector<int64_t> cu_seqlens_dims = {B + 1};
  const std::vector<int64_t> input_dims = {total_tokens, D};
  const std::vector<int64_t> weight_dims = {D, 1, K};
  const std::vector<int64_t> bias_dims = {D};
  std::vector<int64_t> state_dims;
  if (W > 0) {
    state_dims = {W, B, D, pad};
  } else {
    state_dims = {B, D, pad};
  }

  const std::vector<float>& past_state_flat = W > 0 ? packed_past_state_window : packed_past_state;
  const std::vector<float>& present_state_flat = W > 0 ? packed_present_state_window : packed_present_state;

  // OpTester::AddOutput takes (sort_output, rel_error, abs_error); a single shared tolerance
  // plays both roles here since ragged packing/unpacking is exact and float error is small.
  const float tol = c.use_fp16 ? 0.02f : 0.01f;

  if (!c.use_fp16) {
    tester.AddInput<float>("input", input_dims, packed_input);
    tester.AddInput<float>("weight", weight_dims, weight);
    tester.AddInput<int32_t>("cumulative_sequence_length", cu_seqlens_dims, cu_seqlens);
    if (c.with_bias) {
      tester.AddInput<float>("bias", bias_dims, bias);
    } else {
      tester.AddOptionalInputEdge<float>();
    }
    if (c.with_past_state) {
      tester.AddInput<float>("past_state", state_dims, past_state_flat);
    } else {
      tester.AddOptionalInputEdge<float>();
    }
    tester.AddOutput<float>("output", input_dims, packed_output, false, tol, tol);
    tester.AddOutput<float>("present_state", state_dims, present_state_flat, false, tol, tol);
  } else {
    tester.AddInput<MLFloat16>("input", input_dims, ToFloat16(packed_input));
    tester.AddInput<MLFloat16>("weight", weight_dims, ToFloat16(weight));
    tester.AddInput<int32_t>("cumulative_sequence_length", cu_seqlens_dims, cu_seqlens);
    if (c.with_bias) {
      tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias));
    } else {
      tester.AddOptionalInputEdge<MLFloat16>();
    }
    if (c.with_past_state) {
      tester.AddInput<MLFloat16>("past_state", state_dims, ToFloat16(past_state_flat));
    } else {
      tester.AddOptionalInputEdge<MLFloat16>();
    }
    tester.AddOutput<MLFloat16>("output", input_dims, ToFloat16(packed_output), false, tol, tol);
    tester.AddOutput<MLFloat16>("present_state", state_dims, ToFloat16(present_state_flat), false, tol, tol);
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(ep));
  tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

}  // namespace

TEST(ContribOpVarlenCausalConvWithStateTest, UnequalLengths) {
  VarlenCausalConvCase c;
  c.seq_lens = {3, 1, 2};
  RunVarlenCausalConvCase(c);
}

// Every request contributes exactly one token, selecting the all-ones decode fast path
// (cu_seqlens is never read on device). A windowed past state also verifies that decode writes
// only slot W-1 and leaves earlier slots zeroed.
TEST(ContribOpVarlenCausalConvWithStateTest, AllOnesDecode) {
  VarlenCausalConvCase c;
  c.seq_lens = {1, 1, 1, 1};
  c.with_past_state = true;
  c.state_window = 3;
  RunVarlenCausalConvCase(c);
}

TEST(ContribOpVarlenCausalConvWithStateTest, SingleRequestPrefill) {
  VarlenCausalConvCase c;
  c.seq_lens = {6};
  RunVarlenCausalConvCase(c);
}

TEST(ContribOpVarlenCausalConvWithStateTest, WithPastState) {
  VarlenCausalConvCase c;
  c.seq_lens = {3, 1, 2};
  c.with_past_state = true;
  RunVarlenCausalConvCase(c);
}

TEST(ContribOpVarlenCausalConvWithStateTest, NoActivation) {
  VarlenCausalConvCase c;
  c.seq_lens = {3, 1, 2};
  c.activation = "none";
  RunVarlenCausalConvCase(c);
}

TEST(ContribOpVarlenCausalConvWithStateTest, SwishActivation) {
  VarlenCausalConvCase c;
  c.seq_lens = {2, 3};
  c.activation = "swish";
  RunVarlenCausalConvCase(c);
}

TEST(ContribOpVarlenCausalConvWithStateTest, NoBias) {
  VarlenCausalConvCase c;
  c.seq_lens = {3, 1, 2};
  c.with_bias = false;
  RunVarlenCausalConvCase(c);
}

TEST(ContribOpVarlenCausalConvWithStateTest, KernelSize2) {
  VarlenCausalConvCase c;
  c.seq_lens = {3, 1, 2};
  c.kernel_size = 2;
  RunVarlenCausalConvCase(c);
}

TEST(ContribOpVarlenCausalConvWithStateTest, KernelSize1) {
  VarlenCausalConvCase c;
  c.seq_lens = {3, 1, 2};
  c.kernel_size = 1;
  RunVarlenCausalConvCase(c);
}

TEST(ContribOpVarlenCausalConvWithStateTest, KernelSize5) {
  VarlenCausalConvCase c;
  c.seq_lens = {4, 3, 5};
  c.kernel_size = 5;
  RunVarlenCausalConvCase(c);
}

TEST(ContribOpVarlenCausalConvWithStateTest, ManyChannels) {
  VarlenCausalConvCase c;
  c.seq_lens = {5, 2, 4};
  c.channels = 16;
  RunVarlenCausalConvCase(c);
}

TEST(ContribOpVarlenCausalConvWithStateTest, StateWindowUnequalLengths) {
  VarlenCausalConvCase c;
  c.seq_lens = {3, 1, 2};
  c.state_window = 3;
  RunVarlenCausalConvCase(c);
}

TEST(ContribOpVarlenCausalConvWithStateTest, StateWindowWithPastState) {
  VarlenCausalConvCase c;
  c.seq_lens = {3, 1, 2};
  c.state_window = 4;
  c.with_past_state = true;
  RunVarlenCausalConvCase(c);
}

// W wider than every packed request's own length: the leading slots for every request come back
// zeroed rather than as uninitialized device memory.
TEST(ContribOpVarlenCausalConvWithStateTest, StateWindowWiderThanEverySequence) {
  VarlenCausalConvCase c;
  c.seq_lens = {2, 1, 2};
  c.state_window = 5;
  RunVarlenCausalConvCase(c);
}

TEST(ContribOpVarlenCausalConvWithStateTest, Float16) {
  VarlenCausalConvCase c;
  c.seq_lens = {1, 1, 1, 1};
  c.use_fp16 = true;
  RunVarlenCausalConvCase(c);
}

TEST(ContribOpVarlenCausalConvWithStateTest, Float16WithPastState) {
  VarlenCausalConvCase c;
  c.seq_lens = {2, 4};
  c.with_past_state = true;
  c.use_fp16 = true;
  RunVarlenCausalConvCase(c);
}

// Two adjacent requests filled with wildly different constant values, kernel_size >= 3 so at
// least two taps land before each request's own first token, and no past_state so those taps must
// resolve to zero. If the kernel ever read a negative local position from whatever packed input
// happens to sit immediately before a request's own range -- instead of branching to past_state
// or zero -- the leaked neighbor value here is 2000 away from correct and impossible to miss.
TEST(ContribOpVarlenCausalConvWithStateTest, AdjacentDistinctRequestsNoState) {
  VarlenCausalConvCase c;
  c.seq_lens = {4, 4, 4};
  c.kernel_size = 3;
  c.with_bias = false;
  c.with_past_state = false;
  c.adversarial_constants = true;
  RunVarlenCausalConvCase(c);
}

// Same adversarial values, but with a state_window so both the convolution sum and every window
// slot write are checked for boundary leaks at once.
TEST(ContribOpVarlenCausalConvWithStateTest, AdjacentDistinctRequestsNoStateWithWindow) {
  VarlenCausalConvCase c;
  c.seq_lens = {4, 4, 4};
  c.kernel_size = 3;
  c.with_bias = false;
  c.with_past_state = false;
  c.adversarial_constants = true;
  c.state_window = 3;
  RunVarlenCausalConvCase(c);
}

// present_state from one packed call feeds past_state for the next, with the same request
// lengths each call (the shape a continuous-batching decode step actually reuses), and two
// requests carried simultaneously so a slot mix-up between them would be caught. Mirrors the
// dense StateContinuity test's technique of chaining through the reference's own carried state.
TEST(ContribOpVarlenCausalConvWithStateTest, MultiCallStateCarry) {
  const int D = 2, K = 3, pad = K - 1;
  const std::vector<int> seq_lens = {2, 3};
  const int B = static_cast<int>(seq_lens.size());
  std::vector<int32_t> cu_seqlens(static_cast<size_t>(B) + 1, 0);
  for (int i = 0; i < B; i++) cu_seqlens[i + 1] = cu_seqlens[i] + seq_lens[i];
  const int total_tokens = cu_seqlens[B];

  const std::vector<float> weight = {0.2f, 0.3f, 0.5f, 0.1f, 0.4f, 0.5f};  // (D=2, K=3)
  const std::vector<float> bias = {0.05f, -0.05f};
  const size_t slot_elems = static_cast<size_t>(D) * pad;

  std::vector<float> state0(slot_elems, 0.0f), state1(slot_elems, 0.0f);
  std::vector<float>* states[2] = {&state0, &state1};

  for (int call = 0; call < 2; call++) {
    auto ep = TryGetCudaEpWithVarlenCausalConvWithState();
    if (!ep) {
      GTEST_SKIP() << "VarlenCausalConvWithState kernel not registered";
      return;
    }

    std::vector<float> packed_input, packed_output;
    std::vector<float> packed_past_state(static_cast<size_t>(B) * slot_elems);
    std::vector<float> packed_present_state(static_cast<size_t>(B) * slot_elems);
    for (int i = 0; i < B; i++) {
      const int L = seq_lens[i];
      std::vector<float> input(static_cast<size_t>(D) * L);
      for (size_t idx = 0; idx < input.size(); idx++) {
        input[idx] = 0.2f * static_cast<float>(call + 1) +
                     0.1f * std::sin(static_cast<float>(idx + i * 5) * 0.4f);
      }
      const std::vector<float>* past_ptr = (call == 0) ? nullptr : states[i];
      std::vector<float> output_i, next_state_i;
      CausalConvWithStateReference(input, weight, &bias, past_ptr, output_i, next_state_i,
                                   1, D, L, K, "none");

      std::vector<float> input_td = TransposeDL_to_LD(input, D, L);
      std::vector<float> output_td = TransposeDL_to_LD(output_i, D, L);
      packed_input.insert(packed_input.end(), input_td.begin(), input_td.end());
      packed_output.insert(packed_output.end(), output_td.begin(), output_td.end());
      if (call > 0) {
        std::copy(states[i]->begin(), states[i]->end(),
                  packed_past_state.begin() + static_cast<size_t>(i) * slot_elems);
      }
      std::copy(next_state_i.begin(), next_state_i.end(),
                packed_present_state.begin() + static_cast<size_t>(i) * slot_elems);
      *states[i] = next_state_i;
    }

    const std::vector<int64_t> input_dims = {total_tokens, D};
    const std::vector<int64_t> weight_dims = {D, 1, K};
    const std::vector<int64_t> bias_dims = {D};
    const std::vector<int64_t> state_dims = {B, D, pad};

    OpTester tester("VarlenCausalConvWithState", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<std::string>("activation", std::string("none"));
    tester.AddInput<float>("input", input_dims, packed_input);
    tester.AddInput<float>("weight", weight_dims, weight);
    tester.AddInput<int32_t>("cumulative_sequence_length", {B + 1}, cu_seqlens);
    tester.AddInput<float>("bias", bias_dims, bias);
    if (call == 0) {
      tester.AddOptionalInputEdge<float>();
    } else {
      tester.AddInput<float>("past_state", state_dims, packed_past_state);
    }
    tester.AddOutput<float>("output", input_dims, packed_output, false, 0.01f, 0.01f);
    tester.AddOutput<float>("present_state", state_dims, packed_present_state, false, 0.01f, 0.01f);

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(std::move(ep));
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

// Host-verifiable shape errors: these check rank/shape relationships computed purely from tensor
// shape metadata. Device offset contents (including the final cumulative_sequence_length entry)
// are never inspected on the host and so are not exercised here.
static void RunVarlenCausalConvShapeFailure(
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& weight_dims,
    const std::vector<int64_t>& cu_seqlens_dims,
    const std::vector<int32_t>& cu_seqlens_data,
    const std::vector<int64_t>& output_dims,
    const std::vector<int64_t>& state_dims,
    const std::string& expected_error,
    const std::vector<int64_t>& past_state_dims = {}) {
  auto ep = DefaultCudaExecutionProvider();
  if (!ep) {
    GTEST_SKIP() << "CUDA execution provider not available";
    return;
  }

  auto element_count = [](const std::vector<int64_t>& dims) {
    size_t count = 1;
    for (int64_t dim : dims) count *= static_cast<size_t>(dim);
    return count;
  };

  OpTester tester("VarlenCausalConvWithState", 1, onnxruntime::kMSDomain);
  tester.AddAttribute<std::string>("activation", std::string("none"));
  tester.AddInput<float>("input", input_dims, std::vector<float>(element_count(input_dims), 0.1f));
  tester.AddInput<float>("weight", weight_dims, std::vector<float>(element_count(weight_dims), 0.1f));
  tester.AddInput<int32_t>("cumulative_sequence_length", cu_seqlens_dims, cu_seqlens_data);
  tester.AddOptionalInputEdge<float>();  // bias
  if (past_state_dims.empty()) {
    tester.AddOptionalInputEdge<float>();
  } else {
    tester.AddInput<float>("past_state", past_state_dims,
                           std::vector<float>(element_count(past_state_dims), 0.0f));
  }
  tester.AddOutput<float>("output", output_dims, std::vector<float>(element_count(output_dims), 0.0f));
  tester.AddOutput<float>("present_state", state_dims, std::vector<float>(element_count(state_dims), 0.0f));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(std::move(ep));
  tester.Run(OpTester::ExpectResult::kExpectFailure, expected_error, {}, nullptr, &execution_providers);
}

TEST(ContribOpVarlenCausalConvWithStateTest, CudaRejectsInputRank) {
  RunVarlenCausalConvShapeFailure({1, 3, 4}, {4, 1, 2}, {2}, {0, 3}, {1, 3, 4}, {1, 4, 1},
                                  "input must have rank 2");
}

TEST(ContribOpVarlenCausalConvWithStateTest, CudaRejectsWeightRank) {
  RunVarlenCausalConvShapeFailure({3, 4}, {4, 2}, {2}, {0, 3}, {3, 4}, {1, 4, 1},
                                  "weight must have rank 3");
}

TEST(ContribOpVarlenCausalConvWithStateTest, CudaRejectsCuSeqlensRank2) {
  RunVarlenCausalConvShapeFailure({3, 4}, {4, 1, 2}, {1, 2}, {0, 3}, {3, 4}, {1, 4, 1},
                                  "cumulative_sequence_length must have rank 1");
}

TEST(ContribOpVarlenCausalConvWithStateTest, CudaRejectsCuSeqlensTooFewElements) {
  RunVarlenCausalConvShapeFailure({3, 4}, {4, 1, 2}, {1}, {0}, {3, 4}, {1, 4, 1},
                                  "at least 2 elements");
}

TEST(ContribOpVarlenCausalConvWithStateTest, CudaRejectsWeightChannelMismatch) {
  RunVarlenCausalConvShapeFailure({3, 4}, {5, 1, 2}, {2}, {0, 3}, {3, 4}, {1, 4, 1},
                                  "must match input channels");
}

TEST(ContribOpVarlenCausalConvWithStateTest, CudaRejectsWeightMidDimNotOne) {
  RunVarlenCausalConvShapeFailure({3, 4}, {4, 2, 2}, {2}, {0, 3}, {3, 4}, {1, 4, 1},
                                  "must be 1 for depthwise convolution");
}

TEST(ContribOpVarlenCausalConvWithStateTest, CudaRejectsPastStateShapeWithVarlenDiagnostic) {
  RunVarlenCausalConvShapeFailure({3, 4}, {4, 1, 2}, {2}, {0, 3}, {3, 4}, {1, 4, 1},
                                  "VarlenCausalConvWithState uses", {2, 4, 1});
}
#endif  // USE_CUDA

}  // namespace test
}  // namespace onnxruntime
