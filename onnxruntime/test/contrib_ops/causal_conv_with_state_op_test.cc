// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <cmath>
#include <vector>
#include "gtest/gtest.h"
#include "core/common/logging/logging.h"
#include "core/framework/kernel_registry.h"
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
#endif  // USE_CUDA

}  // namespace test
}  // namespace onnxruntime
