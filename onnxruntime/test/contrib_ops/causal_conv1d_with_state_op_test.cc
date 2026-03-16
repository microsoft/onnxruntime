// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cassert>
#include <cmath>
#include <vector>
#include "gtest/gtest.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {
enum class TensorType {
  kFloat,
  kFloat16
};

// Reference implementation for CausalConv1DWithState
// Performs depthwise causal 1D convolution with optional state, bias, and activation.
//
// Input: (B, D, L) channels-first
// Weight: (D, 1, K) depthwise
// Bias: (D,) optional
// conv_state: (B, D, K-1) optional carry state
//
// Output: (B, D, L) convolution output (with optional activation)
// present_state: (B, D, K-1) updated carry state
void CausalConv1DWithStateReference(
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

}  // anonymous namespace

static void RunCausalConv1DWithStateTest(
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
  int state_length = kernel_size - 1;

  std::vector<int64_t> input_shape = {batch_size, channels, input_length};
  std::vector<int64_t> weight_shape = {channels, 1, kernel_size};
  std::vector<int64_t> bias_shape = {channels};
  std::vector<int64_t> state_shape = {batch_size, channels, state_length};
  std::vector<int64_t> output_shape = {batch_size, channels, input_length};

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;

  bool enable_webgpu = nullptr != DefaultWebGpuExecutionProvider().get();
  if (enable_webgpu) {
    execution_providers.push_back(DefaultWebGpuExecutionProvider());
  }

  if (execution_providers.empty()) {
    // Skip if no providers available
    return;
  }

  for (auto& ep : execution_providers) {
    OpTester test("CausalConv1DWithState", 1, onnxruntime::kMSDomain);
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
        test.AddInput<float>("conv_state", state_shape, *conv_state_data);
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
        test.AddInput<MLFloat16>("conv_state", state_shape, ToFloat16(*conv_state_data));
      } else {
        test.AddOptionalInputEdge<MLFloat16>();
      }

      test.AddOutput<MLFloat16>("output", output_shape, ToFloat16(expected_output));
      test.AddOutput<MLFloat16>("present_state", state_shape, ToFloat16(expected_state));
    }

    test.SetOutputAbsErr("output", 0.01f);
    test.SetOutputAbsErr("present_state", 0.01f);

    std::vector<std::unique_ptr<IExecutionProvider>> test_execution_providers;
    test_execution_providers.push_back(std::move(ep));
    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &test_execution_providers);
  }
}

static void RunCausalConv1DWithStateTests(
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
  CausalConv1DWithStateReference(
      input_data, weight_data, bias_data, conv_state_data,
      expected_output, expected_state,
      batch_size, channels, input_length, kernel_size, activation);

  // FP32 test
  RunCausalConv1DWithStateTest(
      input_data, weight_data, bias_data, conv_state_data,
      expected_output, expected_state,
      batch_size, channels, input_length, kernel_size, activation,
      TensorType::kFloat);

  // FP16 test
  RunCausalConv1DWithStateTest(
      input_data, weight_data, bias_data, conv_state_data,
      expected_output, expected_state,
      batch_size, channels, input_length, kernel_size, activation,
      TensorType::kFloat16);
}

// =============================================================================
// Basic tests - simple cases
// =============================================================================

TEST(CausalConv1DWithStateTest, BasicNoStateNoBias) {
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

  RunCausalConv1DWithStateTests(
      input_data, weight_data, nullptr, nullptr,
      batch_size, channels, input_length, kernel_size, "none");
}

TEST(CausalConv1DWithStateTest, BasicWithBias) {
  // B=1, D=2, L=4, K=3, activation=none
  int batch_size = 1, channels = 2, input_length = 4, kernel_size = 3;

  std::vector<float> input_data = {
      1.0f, 2.0f, 3.0f, 4.0f,
      0.5f, 1.5f, 2.5f, 3.5f};
  std::vector<float> weight_data = {
      0.1f, 0.2f, 0.3f,
      0.4f, 0.5f, 0.6f};
  std::vector<float> bias_data = {0.1f, -0.2f};

  RunCausalConv1DWithStateTests(
      input_data, weight_data, &bias_data, nullptr,
      batch_size, channels, input_length, kernel_size, "none");
}

TEST(CausalConv1DWithStateTest, BasicWithState) {
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

  RunCausalConv1DWithStateTests(
      input_data, weight_data, nullptr, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "none");
}

TEST(CausalConv1DWithStateTest, WithStateAndBias) {
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

  RunCausalConv1DWithStateTests(
      input_data, weight_data, &bias_data, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "none");
}

// =============================================================================
// SiLU activation tests
// =============================================================================

TEST(CausalConv1DWithStateTest, SiluActivationNoState) {
  int batch_size = 1, channels = 2, input_length = 4, kernel_size = 3;

  std::vector<float> input_data = {
      1.0f, 2.0f, 3.0f, 4.0f,
      0.5f, 1.5f, 2.5f, 3.5f};
  std::vector<float> weight_data = {
      0.1f, 0.2f, 0.3f,
      0.4f, 0.5f, 0.6f};

  RunCausalConv1DWithStateTests(
      input_data, weight_data, nullptr, nullptr,
      batch_size, channels, input_length, kernel_size, "silu");
}

TEST(CausalConv1DWithStateTest, SiluActivationWithState) {
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

  RunCausalConv1DWithStateTests(
      input_data, weight_data, nullptr, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "silu");
}

TEST(CausalConv1DWithStateTest, SiluActivationWithBiasAndState) {
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

  RunCausalConv1DWithStateTests(
      input_data, weight_data, &bias_data, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "silu");
}

// =============================================================================
// Kernel size variations
// =============================================================================

TEST(CausalConv1DWithStateTest, KernelSize2) {
  int batch_size = 1, channels = 2, input_length = 4, kernel_size = 2;

  std::vector<float> input_data = {
      1.0f, 2.0f, 3.0f, 4.0f,
      0.5f, 1.5f, 2.5f, 3.5f};
  std::vector<float> weight_data = {
      0.3f, 0.7f,
      0.4f, 0.6f};
  // State: (1, 2, 1) - kernel_size - 1 = 1
  std::vector<float> conv_state_data = {0.5f, -0.3f};

  RunCausalConv1DWithStateTests(
      input_data, weight_data, nullptr, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "silu");
}

TEST(CausalConv1DWithStateTest, KernelSize4) {
  int batch_size = 1, channels = 1, input_length = 5, kernel_size = 4;

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> weight_data = {0.1f, 0.2f, 0.3f, 0.4f};
  // State: (1, 1, 3)
  std::vector<float> conv_state_data = {-1.0f, 0.0f, 0.5f};

  RunCausalConv1DWithStateTests(
      input_data, weight_data, nullptr, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "none");
}

// =============================================================================
// Batch size > 1
// =============================================================================

TEST(CausalConv1DWithStateTest, MultiBatch) {
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

  RunCausalConv1DWithStateTests(
      input_data, weight_data, &bias_data, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "silu");
}

// =============================================================================
// Single token decode (L=1) - the primary use case for incremental decoding
// =============================================================================

TEST(CausalConv1DWithStateTest, SingleTokenDecode) {
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

  RunCausalConv1DWithStateTests(
      input_data, weight_data, &bias_data, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "silu");
}

TEST(CausalConv1DWithStateTest, SingleTokenDecodeMultiBatch) {
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

  RunCausalConv1DWithStateTests(
      input_data, weight_data, nullptr, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "silu");
}

// =============================================================================
// State continuity test: verify that present_state from one call can be used
// as conv_state for the next call (simulating autoregressive decode)
// =============================================================================

TEST(CausalConv1DWithStateTest, StateContinuity) {
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
  CausalConv1DWithStateReference(input1, weight_data, &bias_data, &conv_state,
                                 expected_output1, expected_state1,
                                 batch_size, channels, input_length, kernel_size, "none");

  RunCausalConv1DWithStateTest(input1, weight_data, &bias_data, &conv_state,
                               expected_output1, expected_state1,
                               batch_size, channels, input_length, kernel_size, "none",
                               TensorType::kFloat);

  // Second token, using present_state from first as conv_state
  std::vector<float> input2 = {2.0f};
  std::vector<float> expected_output2;
  std::vector<float> expected_state2;
  CausalConv1DWithStateReference(input2, weight_data, &bias_data, &expected_state1,
                                 expected_output2, expected_state2,
                                 batch_size, channels, input_length, kernel_size, "none");

  RunCausalConv1DWithStateTest(input2, weight_data, &bias_data, &expected_state1,
                               expected_output2, expected_state2,
                               batch_size, channels, input_length, kernel_size, "none",
                               TensorType::kFloat);

  // Third token
  std::vector<float> input3 = {3.0f};
  std::vector<float> expected_output3;
  std::vector<float> expected_state3;
  CausalConv1DWithStateReference(input3, weight_data, &bias_data, &expected_state2,
                                 expected_output3, expected_state3,
                                 batch_size, channels, input_length, kernel_size, "none");

  RunCausalConv1DWithStateTest(input3, weight_data, &bias_data, &expected_state2,
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

TEST(CausalConv1DWithStateTest, SequenceVsTokenByToken) {
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
  CausalConv1DWithStateReference(full_input, weight_data, &bias_data, &conv_state,
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
    CausalConv1DWithStateReference(token_input, weight_data, &bias_data, &current_state,
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

TEST(CausalConv1DWithStateTest, LargerDimensions) {
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

  RunCausalConv1DWithStateTests(
      input_data, weight_data, &bias_data, &conv_state_data,
      batch_size, channels, input_length, kernel_size, "silu");
}

}  // namespace test
}  // namespace onnxruntime
