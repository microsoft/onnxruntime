// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

static void RunRelativePositionBiasTest(
    const std::vector<float>& bias_table,         // Shape = [num_buckets, num_heads]
    const std::vector<int64_t>& sequence_length,  // Shape = [1]
    const std::vector<float>& output_data,        // Shape = [1, num_heads, sequence_length, sequence_length]
    int max_distance,
    int num_buckets,
    int num_heads,
    int seq_len,
    int is_bidirectional,
    bool use_float16 = false) {
  int min_cuda_architecture = use_float16 ? 530 : 0;

  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_cpu = false;
  if (enable_cpu || enable_cuda) {
    OpTester tester("RelativePositionBias", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("max_distance", static_cast<int64_t>(max_distance));
    tester.AddAttribute<int64_t>("is_bidirectional", static_cast<int64_t>(is_bidirectional));

    std::vector<int64_t> bias_table_dims = {num_buckets, num_heads};
    std::vector<int64_t> sequence_length_dims = {1};
    std::vector<int64_t> output_dims = {1, num_heads, seq_len, seq_len};

    if (use_float16) {
      tester.AddInput<MLFloat16>("bias_table", bias_table_dims, ToFloat16(bias_table));
      tester.AddInput<int64_t>("query_length", sequence_length_dims, sequence_length);
      tester.AddInput<int64_t>("key_length", sequence_length_dims, sequence_length);
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output_data));
    } else {
      tester.AddInput<float>("bias_table", bias_table_dims, bias_table);
      tester.AddInput<int64_t>("query_length", sequence_length_dims, sequence_length);
      tester.AddInput<int64_t>("key_length", sequence_length_dims, sequence_length);
      tester.AddOutput<float>("output", output_dims, output_data);
    }

    if (enable_cuda) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCudaExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    if (enable_cpu) {
      std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
      execution_providers.push_back(DefaultCpuExecutionProvider());
      tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }
  }
}

TEST(RelativePositionBiasTest, RelativePositionBiasTest_FP32) {
  int max_distance = 128;
  int num_buckets = 4;
  int num_heads = 2;
  int seq_len = 2;
  int is_bidirectional = 1;

  // Huggingface bias_table = [[1, 2], [3, 4], [5, 6], [7, 8]].
  // Save in col-major order in ORT
  std::vector<float> bias_table = {1.f, 3.f, 5.f, 7.f, 2.f, 4.f, 6.f, 8.f};
  std::vector<int64_t> sequence_length = {seq_len};

  std::vector<float> output_data = {1.f, 7.f, 3.f, 1.f, 2.f, 8.f, 4.f, 2.f};

  RunRelativePositionBiasTest(bias_table,
                              sequence_length,
                              output_data,
                              max_distance,
                              num_buckets,
                              num_heads,
                              seq_len,
                              is_bidirectional);
}

TEST(RelativePositionBiasTest, RelativePositionBiasTest_FP16) {
  int max_distance = 128;
  int num_buckets = 4;
  int num_heads = 2;
  int seq_len = 2;
  int is_bidirectional = 1;

  // Huggingface bias_table = [[1, 2], [3, 4], [5, 6], [7, 8]].
  // Save in col-major order in ORT
  std::vector<float> bias_table = {1.f, 3.f, 5.f, 7.f, 2.f, 4.f, 6.f, 8.f};
  std::vector<int64_t> sequence_length = {seq_len};

  std::vector<float> output_data = {1.f, 7.f, 3.f, 1.f, 2.f, 8.f, 4.f, 2.f};

  RunRelativePositionBiasTest(bias_table,
                              sequence_length,
                              output_data,
                              max_distance,
                              num_buckets,
                              num_heads,
                              seq_len,
                              is_bidirectional,
                              true);
}

TEST(RelativePositionBiasTest, RelativePositionBiasTest2_FP16) {
  int max_distance = 128;
  int num_buckets = 4;
  int num_heads = 3;
  int seq_len = 2;
  int is_bidirectional = 1;

  // Huggingface bias_table = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]].
  // Save in col-major order in ORT
  std::vector<float> bias_table = {1.f, 4.f, 7.f, 10.f, 2.f, 5.f, 8.f, 11.f, 3.f, 6.f, 9.f, 12.f};
  std::vector<int64_t> sequence_length = {seq_len};

  std::vector<float> output_data = {1.f, 10.f, 4.f, 1.f, 2.f, 11.f, 5.f, 2.f, 3.f, 12.f, 6.f, 3.f};

  RunRelativePositionBiasTest(bias_table,
                              sequence_length,
                              output_data,
                              max_distance,
                              num_buckets,
                              num_heads,
                              seq_len,
                              is_bidirectional,
                              true);
}

TEST(RelativePositionBiasTest, RelativePositionBiasTest_FP16_No_Bidirectional) {
  int max_distance = 128;
  int num_buckets = 4;
  int num_heads = 3;
  int seq_len = 2;
  int is_bidirectional = 0;

  std::vector<float> bias_table = {1.f, 4.f, 7.f, 10.f, 2.f, 5.f, 8.f, 11.f, 3.f, 6.f, 9.f, 12.f};
  std::vector<int64_t> sequence_length = {seq_len};

  std::vector<float> output_data = {1.f, 1.f, 4.f, 1.f, 2.f, 2.f, 5.f, 2.f, 3.f, 3.f, 6.f, 3.f};

  RunRelativePositionBiasTest(bias_table,
                              sequence_length,
                              output_data,
                              max_distance,
                              num_buckets,
                              num_heads,
                              seq_len,
                              is_bidirectional,
                              true);
}

/***************Following scripts is used to generate test data, for your reference*************
import torch

batch_size = 2
num_heads = 2
seq_len = 3
head_size = 4
D = 8

def dim_string_of(tensor):
    return "{" +  ", ".join([str(d) for d in tensor.shape]) + "}"

def value_string_of(tensor):
    arr = tensor.flatten().numpy()
    lines = ["f, ".join([str(v) for v in arr[i : min(i+8, arr.size)]]) for i in range(0, arr.size, 8)]
    return "{\n    " + "f,\n    ".join(lines) + "f}"

def print_tensor(name, tensor):
    print(f"const std::vector<int64_t> {name}_dim = {dim_string_of(tensor)};")
    print(f"const std::vector<float> {name} = {value_string_of(tensor)};")

torch.manual_seed(0)
query_layer = torch.rand(batch_size, seq_len, num_heads * head_size)
query_bias = torch.rand(num_heads * head_size)
rel_pos = torch.rand(1, num_heads, seq_len, seq_len)
weight = torch.rand(head_size, D)
bias = torch.rand(D)
eco_a = torch.rand(1, num_heads, 1, 1)

qw = (query_layer + query_bias).reshape(batch_size, seq_len, num_heads, head_size).transpose(1, 2)
gate_u,gate_r = torch.sigmoid(
    (torch.matmul(qw, weight) + bias).view(batch_size, num_heads, seq_len,2, D//2).sum(-1, keepdim=False)
  ).chunk(2, dim=-1)
gate_u_1 = gate_u * (gate_r * eco_a - 1.0) + 2.0
output = gate_u_1 * rel_pos

# output for test case
print(f"const int batch_size = {batch_size};")
print(f"const int num_heads = {num_heads};")
print(f"const int seq_len = {seq_len};")
print(f"const int head_size = {head_size};")
print(f"const int D = {D};")

print_tensor("query_layer", query_layer)
print_tensor("query_bias", query_bias)
print_tensor("rel_pos", rel_pos)
print_tensor("weight", weight)
print_tensor("bias", bias)
print_tensor("eco_a", eco_a)
print_tensor("output", output)
****************/

// .Input(0, "query_layer", "tensor with shape (batch_size, seq_len, num_heads x head_size)", "T")
// .Input(1, "query_bias", "1-d tensor with shape (num_heads x head_size)", "T")
// .Input(2, "rel_pos", "tensor with shape (1, num_head, seq_len, seq_len)", "T")
// .Input(3, "weight", "gemm weight for the gated_ur_linear, shape (head_size, D), D is divisible by 2", "T")
// .Input(4, "bias", "bias for the gated_ur_linear, shape (D)", "T")
// .Input(5, "eco_a", "tensor of shape (1, num_heads, 1, 1)", "T")
// .Output(0, "output", "output tensor with shape (batch_size, num_heads, seq_len, seq_len)", "T")
static void RunGatedRelativePositionBiasTest(
    const std::vector<float>& query_layer,
    const std::vector<float>& query_bias,
    const std::vector<float>& rel_pos,
    const std::vector<float>& weight,
    const std::vector<float>& bias,
    const std::vector<float>& eco_a,
    const std::vector<float>& output,
    int batch_size,
    int seq_len,
    int num_heads,
    int head_size,
    int D,
    bool use_float16,
    const std::vector<int>& token_offset,
    int token_count) {
  int min_cuda_architecture = use_float16 ? 530 : 0;

  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  if (enable_cuda) {
    OpTester tester("GatedRelativePositionBias", 1, onnxruntime::kMSDomain);
    tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(num_heads));

    std::vector<int64_t> query_layer_dims = {batch_size, seq_len, num_heads * head_size};
    std::vector<int64_t> query_bias_dims = {num_heads * head_size};
    std::vector<int64_t> rel_pos_dims = {1, num_heads, seq_len, seq_len};
    std::vector<int64_t> weight_dims = {head_size, D};
    std::vector<int64_t> bias_dims = {D};
    std::vector<int64_t> eco_a_dims = {1, num_heads, 1, 1};
    std::vector<int64_t> output_dims = {batch_size, num_heads, seq_len, seq_len};

    std::vector<int64_t> packed_query_dims = {token_count, num_heads * head_size};
    std::vector<int64_t> token_offset_dims = {batch_size, seq_len};
    bool is_padding_removed = token_offset.size() > 0;

    if (use_float16) {
      tester.AddInput<MLFloat16>("query_layer", is_padding_removed ? packed_query_dims : query_layer_dims, ToFloat16(query_layer));
      tester.AddInput<MLFloat16>("query_bias", query_bias_dims, ToFloat16(query_bias));
      tester.AddInput<MLFloat16>("rel_pos", rel_pos_dims, ToFloat16(rel_pos));
      tester.AddInput<MLFloat16>("weight", weight_dims, ToFloat16(weight));
      tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias));
      tester.AddInput<MLFloat16>("eco_a", eco_a_dims, ToFloat16(eco_a));
      if (is_padding_removed) {
        tester.AddInput<int>("token_offset", token_offset_dims, token_offset);
      }
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output));
    } else {
      tester.AddInput<float>("query_layer", is_padding_removed ? packed_query_dims : query_layer_dims, query_layer);
      tester.AddInput<float>("query_bias", query_bias_dims, query_bias);
      tester.AddInput<float>("rel_pos", rel_pos_dims, rel_pos);
      tester.AddInput<float>("weight", weight_dims, weight);
      tester.AddInput<float>("bias", bias_dims, bias);
      tester.AddInput<float>("eco_a", eco_a_dims, eco_a);
      if (is_padding_removed) {
        tester.AddInput<int>("token_offset", token_offset_dims, token_offset);
      }

      tester.AddOutput<float>("output", output_dims, output);
    }

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCudaExecutionProvider());
    tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

TEST(GatedRelativePositionBiasTest, FP16_BSNHD_1x3x2x4x8) {
  constexpr int batch_size = 1;
  constexpr int num_heads = 2;
  constexpr int seq_len = 3;
  constexpr int head_size = 4;
  constexpr int D = 8;
  const std::vector<int64_t> query_layer_dim = {1, 3, 8};
  const std::vector<float> query_layer = {
      0.4962566f, 0.7682218f, 0.08847743f, 0.13203049f, 0.30742282f, 0.6340787f, 0.4900934f, 0.89644474f,
      0.45562798f, 0.6323063f, 0.34889346f, 0.4017173f, 0.022325754f, 0.16885895f, 0.29388845f, 0.5185218f,
      0.6976676f, 0.8000114f, 0.16102946f, 0.28226858f, 0.68160856f, 0.915194f, 0.3970999f, 0.8741559f};
  const std::vector<int64_t> query_bias_dim = {8};
  const std::vector<float> query_bias = {
      0.41940832f, 0.55290705f, 0.9527381f, 0.03616482f, 0.18523103f, 0.37341738f, 0.30510002f, 0.9320004f};
  const std::vector<int64_t> rel_pos_dim = {1, 2, 3, 3};
  const std::vector<float> rel_pos = {
      0.17591017f, 0.26983356f, 0.15067977f, 0.031719506f, 0.20812976f, 0.929799f, 0.7231092f, 0.7423363f,
      0.5262958f, 0.24365824f, 0.58459234f, 0.03315264f, 0.13871688f, 0.242235f, 0.81546897f, 0.7931606f,
      0.27825248f, 0.4819588f};
  const std::vector<int64_t> weight_dim = {4, 8};
  const std::vector<float> weight = {
      0.81978035f, 0.99706656f, 0.6984411f, 0.5675464f, 0.83524317f, 0.20559883f, 0.593172f, 0.112347245f,
      0.15345693f, 0.24170822f, 0.7262365f, 0.7010802f, 0.20382375f, 0.65105355f, 0.774486f, 0.43689132f,
      0.5190908f, 0.61585236f, 0.8101883f, 0.98009706f, 0.11468822f, 0.31676513f, 0.69650495f, 0.9142747f,
      0.93510365f, 0.9411784f, 0.5995073f, 0.06520867f, 0.54599625f, 0.18719733f, 0.034022927f, 0.94424623f};
  const std::vector<int64_t> bias_dim = {8};
  const std::vector<float> bias = {
      0.8801799f, 0.0012360215f, 0.593586f, 0.41577f, 0.41771942f, 0.27112156f, 0.6922781f, 0.20384824f};
  const std::vector<int64_t> eco_a_dim = {1, 2, 1, 1};
  const std::vector<float> eco_a = {
      0.68329567f, 0.75285405f};
  const std::vector<int64_t> output_dim = {1, 2, 3, 3};
  const std::vector<float> output = {
      0.29608122f, 0.45416728f, 0.25361493f, 0.053390637f, 0.3503264f, 1.5650483f, 1.2171557f, 1.2495192f,
      0.88587445f, 0.42708054f, 1.0246648f, 0.05810945f, 0.2430356f, 0.4244021f, 1.428723f, 1.3902748f,
      0.48772895f, 0.84479123f};

  const std::vector<int> token_offset;
  int token_count = 0;
  RunGatedRelativePositionBiasTest(query_layer, query_bias, rel_pos, weight, bias, eco_a, output,
                                   batch_size, seq_len, num_heads, head_size, D, true, token_offset, token_count);
}

TEST(GatedRelativePositionBiasTest, FP32_BSNHD_2x3x2x4x8) {
  constexpr int batch_size = 2;
  constexpr int num_heads = 2;
  constexpr int seq_len = 3;
  constexpr int head_size = 4;
  constexpr int D = 8;
  const std::vector<int64_t> query_layer_dim = {2, 3, 8};
  const std::vector<float> query_layer = {
      0.4962566f, 0.7682218f, 0.08847743f, 0.13203049f, 0.30742282f, 0.6340787f, 0.4900934f, 0.89644474f,
      0.45562798f, 0.6323063f, 0.34889346f, 0.4017173f, 0.022325754f, 0.16885895f, 0.29388845f, 0.5185218f,
      0.6976676f, 0.8000114f, 0.16102946f, 0.28226858f, 0.68160856f, 0.915194f, 0.3970999f, 0.8741559f,
      0.41940832f, 0.55290705f, 0.9527381f, 0.03616482f, 0.18523103f, 0.37341738f, 0.30510002f, 0.9320004f,
      0.17591017f, 0.26983356f, 0.15067977f, 0.031719506f, 0.20812976f, 0.929799f, 0.7231092f, 0.7423363f,
      0.5262958f, 0.24365824f, 0.58459234f, 0.03315264f, 0.13871688f, 0.242235f, 0.81546897f, 0.7931606f};
  const std::vector<int64_t> query_bias_dim = {8};
  const std::vector<float> query_bias = {
      0.27825248f, 0.4819588f, 0.81978035f, 0.99706656f, 0.6984411f, 0.5675464f, 0.83524317f, 0.20559883f};
  const std::vector<int64_t> rel_pos_dim = {1, 2, 3, 3};
  const std::vector<float> rel_pos = {
      0.593172f, 0.112347245f, 0.15345693f, 0.24170822f, 0.7262365f, 0.7010802f, 0.20382375f, 0.65105355f,
      0.774486f, 0.43689132f, 0.5190908f, 0.61585236f, 0.8101883f, 0.98009706f, 0.11468822f, 0.31676513f,
      0.69650495f, 0.9142747f};
  const std::vector<int64_t> weight_dim = {4, 8};
  const std::vector<float> weight = {
      0.93510365f, 0.9411784f, 0.5995073f, 0.06520867f, 0.54599625f, 0.18719733f, 0.034022927f, 0.94424623f,
      0.8801799f, 0.0012360215f, 0.593586f, 0.41577f, 0.41771942f, 0.27112156f, 0.6922781f, 0.20384824f,
      0.68329567f, 0.75285405f, 0.8579358f, 0.6869556f, 0.005132377f, 0.17565155f, 0.7496575f, 0.6046507f,
      0.10995799f, 0.21209025f, 0.97037464f, 0.83690894f, 0.28198743f, 0.3741576f, 0.023700953f, 0.49101293f};
  const std::vector<int64_t> bias_dim = {8};
  const std::vector<float> bias = {
      0.123470545f, 0.11432165f, 0.4724502f, 0.5750725f, 0.29523486f, 0.7966888f, 0.19573045f, 0.95368505f};
  const std::vector<int64_t> eco_a_dim = {1, 2, 1, 1};
  const std::vector<float> eco_a = {
      0.84264994f, 0.07835853f};
  const std::vector<int64_t> output_dim = {2, 2, 3, 3};
  const std::vector<float> output = {
      1.0928818f, 0.20699267f, 0.28273466f, 0.44534987f, 1.3380982f, 1.2917475f, 0.3755537f, 1.1995932f,
      1.4270226f, 0.47112367f, 0.5597638f, 0.6641071f, 0.87368786f, 1.0569134f, 0.12367705f, 0.34158573f,
      0.75108063f, 0.98591405f, 1.0929474f, 0.2070051f, 0.28275162f, 0.4451845f, 1.3376014f, 1.2912678f,
      0.37552574f, 1.1995038f, 1.4269164f, 0.47112313f, 0.5597632f, 0.6641063f, 0.87367094f, 1.056893f,
      0.12367466f, 0.34158388f, 0.7510766f, 0.98590875f};

  const std::vector<int> token_offset;
  int token_count = 0;
  RunGatedRelativePositionBiasTest(query_layer, query_bias, rel_pos, weight, bias, eco_a, output,
                                   batch_size, seq_len, num_heads, head_size, D, false, token_offset, token_count);
}

TEST(GatedRelativePositionBiasTest, FP32_LongSeq_BSNHD_2x5x2x4x4) {
  constexpr int batch_size = 2;
  constexpr int num_heads = 2;
  constexpr int seq_len = 5;
  constexpr int head_size = 4;
  constexpr int D = 4;
  const std::vector<int64_t> query_layer_dim = {2, 5, 8};
  const std::vector<float> query_layer = {
      0.4962566f, 0.7682218f, 0.08847743f, 0.13203049f, 0.30742282f, 0.6340787f, 0.4900934f, 0.89644474f,
      0.45562798f, 0.6323063f, 0.34889346f, 0.4017173f, 0.022325754f, 0.16885895f, 0.29388845f, 0.5185218f,
      0.6976676f, 0.8000114f, 0.16102946f, 0.28226858f, 0.68160856f, 0.915194f, 0.3970999f, 0.8741559f,
      0.41940832f, 0.55290705f, 0.9527381f, 0.03616482f, 0.18523103f, 0.37341738f, 0.30510002f, 0.9320004f,
      0.17591017f, 0.26983356f, 0.15067977f, 0.031719506f, 0.20812976f, 0.929799f, 0.7231092f, 0.7423363f,
      0.5262958f, 0.24365824f, 0.58459234f, 0.03315264f, 0.13871688f, 0.242235f, 0.81546897f, 0.7931606f,
      0.27825248f, 0.4819588f, 0.81978035f, 0.99706656f, 0.6984411f, 0.5675464f, 0.83524317f, 0.20559883f,
      0.593172f, 0.112347245f, 0.15345693f, 0.24170822f, 0.7262365f, 0.7010802f, 0.20382375f, 0.65105355f,
      0.774486f, 0.43689132f, 0.5190908f, 0.61585236f, 0.8101883f, 0.98009706f, 0.11468822f, 0.31676513f,
      0.69650495f, 0.9142747f, 0.93510365f, 0.9411784f, 0.5995073f, 0.06520867f, 0.54599625f, 0.18719733f};
  const std::vector<int64_t> query_bias_dim = {8};
  const std::vector<float> query_bias = {
      0.034022927f, 0.94424623f, 0.8801799f, 0.0012360215f, 0.593586f, 0.41577f, 0.41771942f, 0.27112156f};
  const std::vector<int64_t> rel_pos_dim = {1, 2, 5, 5};
  const std::vector<float> rel_pos = {
      0.6922781f, 0.20384824f, 0.68329567f, 0.75285405f, 0.8579358f, 0.6869556f, 0.005132377f, 0.17565155f,
      0.7496575f, 0.6046507f, 0.10995799f, 0.21209025f, 0.97037464f, 0.83690894f, 0.28198743f, 0.3741576f,
      0.023700953f, 0.49101293f, 0.123470545f, 0.11432165f, 0.4724502f, 0.5750725f, 0.29523486f, 0.7966888f,
      0.19573045f, 0.95368505f, 0.84264994f, 0.07835853f, 0.37555784f, 0.5225613f, 0.57295054f, 0.61858714f,
      0.69621414f, 0.5299501f, 0.25603563f, 0.7365945f, 0.02037555f, 0.20364666f, 0.37483507f, 0.25644332f,
      0.32508332f, 0.09018916f, 0.39364243f, 0.6068782f, 0.17426711f, 0.47434032f, 0.8579254f, 0.44859987f,
      0.5138961f, 0.45686555f};
  const std::vector<int64_t> weight_dim = {4, 4};
  const std::vector<float> weight = {
      0.6011907f, 0.81791973f, 0.9736231f, 0.81752795f, 0.97470677f, 0.46383917f, 0.050839245f, 0.2629614f,
      0.8404526f, 0.49675876f, 0.25147682f, 0.11684412f, 0.032073975f, 0.0779959f, 0.39858162f, 0.774203f};
  const std::vector<int64_t> bias_dim = {4};
  const std::vector<float> bias = {
      0.77032053f, 0.017784059f, 0.811891f, 0.10874528f};
  const std::vector<int64_t> eco_a_dim = {1, 2, 1, 1};
  const std::vector<float> eco_a = {
      0.39429486f, 0.29726368f};
  const std::vector<int64_t> output_dim = {2, 2, 5, 5};
  const std::vector<float> output = {
      0.9534052f, 0.28073975f, 0.9410346f, 1.0368304f, 1.181549f, 0.94923383f, 0.0070919087f, 0.24271497f,
      1.0358753f, 0.8355051f, 0.15224966f, 0.29366368f, 1.3435968f, 1.158798f, 0.3904445f, 0.5147038f,
      0.03260383f, 0.67545396f, 0.16985025f, 0.15726471f, 0.64280313f, 0.7824283f, 0.40168867f, 1.0839535f,
      0.26630563f, 1.2391479f, 1.0948771f, 0.101813294f, 0.48797214f, 0.6789776f, 0.7492329f, 0.8089107f,
      0.91042155f, 0.6930023f, 0.3348113f, 0.95611423f, 0.026447866f, 0.2643374f, 0.48654333f, 0.3328685f,
      0.4239932f, 0.117630124f, 0.5134121f, 0.7915271f, 0.22728965f, 0.61497897f, 1.1122944f, 0.5816067f,
      0.6662628f, 0.59232306f, 0.95294285f, 0.2806036f, 0.9405782f, 1.0363276f, 1.1809759f, 0.95289487f,
      0.007119261f, 0.24365108f, 1.0398705f, 0.83872753f, 0.15201466f, 0.29321042f, 1.3415229f, 1.1570094f,
      0.38984182f, 0.51978874f, 0.032925934f, 0.682127f, 0.17152825f, 0.15881838f, 0.6571103f, 0.79984313f,
      0.4106292f, 1.1080796f, 0.2722329f, 1.2398669f, 1.0955123f, 0.101872355f, 0.4882552f, 0.6793715f,
      0.7427765f, 0.8019401f, 0.9025762f, 0.6870305f, 0.33192614f, 0.9568577f, 0.026468432f, 0.26454294f,
      0.48692167f, 0.33312735f, 0.4217717f, 0.117013805f, 0.5107221f, 0.78737986f, 0.22609876f, 0.6166911f,
      1.1153911f, 0.5832259f, 0.6681177f, 0.59397215f};

  const std::vector<int> token_offset;
  int token_count = 0;
  RunGatedRelativePositionBiasTest(query_layer, query_bias, rel_pos, weight, bias, eco_a, output,
                                   batch_size, seq_len, num_heads, head_size, D, false, token_offset, token_count);
}

TEST(GatedRelativePositionBiasTest, FP16_BSNHD_2x8x2x4x8_NoPadding) {
  constexpr int batch_size = 2;
  constexpr int num_heads = 2;
  constexpr int seq_len = 8;
  constexpr int head_size = 4;
  constexpr int D = 8;
  const std::vector<int64_t> query_layer_dim = {16, 8};
  const std::vector<float> query_layer = {
      0.4962566f, 0.7682218f, 0.08847743f, 0.13203049f, 0.30742282f, 0.6340787f, 0.4900934f, 0.89644474f,
      0.45562798f, 0.6323063f, 0.34889346f, 0.4017173f, 0.022325754f, 0.16885895f, 0.29388845f, 0.5185218f,
      0.6976676f, 0.8000114f, 0.16102946f, 0.28226858f, 0.68160856f, 0.915194f, 0.3970999f, 0.8741559f,
      0.41940832f, 0.55290705f, 0.9527381f, 0.03616482f, 0.18523103f, 0.37341738f, 0.30510002f, 0.9320004f,
      0.17591017f, 0.26983356f, 0.15067977f, 0.031719506f, 0.20812976f, 0.929799f, 0.7231092f, 0.7423363f,
      0.5262958f, 0.24365824f, 0.58459234f, 0.03315264f, 0.13871688f, 0.242235f, 0.81546897f, 0.7931606f,
      0.27825248f, 0.4819588f, 0.81978035f, 0.99706656f, 0.6984411f, 0.5675464f, 0.83524317f, 0.20559883f,
      0.593172f, 0.112347245f, 0.15345693f, 0.24170822f, 0.7262365f, 0.7010802f, 0.20382375f, 0.65105355f,
      0.774486f, 0.43689132f, 0.5190908f, 0.61585236f, 0.8101883f, 0.98009706f, 0.11468822f, 0.31676513f,
      0.69650495f, 0.9142747f, 0.93510365f, 0.9411784f, 0.5995073f, 0.06520867f, 0.54599625f, 0.18719733f,
      0.034022927f, 0.94424623f, 0.8801799f, 0.0012360215f, 0.593586f, 0.41577f, 0.41771942f, 0.27112156f,
      0.6922781f, 0.20384824f, 0.68329567f, 0.75285405f, 0.8579358f, 0.6869556f, 0.005132377f, 0.17565155f,
      0.7496575f, 0.6046507f, 0.10995799f, 0.21209025f, 0.97037464f, 0.83690894f, 0.28198743f, 0.3741576f,
      0.023700953f, 0.49101293f, 0.123470545f, 0.11432165f, 0.4724502f, 0.5750725f, 0.29523486f, 0.7966888f,
      0.19573045f, 0.95368505f, 0.84264994f, 0.07835853f, 0.37555784f, 0.5225613f, 0.57295054f, 0.61858714f,
      0.69621414f, 0.5299501f, 0.25603563f, 0.7365945f, 0.02037555f, 0.20364666f, 0.37483507f, 0.25644332f};
  const std::vector<int64_t> query_bias_dim = {8};
  const std::vector<float> query_bias = {
      0.32508332f, 0.09018916f, 0.39364243f, 0.6068782f, 0.17426711f, 0.47434032f, 0.8579254f, 0.44859987f};
  const std::vector<int64_t> rel_pos_dim = {1, 2, 8, 8};
  const std::vector<float> rel_pos = {
      0.5138961f, 0.45686555f, 0.6011907f, 0.81791973f, 0.9736231f, 0.81752795f, 0.97470677f, 0.46383917f,
      0.050839245f, 0.2629614f, 0.8404526f, 0.49675876f, 0.25147682f, 0.11684412f, 0.032073975f, 0.0779959f,
      0.39858162f, 0.774203f, 0.77032053f, 0.017784059f, 0.811891f, 0.10874528f, 0.39429486f, 0.29726368f,
      0.40369236f, 0.40182865f, 0.051325023f, 0.068281054f, 0.42176026f, 0.5064661f, 0.27286255f, 0.6883496f,
      0.049970806f, 0.46625638f, 0.9397097f, 0.296054f, 0.95150155f, 0.6810769f, 0.048769534f, 0.8163487f,
      0.44230276f, 0.27679658f, 0.89982665f, 0.09595239f, 0.55365247f, 0.39531565f, 0.8570563f, 0.63957226f,
      0.7402527f, 0.6765795f, 0.37976265f, 0.39484727f, 0.08795929f, 0.77092206f, 0.89698905f, 0.8421124f,
      0.14731085f, 0.52229995f, 0.14753294f, 0.22475791f, 0.20864725f, 0.6708725f, 0.20204341f, 0.4890914f,
      0.52103406f, 0.8223115f, 0.122039974f, 0.15674388f, 0.20966923f, 0.8499667f, 0.3202675f, 0.92174435f,
      0.6808038f, 0.563313f, 0.496278f, 0.40115923f, 0.5627332f, 0.38582766f, 0.49648678f, 0.5637965f,
      0.10889745f, 0.23793429f, 0.90374637f, 0.09422666f, 0.4640969f, 0.99461937f, 0.6806185f, 0.5141565f,
      0.066695035f, 0.74768895f, 0.14385962f, 0.35806787f, 0.33224183f, 0.4259563f, 0.50546914f, 0.91240376f,
      0.5624194f, 0.9478464f, 0.8058562f, 0.18389302f, 0.72425205f, 0.14655197f, 0.28808743f, 0.64706135f,
      0.66509604f, 0.875114f, 0.33904207f, 0.50080043f, 0.7574118f, 0.016453922f, 0.8614903f, 0.08653879f,
      0.50689125f, 0.41499162f, 0.23666352f, 0.5660855f, 0.91345936f, 0.35384023f, 0.20315295f, 0.31508058f,
      0.0044258237f, 0.725697f, 0.25986814f, 0.16632986f, 0.21194929f, 0.787478f, 0.76478684f, 0.8837609f};
  const std::vector<int64_t> weight_dim = {4, 8};
  const std::vector<float> weight = {
      0.68136156f, 0.33302015f, 0.36027592f, 0.647715f, 0.91101736f, 0.6359461f, 0.26342732f, 0.2649613f,
      0.02726549f, 0.608024f, 0.21940875f, 0.054212093f, 0.93843824f, 0.1752944f, 0.44311923f, 0.64324677f,
      0.51592916f, 0.16355914f, 0.09583914f, 0.8985412f, 0.58141935f, 0.91481227f, 0.3323797f, 0.6472777f,
      0.3856619f, 0.47776443f, 0.1954779f, 0.66910046f, 0.65808296f, 0.4896857f, 0.38754892f, 0.1917851f};
  const std::vector<int64_t> bias_dim = {8};
  const std::vector<float> bias = {
      0.8457724f, 0.12778795f, 0.70483273f, 0.33187324f, 0.258766f, 0.58982253f, 0.24027151f, 0.6152024f};
  const std::vector<int64_t> eco_a_dim = {1, 2, 1, 1};
  const std::vector<float> eco_a = {
      0.5981904f, 0.12875527f};
  const std::vector<int64_t> output_dim = {2, 2, 8, 8};
  const std::vector<float> output = {
      0.8214731f, 0.73030865f, 0.9610152f, 1.3074609f, 1.5563558f, 1.3068346f, 1.5580881f, 0.7414561f,
      0.08125934f, 0.42030656f, 1.3433446f, 0.7939986f, 0.40195012f, 0.18675879f, 0.05126571f, 0.1246654f,
      0.63707197f, 1.2374455f, 1.2312399f, 0.028425107f, 1.2976841f, 0.17381275f, 0.63022023f, 0.47513068f,
      0.6452434f, 0.6422645f, 0.082035564f, 0.10913731f, 0.6741223f, 0.8095122f, 0.436131f, 1.1002265f,
      0.07988175f, 0.7453427f, 1.5021902f, 0.47326255f, 1.5210402f, 1.088748f, 0.07796143f, 1.3049891f,
      0.7069702f, 0.44242755f, 1.4382696f, 0.15336889f, 0.8849499f, 0.6318667f, 1.3699062f, 1.0222828f,
      1.1831017f, 1.0813366f, 0.606952f, 0.63106084f, 0.14058009f, 1.2321187f, 1.4336041f, 1.345898f,
      0.235435f, 0.83474964f, 0.23578994f, 0.35921234f, 0.333464f, 1.0722011f, 0.3229096f, 0.7816751f,
      0.58820444f, 0.92832184f, 0.13777305f, 0.1769509f, 0.23669925f, 0.9595423f, 0.36155558f, 1.0405732f,
      0.76924545f, 0.6364917f, 0.56074834f, 0.4532729f, 0.63583654f, 0.43594965f, 0.56098425f, 0.63703805f,
      0.12292639f, 0.26858667f, 1.0201734f, 0.106365606f, 0.52388513f, 1.1227533f, 0.76830065f, 0.5803938f,
      0.075304635f, 0.84420747f, 0.16243033f, 0.40429053f, 0.37513062f, 0.48094264f, 0.5707197f, 1.0301851f,
      0.6349097f, 1.0700144f, 0.90972304f, 0.207595f, 0.8176009f, 0.16544105f, 0.32521904f, 0.7304611f,
      0.75088024f, 0.9879864f, 0.38277176f, 0.56539375f, 0.8551029f, 0.01857615f, 0.97260547f, 0.09770058f,
      0.57222974f, 0.4684842f, 0.26716954f, 0.6390542f, 1.0312047f, 0.3994504f, 0.22933945f, 0.3556946f,
      0.004996385f, 0.81925124f, 0.2933694f, 0.1877725f, 0.23927303f, 0.88899684f, 0.8633804f, 0.99769217f,
      0.8213299f, 0.7301813f, 0.9608477f, 1.307233f, 1.5560844f, 1.3066068f, 1.5578164f, 0.7413268f,
      0.08125164f, 0.42026675f, 1.3432174f, 0.7939234f, 0.40191206f, 0.1867411f, 0.051260855f, 0.1246536f,
      0.63714564f, 1.2375886f, 1.2313824f, 0.028428394f, 1.2978342f, 0.17383285f, 0.63029313f, 0.47518563f,
      0.64519227f, 0.6422136f, 0.08202907f, 0.10912866f, 0.67406887f, 0.80944806f, 0.43609643f, 1.1001393f,
      0.07987102f, 0.7452426f, 1.5019884f, 0.47319898f, 1.520836f, 1.0886017f, 0.07795096f, 1.3048139f,
      0.70720285f, 0.44257313f, 1.4387429f, 0.15341935f, 0.8852411f, 0.6320746f, 1.3703569f, 1.0226192f,
      1.1832331f, 1.0814568f, 0.6070194f, 0.63113093f, 0.1405957f, 1.2322556f, 1.4337634f, 1.3460475f,
      0.23544003f, 0.8347675f, 0.23579498f, 0.35922003f, 0.33347112f, 1.0722241f, 0.3229165f, 0.78169185f,
      0.5882341f, 0.9283687f, 0.13778001f, 0.17695984f, 0.2367112f, 0.9595907f, 0.36157382f, 1.0406258f,
      0.768774f, 0.6361016f, 0.56040466f, 0.45299512f, 0.63544685f, 0.43568248f, 0.56064045f, 0.63664764f,
      0.12295848f, 0.2686568f, 1.0204397f, 0.106393375f, 0.5240219f, 1.1230464f, 0.7685012f, 0.5805453f,
      0.07530872f, 0.8442532f, 0.16243914f, 0.40431243f, 0.37515095f, 0.48096868f, 0.5707506f, 1.030241f,
      0.6349034f, 1.0700037f, 0.90971404f, 0.20759295f, 0.8175928f, 0.16543941f, 0.3252158f, 0.73045385f,
      0.7508644f, 0.9879655f, 0.38276368f, 0.5653818f, 0.85508484f, 0.01857576f, 0.9725849f, 0.09769852f,
      0.57226765f, 0.46851522f, 0.26718724f, 0.6390965f, 1.0312729f, 0.39947683f, 0.22935463f, 0.35571814f,
      0.005002509f, 0.82025534f, 0.29372898f, 0.18800265f, 0.2395663f, 0.8900865f, 0.8644386f, 0.998915f};

  const std::vector<int> token_offset = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
  int token_count = 16;
  RunGatedRelativePositionBiasTest(query_layer, query_bias, rel_pos, weight, bias, eco_a, output,
                                   batch_size, seq_len, num_heads, head_size, D, true, token_offset, token_count);
}

}  // namespace test
}  // namespace onnxruntime
