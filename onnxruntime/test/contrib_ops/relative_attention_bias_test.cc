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
    bool use_float16 = false) {
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

    if (use_float16) {
      tester.AddInput<MLFloat16>("query_layer", query_layer_dims, ToFloat16(query_layer));
      tester.AddInput<MLFloat16>("query_bias", query_bias_dims, ToFloat16(query_bias));
      tester.AddInput<MLFloat16>("rel_pos", rel_pos_dims, ToFloat16(rel_pos));
      tester.AddInput<MLFloat16>("weight", weight_dims, ToFloat16(weight));
      tester.AddInput<MLFloat16>("bias", bias_dims, ToFloat16(bias));
      tester.AddInput<MLFloat16>("eco_a", eco_a_dims, ToFloat16(eco_a));
      tester.AddOutput<MLFloat16>("output", output_dims, ToFloat16(output));
    } else {
      tester.AddInput<float>("query_layer", query_layer_dims, query_layer);
      tester.AddInput<float>("query_bias", query_bias_dims, query_bias);
      tester.AddInput<float>("rel_pos", rel_pos_dims, rel_pos);
      tester.AddInput<float>("weight", weight_dims, weight);
      tester.AddInput<float>("bias", bias_dims, bias);
      tester.AddInput<float>("eco_a", eco_a_dims, eco_a);
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

  RunGatedRelativePositionBiasTest(query_layer, query_bias, rel_pos, weight, bias, eco_a, output,
                                   batch_size, seq_len, num_heads, head_size, D, true);
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

  RunGatedRelativePositionBiasTest(query_layer, query_bias, rel_pos, weight, bias, eco_a, output,
                                   batch_size, seq_len, num_heads, head_size, D, false);
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

  RunGatedRelativePositionBiasTest(query_layer, query_bias, rel_pos, weight, bias, eco_a, output,
                                   batch_size, seq_len, num_heads, head_size, D, false);
}

}  // namespace test
}  // namespace onnxruntime
