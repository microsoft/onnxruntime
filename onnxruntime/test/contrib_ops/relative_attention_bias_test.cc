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
  bool enable_cpu = false;
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
  const int batch_size = 1;
  const int num_heads = 2;
  const int seq_len = 3;
  const int head_size = 4;
  const int D = 8;
  const std::vector<int64_t> query_layer_dim = {1, 3, 8};
  const std::vector<float> query_layer = {
      0.45471495f, 0.8266302f, 0.21814579f, 0.958557f, 0.8630423f, 0.07658166f, 0.6217646f, 0.7699812f,
      0.43557996f, 0.88360316f, 0.8031714f, 0.94664145f, 0.37502897f, 0.6553267f, 0.09649378f, 0.41613388f,
      0.13720119f, 0.5530107f, 0.9206199f, 0.7022264f, 0.7214146f, 0.4392771f, 0.2618755f, 0.096226454f};
  const std::vector<int64_t> query_bias_dim = {8};
  const std::vector<float> query_bias = {
      0.0005787611f, 0.80999535f, 0.9992162f, 0.050085425f, 0.33461642f, 0.5555024f, 0.044624865f, 0.20329022f};
  const std::vector<int64_t> rel_pos_dim = {1, 2, 3, 3};
  const std::vector<float> rel_pos = {
      0.45087618f, 0.12421924f, 0.77322024f, 0.41314727f, 0.36701924f, 0.16967869f, 0.08856118f, 0.3851894f,
      0.24701673f, 0.23489982f, 0.34972727f, 0.6124148f, 0.64025134f, 0.9491165f, 0.73348546f, 0.60984415f,
      0.8117957f, 0.47210503f};
  const std::vector<int64_t> weight_dim = {4, 8};
  const std::vector<float> weight = {
      0.11955571f, 0.16296995f, 0.023296416f, 0.018002152f, 0.5247776f, 0.6783996f, 0.5295885f, 0.21289009f,
      0.6200221f, 0.69121385f, 0.30983877f, 0.5228178f, 0.94767714f, 0.8823836f, 0.74784315f, 0.0042557716f,
      0.59795046f, 0.6652346f, 0.40319276f, 0.16138184f, 0.36803174f, 0.8221892f, 0.9672306f, 0.9401135f,
      0.81025666f, 0.3004946f, 0.22577477f, 0.49245697f, 0.16895646f, 0.2150622f, 0.037842274f, 0.09327704f};
  const std::vector<int64_t> bias_dim = {8};
  const std::vector<float> bias = {
      0.60208213f, 0.37420756f, 0.48892576f, 0.9704983f, 0.13036764f, 0.8157921f, 0.024861515f, 0.9645103f};
  const std::vector<int64_t> eco_a_dim = {1, 2, 1, 1};
  const std::vector<float> eco_a = {
      0.03187573f, 0.2050497f};
  const std::vector<int64_t> output_dim = {1, 2, 3, 3};
  const std::vector<float> output = {
      0.4652649f, 0.12818342f, 0.79789585f, 0.42632142f, 0.3787225f, 0.17508928f, 0.09138703f, 0.39748025f,
      0.25489867f, 0.28319836f, 0.42163587f, 0.73833543f, 0.7720931f, 1.1445603f, 0.8845262f, 0.7360108f,
      0.97974277f, 0.56977576f};

  RunGatedRelativePositionBiasTest(query_layer, query_bias, rel_pos, weight, bias, eco_a, output,
                                   batch_size, seq_len, num_heads, head_size, D, true);
}

TEST(GatedRelativePositionBiasTest, FP32_BSNHD_2x3x2x4x8) {
  const int batch_size = 2;
  const int num_heads = 2;
  const int seq_len = 3;
  const int head_size = 4;
  const int D = 8;
  const std::vector<int64_t> query_layer_dim = {2, 3, 8};
  const std::vector<float> query_layer = {
      0.48441923f, 0.7150636f, 0.76932335f, 0.25310212f, 0.12356353f, 0.6776099f, 0.62411565f, 0.49144292f,
      0.13192421f, 0.503151f, 0.20556897f, 0.34899718f, 0.8495561f, 0.5113745f, 0.42068988f, 0.9465628f,
      0.37419575f, 0.07972479f, 0.2743234f, 0.56728524f, 0.8628615f, 0.21920788f, 0.983549f, 0.36597806f,
      0.95083374f, 0.042520583f, 0.18990517f, 0.23058462f, 0.7244218f, 0.97414017f, 0.759224f, 0.9526477f,
      0.46026123f, 0.16701996f, 0.49727958f, 0.5861346f, 0.16507226f, 0.31876326f, 0.92014956f, 0.17148745f,
      0.9565408f, 0.0022149086f, 0.2358771f, 0.61508197f, 0.24634916f, 0.71944356f, 0.84923774f, 0.93440473f};
  const std::vector<int64_t> query_bias_dim = {8};
  const std::vector<float> query_bias = {
      0.10596174f, 0.48349845f, 0.90227616f, 0.027397573f, 0.41862023f, 0.8646208f, 0.53739303f, 0.66040105f};
  const std::vector<int64_t> rel_pos_dim = {1, 2, 3, 3};
  const std::vector<float> rel_pos = {
      0.18819857f, 0.34335667f, 0.18107283f, 0.6133745f, 0.46577823f, 0.31295186f, 0.36079818f, 0.2621042f,
      0.24176604f, 0.89215153f, 0.7353353f, 0.5616637f, 0.49318773f, 0.72385865f, 0.2917483f, 0.18357253f,
      0.8809989f, 0.8066003f};
  const std::vector<int64_t> weight_dim = {4, 8};
  const std::vector<float> weight = {
      0.73570204f, 0.47519857f, 0.57706034f, 0.8294208f, 0.19541532f, 0.76928985f, 0.9420716f, 0.86534625f,
      0.4833178f, 0.26062173f, 0.32717f, 0.11352968f, 0.56279147f, 0.32844353f, 0.6513979f, 0.46384662f,
      0.62156385f, 0.5171951f, 0.6437909f, 0.90714616f, 0.70234066f, 0.89185834f, 0.4795494f, 0.09910345f,
      0.6939806f, 0.44974977f, 0.29420745f, 0.0039123893f, 0.9451477f, 0.52628654f, 0.43058908f, 0.27354598f};
  const std::vector<int64_t> bias_dim = {8};
  const std::vector<float> bias = {
      0.7507707f, 0.55417097f, 0.20892018f, 0.3523249f, 0.9647635f, 0.040732622f, 0.92307454f, 0.7217171f};
  const std::vector<int64_t> eco_a_dim = {1, 2, 1, 1};
  const std::vector<float> eco_a = {
      0.7025448f, 0.15687895f};
  const std::vector<int64_t> output_dim = {2, 2, 3, 3};
  const std::vector<float> output = {
      0.32041746f, 0.58458185f, 0.30828553f, 1.0443501f, 0.7930482f, 0.5328414f, 0.614286f, 0.4462521f,
      0.41162482f, 1.0321486f, 0.8507246f, 0.6498003f, 0.5705619f, 0.83742183f, 0.33751947f, 0.21237206f,
      1.0192133f, 0.9331427f, 0.32041794f, 0.58458275f, 0.30828598f, 1.0443044f, 0.79301345f, 0.532818f,
      0.6142784f, 0.44624656f, 0.41161972f, 1.0321132f, 0.8506955f, 0.6497781f, 0.57057834f, 0.8374459f,
      0.33752918f, 0.21237274f, 1.0192164f, 0.9331456f};

  RunGatedRelativePositionBiasTest(query_layer, query_bias, rel_pos, weight, bias, eco_a, output,
                                   batch_size, seq_len, num_heads, head_size, D, false);
}

}  // namespace test
}  // namespace onnxruntime
