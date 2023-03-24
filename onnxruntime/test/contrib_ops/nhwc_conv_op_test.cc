// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"

using namespace std;
namespace onnxruntime {
namespace test {

namespace {

struct NhwcConvOpAndTestAttributes {
  string auto_pad;
  vector<int64_t> dilations;
  int64_t group;
  vector<int64_t> kernel_shape;
  vector<int64_t> pads;
  vector<int64_t> strides;
  std::unordered_set<std::string> excluded_providers;
};

void TestNhwcConvOp(const NhwcConvOpAndTestAttributes& attributes,
                    const vector<vector<float>>& inputs,
                    const vector<vector<int64_t>>& input_shapes,
                    const std::initializer_list<float>& expected_output,
                    const vector<int64_t>& expected_output_shape,
                    bool use_float16,
                    bool weight_is_initializer = false) {
  int min_cuda_architecture = use_float16 ? 530 : 0;
  bool enable_cuda = HasCudaEnvironment(min_cuda_architecture);
  bool enable_rocm = (nullptr != DefaultRocmExecutionProvider().get());
  bool enable_dml = (nullptr != DefaultDmlExecutionProvider().get());

  if (enable_cuda || enable_rocm || enable_dml) {
    OpTester test("NhwcConv", 1, onnxruntime::kMSDomain);
    test.AddAttribute("group", attributes.group);
    test.AddAttribute("kernel_shape", attributes.kernel_shape);

    if (!attributes.dilations.empty()) {
      test.AddAttribute("dilations", attributes.dilations);
    }

    // Only one of pads / auto_pad can be present
    if (!attributes.pads.empty()) {
      test.AddAttribute("pads", attributes.pads);
    } else {
      test.AddAttribute("auto_pad", attributes.auto_pad);
    }

    if (!attributes.strides.empty()) {
      test.AddAttribute("strides", attributes.strides);
    }

    ORT_ENFORCE(inputs.size() <= 3, "Our name array is only setup to handle 3 inputs");
    const char* szNames[] = {"X", "W", "B"};

    if (use_float16) {
      test.AddInput<MLFloat16>(szNames[0], input_shapes[0], ToFloat16(inputs[0]));
      test.AddInput<MLFloat16>(szNames[1], input_shapes[1], ToFloat16(inputs[1]), weight_is_initializer);
      if (inputs.size() == 3) {
        test.AddInput<MLFloat16>(szNames[2], input_shapes[2], ToFloat16(inputs[2]));
      }
      test.AddOutput<MLFloat16>("Y", expected_output_shape, ToFloat16(expected_output));
    } else {
      test.AddInput<float>(szNames[0], input_shapes[0], inputs[0]);
      test.AddInput<float>(szNames[1], input_shapes[1], inputs[1], weight_is_initializer);
      if (inputs.size() == 3) {
        test.AddInput<float>(szNames[2], input_shapes[2], inputs[2]);
      }
      test.AddOutput<float>("Y", expected_output_shape, expected_output);
    }

    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;

    if (enable_cuda) {
      execution_providers.push_back(DefaultCudaExecutionProvider());
    }

    if (enable_rocm) {
      execution_providers.push_back(DefaultRocmExecutionProvider());
    }

    if (enable_dml) {
      execution_providers.push_back(DefaultDmlExecutionProvider());
    }

    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
  }
}

void RunNhwcConv(const NhwcConvOpAndTestAttributes& attributes,
                 const vector<vector<float>>& inputs,
                 const vector<vector<int64_t>>& input_shapes,
                 const std::initializer_list<float>& expected_output,
                 const vector<int64_t>& expected_output_shape) {
  bool use_float16 = true;
  bool weight_is_initializer = true;
  TestNhwcConvOp(attributes, inputs, input_shapes, expected_output, expected_output_shape, use_float16, weight_is_initializer);

  use_float16 = false;
  weight_is_initializer = false;
  TestNhwcConvOp(attributes, inputs, input_shapes, expected_output, expected_output_shape, use_float16, weight_is_initializer);
}

}  // namespace

TEST(NhwcConvTest, Conv2D_2) {
  NhwcConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{1, 1},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      {}                            // excluded EPs
  };

  vector<float> X = {
      0.45246148109436035f, 0.15498268604278564f, 0.11199361085891724f, -0.39421093463897705f,
      0.2626858949661255f, 0.13414543867111206f, -0.27184486389160156f, -0.43028733134269714f,
      -0.26825493574142456f, 0.3893144130706787f, -0.13631996512413025f, -0.009590476751327515f,
      -0.48771554231643677f, -0.25256502628326416f, -0.2812897562980652f, 0.4043201804161072f,
      0.07795023918151855f, 0.326981782913208f, 0.13114392757415771f, -0.4416425824165344f,
      0.12446999549865723f, 0.36739975214004517f, 0.1698915958404541f, 0.2008744478225708f,
      0.23339951038360596f, 0.38613730669021606f, 0.11117297410964966f, 0.3877097964286804f,
      0.20812749862670898f, -0.34297940135002136f, -0.029246658086776733f, -0.20483523607254028f,
      -0.19244328141212463f, -0.11104947328567505f, -0.32830488681793213f, -0.01800677180290222f,
      0.3618946671485901f, -0.40949052572250366f, -0.18248388171195984f, -0.3349453806877136f,
      -0.34091079235076904f, 0.006497859954833984f, 0.4537564516067505f, 0.08006560802459717f,
      -0.14788749814033508f, 0.034442365169525146f, -0.33322954177856445f, 0.06049239635467529f,
      0.42619407176971436f};
  vector<int64_t> X_shape = {1, 7, 7, 1};
  vector<float> W = {-0.4406261742115021f};
  vector<int64_t> W_shape = {1, 1, 1, 1};
  vector<int64_t> Y_shape = {1, 7, 7, 1};
  auto expected_vals = {
      -0.19936637580394745f, -0.06828942894935608f, -0.04934731498360634f, 0.17369966208934784f,
      -0.11574628204107285f, -0.05910799279808998f, 0.1197819635272026f, 0.18959586322307587f,
      0.1182001456618309f, -0.17154212296009064f, 0.06006614491343498f, 0.0042258151806890965f,
      0.21490024030208588f, 0.11128675937652588f, 0.12394362688064575f, -0.17815405130386353f,
      -0.034346915781497955f, -0.14407673478126526f, -0.05778544768691063f, 0.19459928572177887f,
      -0.05484473705291748f, -0.16188594698905945f, -0.07485868036746979f, -0.08851054310798645f,
      -0.10284193605184555f, -0.17014220356941223f, -0.04898572340607643f, -0.17083507776260376f,
      -0.09170642495155334f, 0.1511256992816925f, 0.012886842712759972f, 0.09025576710700989f,
      0.08479554951190948f, 0.0489313043653965f, 0.14465972781181335f, 0.007934254594147205f,
      -0.15946026146411896f, 0.1804322451353073f, 0.08040717244148254f, 0.1475857049226761f,
      0.15021422505378723f, -0.0028631272725760937f, -0.19993697106838226f, -0.03527900204062462f,
      0.06516310572624207f, -0.015176207758486271f, 0.14682966470718384f, -0.02665453404188156f,
      -0.18779225647449493f};
  RunNhwcConv(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(NhwcConvTest, Conv2D_Bias_1) {
  NhwcConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{2, 2},        // kernel_shape
      vector<int64_t>{0, 0, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      {}                            // excluded EPs
  };

  vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  vector<int64_t> X_shape = {1, 3, 3, 1};
  vector<float> W = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
  vector<int64_t> W_shape = {2, 2, 2, 1};
  vector<int64_t> Y_shape = {1, 2, 2, 2};
  vector<float> B = {1.0f, -1.0f};
  vector<int64_t> B_shape = {2};
  auto expected_vals = {13.0f, 11.0f, 17.0f, 15.0f, 25.0f, 23.0f, 29.0f, 27.0f};

  RunNhwcConv(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
}

TEST(NhwcConvTest, Conv2D_AutoPad1) {
  NhwcConvOpAndTestAttributes attrs = {
      "SAME_UPPER",           // auto_pad
      vector<int64_t>{1, 1},  // dilations
      1,                      // group
      vector<int64_t>{3, 3},  // kernel_shape
      {},                     // pads
      vector<int64_t>{1, 1},  // strides
      {}                      // excluded EPs
  };

  vector<float> X = vector<float>(25, 1.0f);
  vector<int64_t> X_shape = {1, 5, 5, 1};
  vector<float> W = {0.0f, 1.0f, 2.0f,
                     3.0f, 4.0f, 5.0f,
                     6.0f, 7.0f, 8.0f};

  vector<int64_t> W_shape = {1, 3, 3, 1};
  vector<int64_t> Y_shape = {1, 5, 5, 1};
  auto expected_vals = {24.0f, 33.0f, 33.0f, 33.0f, 20.0f,
                        27.0f, 36.0f, 36.0f, 36.0f, 21.0f,
                        27.0f, 36.0f, 36.0f, 36.0f, 21.0f,
                        27.0f, 36.0f, 36.0f, 36.0f, 21.0f,
                        12.0f, 15.0f, 15.0f, 15.0f, 8.0f};
  RunNhwcConv(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(NhwcConvTest, Conv2D_AutoPad2) {
  NhwcConvOpAndTestAttributes attrs = {
      "SAME_LOWER",           // auto_pad
      vector<int64_t>{1, 1},  // dilations
      1,                      // group
      vector<int64_t>{3, 3},  // kernel_shape
      {},                     // pads
      vector<int64_t>{1, 1},  // strides
      {}                      // excluded EPs
  };

  vector<float> X = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                     1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                     1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                     1.0f, 0.0f, 1.0f, 0.0f, 1.0f,
                     1.0f, 0.0f, 1.0f, 0.0f, 1.0f};
  vector<int64_t> X_shape = {1, 5, 5, 1};
  vector<float> W = {0.0f, 1.0f, 2.0f,
                     3.0f, 4.0f, 5.0f,
                     6.0f, 7.0f, 8.0f};

  vector<int64_t> W_shape = {1, 3, 3, 1};
  vector<int64_t> Y_shape = {1, 5, 5, 1};
  auto expected_vals = {11.0f, 22.0f, 11.0f, 22.0f, 11.0f,
                        12.0f, 24.0f, 12.0f, 24.0f, 12.0f,
                        12.0f, 24.0f, 12.0f, 24.0f, 12.0f,
                        12.0f, 24.0f, 12.0f, 24.0f, 12.0f,
                        5.0f, 10.0f, 5.0f, 10.0f, 5.0f};
  RunNhwcConv(attrs, {X, W}, {X_shape, W_shape}, expected_vals, Y_shape);
}

TEST(NhwcConvTest, Conv2D_asymmetric_padding1) {
  NhwcConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{3, 3},        // kernel_shape
      vector<int64_t>{1, 1, 0, 0},  // pads
      vector<int64_t>{1, 1},        // strides
      {}                            // excluded EPs
  };

  vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  vector<int64_t> X_shape = {1, 3, 3, 1};
  vector<float> W = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  vector<int64_t> W_shape = {1, 3, 3, 1};
  vector<float> B = {1.f};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {1, 2, 2, 1};
  auto expected_vals = {13.f, 22.f, 28.f, 46.f};

  RunNhwcConv(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
}

TEST(NhwcConvTest, Conv2D_asymmetric_padding2) {
  NhwcConvOpAndTestAttributes attrs = {
      "",                           // auto_pad
      vector<int64_t>{1, 1},        // dilations
      1,                            // group
      vector<int64_t>{3, 3},        // kernel_shape
      vector<int64_t>{0, 0, 1, 1},  // pads
      vector<int64_t>{1, 1},        // strides
      {}                            // excluded EPs
  };

  vector<float> X = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f};
  vector<int64_t> X_shape = {1, 3, 3, 1};
  vector<float> W = {1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f};
  vector<int64_t> W_shape = {1, 3, 3, 1};
  vector<float> B = {1.f};
  vector<int64_t> B_shape = {1};
  vector<int64_t> Y_shape = {1, 2, 2, 1};
  auto expected_vals = {46.f, 34.f, 40.f, 29.f};

  RunNhwcConv(attrs, {X, W, B}, {X_shape, W_shape, B_shape}, expected_vals, Y_shape);
}

}  // namespace test
}  // namespace onnxruntime
