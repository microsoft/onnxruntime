// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <random>

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {

struct TensorInfo {
  TensorInfo(VectorInt64 shapes, std::vector<float> values) {
    shapes_ = shapes;
    fp32_values_ = values;

    size_t total_size = 1;
    for (size_t i = 0; i < shapes_.size(); ++i) {
      total_size *= shapes_[i];
    }

    ORT_ENFORCE(fp32_values_.size() == total_size,
                "Number of elements mismtach betwen shapes and values");
  }

  template <typename OutT>
  std::vector<OutT> Values() const {
    if (false && std::is_same<OutT, MLFloat16>::value) {
      std::vector<OutT> fp16_values;
      fp16_values.reserve(fp32_values_.size());
      ConvertFloatToMLFloat16(fp32_values_.data(), reinterpret_cast<onnxruntime::MLFloat16*>(fp16_values.data()), fp32_values_.size());
      return fp16_values;
    } else if (std::is_same<OutT, float>::value) {
      return fp32_values_;
    } else {
      ORT_THROW("Not supported data type.");
    }
  }

  VectorInt64 Shapes() const {
    return shapes_;
  }

  VectorInt64 shapes_;
  std::vector<float> fp32_values_;
};

template <typename T>
struct AdamTestInputOutput {
  AdamTestInputOutput(
      float lr,
      int64_t step,
      const std::vector<TensorInfo>& weight_tensor_infos,
      const std::vector<TensorInfo>& gradient_tensor_infos,
      const std::vector<TensorInfo>& momentum_1_tensor_infos,
      const std::vector<TensorInfo>& momentum_2_tensor_infos,
      const std::vector<TensorInfo>& updated_weight_tensor_infos,
      const std::vector<TensorInfo>& updated_momentum_1_tensor_infos,
      const std::vector<TensorInfo>& updated_momentum_2_tensor_infos) {
    lr_vector.push_back(lr);
    step_vector.push_back(step);

    // Input Sequence tensors.

    for (const TensorInfo& ti : weight_tensor_infos) {
      weight_seq_tensors_.AddTensor(ti.Shapes(), ti.Values<T>());
    }

    for (const TensorInfo& ti : gradient_tensor_infos) {
      gradient_seq_tensors_.AddTensor(ti.Shapes(), ti.Values<T>());
    }

    for (const TensorInfo& ti : momentum_1_tensor_infos) {
      momentum_1_seq_tensors_.AddTensor(ti.Shapes(), ti.Values<T>());
    }

    for (const TensorInfo& ti : momentum_2_tensor_infos) {
      momentum_2_seq_tensors_.AddTensor(ti.Shapes(), ti.Values<T>());
    }

    // Update sequence tensors.

    for (const TensorInfo& ti : updated_weight_tensor_infos) {
      updated_weight_seq_tensors_.AddTensor(ti.Shapes(), ti.Values<T>());
    }

    for (const TensorInfo& ti : updated_momentum_1_tensor_infos) {
      updated_momentum_1_seq_tensors_.AddTensor(ti.Shapes(), ti.Values<T>());
    }

    for (const TensorInfo& ti : updated_momentum_2_tensor_infos) {
      updated_momentum_2_seq_tensors_.AddTensor(ti.Shapes(), ti.Values<T>());
    }
  }

  SeqTensors<T>& WeightSeq() {
    return weight_seq_tensors_;
  }

  SeqTensors<T>& GradientSeq() {
    return gradient_seq_tensors_;
  }

  SeqTensors<T>& Momentum_1_Seq() {
    return momentum_1_seq_tensors_;
  }

  SeqTensors<T>& Momentum_2_Seq() {
    return momentum_2_seq_tensors_;
  }

  SeqTensors<T>& UpdatedWeightSeq() {
    return updated_weight_seq_tensors_;
  }

  SeqTensors<T>& UpdatedMomentum_1_Seq() {
    return updated_momentum_1_seq_tensors_;
  }

  SeqTensors<T>& UpdatedMomentum_2_Seq() {
    return updated_momentum_2_seq_tensors_;
  }

  std::vector<float> lr_vector;
  std::vector<int64_t> step_vector;

 private:
  SeqTensors<T> weight_seq_tensors_;
  SeqTensors<T> gradient_seq_tensors_;
  SeqTensors<T> momentum_1_seq_tensors_;
  SeqTensors<T> momentum_2_seq_tensors_;

  SeqTensors<T> updated_weight_seq_tensors_;
  SeqTensors<T> updated_momentum_1_seq_tensors_;
  SeqTensors<T> updated_momentum_2_seq_tensors_;
};

TEST(AdamTest, TorchAdamSingleWeightTest) {
  OpTester test("Adam", 1, onnxruntime::kMSDomain);
  float lr = 1e-3f;
  int64_t step = 3;
  std::vector<TensorInfo> weight_tensor_infos{TensorInfo({3}, {1.0f, 2.0f, 3.0f})};
  std::vector<TensorInfo> gradient_tensor_infos{TensorInfo({3}, {4.0f, 5.0f, 6.0f})};
  std::vector<TensorInfo> momentum_1_tensor_infos{TensorInfo({3}, {0.1f, 0.2f, 0.3f})};
  std::vector<TensorInfo> momentum_2_tensor_infos{TensorInfo({3}, {0.4f, 0.5f, 0.6f})};

  std::vector<TensorInfo> updated_weight_tensor_infos{TensorInfo({3}, {0.6199609f, 1.5305318f, 2.4542853f})};
  std::vector<TensorInfo> updated_momentum_1_tensor_infos{TensorInfo({3}, {0.49f, 0.68f, 0.87f})};
  std::vector<TensorInfo> updated_momentum_2_tensor_infos{TensorInfo({3}, {0.4156f, 0.5245f, 0.6354f})};
  AdamTestInputOutput<float> data(lr, step,
                                  weight_tensor_infos, gradient_tensor_infos, momentum_1_tensor_infos,
                                  momentum_2_tensor_infos, updated_weight_tensor_infos, updated_momentum_1_tensor_infos,
                                  updated_momentum_2_tensor_infos);

  test.AddSeqInput("weights", data.WeightSeq());
  test.AddSeqInput("gradients", data.GradientSeq());
  test.AddSeqInput("momentums_1", data.Momentum_1_Seq());
  test.AddSeqInput("momentums_2", data.Momentum_2_Seq());
  test.AddInput<float>("lr", {1}, data.lr_vector);
  test.AddInput<int64_t>("step", {1}, data.step_vector);

  // Verify AdamOptimizer outputs
  test.AddSeqOutput("updated_weights", data.UpdatedWeightSeq());
  test.AddSeqOutput("updated_momentums_1", data.UpdatedMomentum_1_Seq());
  test.AddSeqOutput("updated_momentums_2", data.UpdatedMomentum_2_Seq());
  test.AddOutput<int64_t>("updated_flag", {1}, {1});

  test.AddAttribute("alpha", static_cast<float>(0.9f));
  test.AddAttribute("beta", static_cast<float>(0.999f));
  test.AddAttribute("epsilon", static_cast<float>(1e-8f));
  test.AddAttribute("weight_decay", static_cast<float>(1e-2f));
  test.AddAttribute("adam_mode", static_cast<int64_t>(0));
  test.AddAttribute("correct_bias", static_cast<int64_t>(1));

  test.Run();
}

// TEST(AdamTest, AdamOptimizerTest_Gradient) {
//   OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
//   AdamTestInputOutput data;

//   test.AddInput<float>("ETA", {}, data.eta);
//   test.AddInput<int64_t>("Update_Count", {}, {3});
//   test.AddInput<float>("W", {3}, data.w);
//   test.AddInput<float>("G", {3}, data.g);
//   test.AddInput<float>("Moment_1", {3}, data.m1);
//   test.AddInput<float>("Moment_2", {3}, data.m2);

//   // Verify AdamOptimizer outputs
//   test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
//   test.AddOutput<float>("Moment_1_Out", {3}, data.m1_new);
//   test.AddOutput<float>("Moment_2_Out", {3}, data.m2_new);
//   test.AddOptionalOutputEdge<float>();
//   test.AddOutput<float>("G_Out", {3}, data.g_new);

//   test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
//   test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));

//   test.Run();
// }

// TEST(AdamTest, AdamBiasCorrection) {
//   OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
//   AdamTestInputOutput data;

//   test.AddInput<float>("ETA", {}, {1.f});
//   test.AddInput<int64_t>("Update_Count", {}, {1});
//   test.AddInput<float>("W", {3}, {-0.4634f, 0.3584f, -0.2121f});
//   test.AddInput<float>("G", {3}, {0.4171f, 0.9485f, 1.2289f});
//   test.AddInput<float>("Moment_1", {3}, {0.f, 0.f, 0.f});
//   test.AddInput<float>("Moment_2", {3}, {0.f, 0.f, 0.f});

//   test.AddOutput<int64_t>("Update_Count_Out", {}, {2});
//   test.AddOutput<float>("Moment_1_Out", {3}, {0.0417f, 0.0949f, 0.1229f});
//   test.AddOutput<float>("Moment_2_Out", {3}, {1.7400e-04f, 8.9966e-04f, 1.5102e-03f});
//   test.AddOutput<float>("W_Out", {3}, {-1.4634f, -0.6416f, -1.2121f});

//   test.AddAttribute("do_bias_correction", static_cast<int64_t>(1));
//   test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));

//   test.Run();
// }

// TEST(AdamTest, AdamWeightDecayMode0NoBiasCorrection) {
//   OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
//   AdamTestInputOutput data;

//   test.AddInput<float>("ETA", {}, {1.f});
//   test.AddInput<int64_t>("Update_Count", {}, {1});
//   test.AddInput<float>("W", {3}, {-0.4634f, 0.3584f, -0.2121f});
//   test.AddInput<float>("G", {3}, {0.4171f, 0.9485f, 1.2289f});
//   test.AddInput<float>("Moment_1", {3}, {0.f, 0.f, 0.f});
//   test.AddInput<float>("Moment_2", {3}, {0.f, 0.f, 0.f});

//   test.AddOutput<int64_t>("Update_Count_Out", {}, {2});
//   test.AddOutput<float>("Moment_1_Out", {3}, {0.0417f, 0.0949f, 0.1229f});
//   test.AddOutput<float>("Moment_2_Out", {3}, {1.7400e-04f, 8.9966e-04f, 1.5102e-03f});
//   test.AddOutput<float>("W_Out", {3}, {-3.6210f, -2.8075f, -3.3723f});
//   test.AddOutput<float>("G_Out", {3}, {-3.1576f, -3.1658f, -3.1601f});

//   test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
//   test.AddAttribute("lambda", 0.01f);
//   test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));

//   test.Run();
// }

// TEST(AdamTest, AdamWeightDecayMode0WithBiasCorrection) {
//   OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
//   AdamTestInputOutput data;

//   test.AddInput<float>("ETA", {}, {1.f});
//   test.AddInput<int64_t>("Update_Count", {}, {1});
//   test.AddInput<float>("W", {3}, {-0.4634f, 0.3584f, -0.2121f});
//   test.AddInput<float>("G", {3}, {0.4171f, 0.9485f, 1.2289f});
//   test.AddInput<float>("Moment_1", {3}, {0.f, 0.f, 0.f});
//   test.AddInput<float>("Moment_2", {3}, {0.f, 0.f, 0.f});

//   test.AddOutput<int64_t>("Update_Count_Out", {}, {2});
//   test.AddOutput<float>("Moment_1_Out", {3}, {0.0417f, 0.0949f, 0.1229f});
//   test.AddOutput<float>("Moment_2_Out", {3}, {1.7400e-04f, 8.9966e-04f, 1.5102e-03f});
//   test.AddOutput<float>("W_Out", {3}, {-1.4587f, -0.6452f, -1.2099f});
//   test.AddOutput<float>("G_Out", {3}, {-0.9954f, -1.0036f, -0.9979f});

//   test.AddAttribute("do_bias_correction", static_cast<int64_t>(1));
//   test.AddAttribute("lambda", 0.01f);
//   test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));

//   test.Run();
// }

// TEST(AdamTest, AdamWeightDecayMode1NoBiasCorrection) {
//   OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
//   AdamTestInputOutput data;

//   test.AddInput<float>("ETA", {}, {1.f});
//   test.AddInput<int64_t>("Update_Count", {}, {1});
//   test.AddInput<float>("W", {3}, {-0.4634f, 0.3584f, -0.2121f});
//   test.AddInput<float>("G", {3}, {0.4171f, 0.9485f, 1.2289f});
//   test.AddInput<float>("Moment_1", {3}, {0.f, 0.f, 0.f});
//   test.AddInput<float>("Moment_2", {3}, {0.f, 0.f, 0.f});

//   test.AddOutput<int64_t>("Update_Count_Out", {}, {2});
//   test.AddOutput<float>("Moment_1_Out", {3}, {0.0417f, 0.0949f, 0.1229f});
//   test.AddOutput<float>("Moment_2_Out", {3}, {1.7400e-04f, 8.9966e-04f, 1.5102e-03f});
//   test.AddOutput<float>("W_Out", {3}, {-3.5894f, -2.7758f, -3.3406f});

//   test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
//   test.AddAttribute("lambda", 0.01f);
//   test.AddAttribute("weight_decay_mode", static_cast<int64_t>(1));

//   test.Run();
// }

// TEST(AdamTest, AdamWeightDecayMode1WithBiasCorrection) {
//   OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
//   AdamTestInputOutput data;

//   test.AddInput<float>("ETA", {}, {1.f});
//   test.AddInput<int64_t>("Update_Count", {}, {1});
//   test.AddInput<float>("W", {3}, {-0.4634f, 0.3584f, -0.2121f});
//   test.AddInput<float>("G", {3}, {0.4171f, 0.9485f, 1.2289f});
//   test.AddInput<float>("Moment_1", {3}, {0.f, 0.f, 0.f});
//   test.AddInput<float>("Moment_2", {3}, {0.f, 0.f, 0.f});

//   test.AddOutput<int64_t>("Update_Count_Out", {}, {2});
//   test.AddOutput<float>("Moment_1_Out", {3}, {0.0417f, 0.0949f, 0.1229f});
//   test.AddOutput<float>("Moment_2_Out", {3}, {1.7400e-04f, 8.9966e-04f, 1.5102e-03f});
//   test.AddOutput<float>("W_Out", {3}, {-1.4488f, -0.6352f, -1.1999f});

//   test.AddAttribute("do_bias_correction", static_cast<int64_t>(1));
//   test.AddAttribute("lambda", 0.01f);
//   test.AddAttribute("weight_decay_mode", static_cast<int64_t>(1));

//   test.Run();
// }

// #if defined(USE_CUDA) || defined(USE_ROCM)

// float GetGradientL2Norm(const std::vector<float>& gradient_vector) {
//   float gradient_norm = 0.0f;
//   for (const auto g_value : gradient_vector) {
//     gradient_norm += g_value * g_value;
//   }
//   gradient_norm = std::sqrt(gradient_norm);
//   return gradient_norm;
// }

// TEST(AdamTest, AdamOptimizerMixPrecisionTest) {
//   OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
//   AdamTestInputOutput data;

//   test.AddInput<MLFloat16>("ETA", {}, data.eta_half);
//   test.AddInput<int64_t>("Update_Count", {}, {3});
//   test.AddInput<float>("W", {3}, data.w);
//   test.AddInput<MLFloat16>("G", {3}, data.g_half);
//   test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
//   test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);

//   // Verify AdamOptimizer outputs
//   test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
//   test.AddOutput<MLFloat16>("Moment_1_Out", {3}, data.m1_new_half);
//   test.AddOutput<MLFloat16>("Moment_2_Out", {3}, data.m2_new_half);
//   test.AddOutput<float>("W_Out", {3}, data.w_new);

//   test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
//   test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));

//   test.Run();
// }

// TEST(AdamTest, AdamOptimizerMixPrecision_FP16Weight_Test) {
//   OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
//   AdamTestInputOutput data;

//   test.AddInput<MLFloat16>("ETA", {}, data.eta_half);
//   test.AddInput<int64_t>("Update_Count", {}, {3});
//   test.AddInput<float>("W", {3}, data.w);
//   test.AddInput<MLFloat16>("G", {3}, data.g_half);
//   test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
//   test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);
//   test.AddInput<MLFloat16>("FP16_W", {3}, data.w_half);

//   // Verify AdamOptimizer outputs
//   test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
//   test.AddOutput<MLFloat16>("Moment_1_Out", {3}, data.m1_new_half);
//   test.AddOutput<MLFloat16>("Moment_2_Out", {3}, data.m2_new_half);
//   test.AddOutput<float>("W_Out", {3}, data.w_new);
//   test.AddOptionalOutputEdge<MLFloat16>();
//   test.AddOutput<MLFloat16>("FP16_W_Out", {3}, data.w_new_half);

//   test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
//   test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));

//   test.Run();
// }

// TEST(AdamTest, AdamOptimizerMixPrecision_FP16Weight_NoClipNorm_Test) {
//   OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
//   AdamTestInputOutput data;

//   test.AddInput<MLFloat16>("ETA", {}, data.eta_half);
//   test.AddInput<int64_t>("Update_Count", {}, {3});
//   test.AddInput<float>("W", {3}, data.w);
//   test.AddInput<MLFloat16>("G", {3}, data.g_half);
//   test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
//   test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);
//   test.AddInput<MLFloat16>("FP16_W", {3}, data.w_half);
//   test.AddInput<float>("loss_scale", {1}, {1.0f});
//   // grad clipping should not take effect because default max_norm is 1.0f
//   test.AddInput<float>("grad_norm", {1}, {0.01f});
//   // Verify AdamOptimizer outputs
//   test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
//   test.AddOutput<MLFloat16>("Moment_1_Out", {3}, data.m1_new_half);
//   test.AddOutput<MLFloat16>("Moment_2_Out", {3}, data.m2_new_half);
//   test.AddOutput<float>("W_Out", {3}, data.w_new);
//   test.AddOptionalOutputEdge<MLFloat16>();
//   test.AddOutput<MLFloat16>("FP16_W_Out", {3}, data.w_new_half);

//   test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
//   test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));
//   test.Run();
// }

// TEST(AdamTest, AdamOptimizerMixPrecision_FP16Weight_ClipNorm_Test) {
//   OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
//   AdamTestInputOutput data;

//   // Expected FP32 Outputs
//   std::vector<float> m1_new = {0.13f, 0.23f, 0.33f};
//   std::vector<float> m2_new = {0.3997f, 0.4998f, 0.6001f};
//   std::vector<float> w_new = {0.8972168f, 1.8369141f, 2.7871094f};
//   // FP16 Outputs
//   std::vector<MLFloat16> m1_new_half;
//   std::vector<MLFloat16> m2_new_half;
//   std::vector<MLFloat16> w_new_half;

//   m1_new_half.resize(m1_new.size());
//   m2_new_half.resize(m2_new.size());
//   w_new_half.resize(w_new.size());
//   ConvertFloatToMLFloat16(m1_new.data(), m1_new_half.data(), int(m1_new.size()));
//   ConvertFloatToMLFloat16(m2_new.data(), m2_new_half.data(), int(m2_new.size()));
//   ConvertFloatToMLFloat16(w_new.data(), w_new_half.data(), int(w_new.size()));

//   test.AddInput<MLFloat16>("ETA", {}, data.eta_half);
//   test.AddInput<int64_t>("Update_Count", {}, {3});
//   test.AddInput<float>("W", {3}, data.w);
//   test.AddInput<MLFloat16>("G", {3}, data.g_half);
//   test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
//   test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);
//   test.AddInput<MLFloat16>("FP16_W", {3}, data.w_half);
//   test.AddInput<float>("loss_scale", {1}, {1.0f});
//   test.AddInput<float>("grad_norm", {1}, {0.01f});
//   // Verify AdamOptimizer outputs
//   test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
//   test.AddOutput<MLFloat16>("Moment_1_Out", {3}, m1_new_half);
//   test.AddOutput<MLFloat16>("Moment_2_Out", {3}, m2_new_half);
//   test.AddOutput<float>("W_Out", {3}, w_new);
//   test.AddOptionalOutputEdge<MLFloat16>();
//   test.AddOutput<MLFloat16>("FP16_W_Out", {3}, w_new_half);

//   test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
//   test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));
//   test.AddAttribute("max_norm_clip", 0.001f);
//   test.Run();
// }

// TEST(AdamTest, AdamOptimizerMixPrecision_FP16Weight_SkipUpdate_Test) {
//   OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
//   AdamTestInputOutput data;

//   test.AddInput<MLFloat16>("ETA", {}, data.eta_half);
//   test.AddInput<int64_t>("Update_Count", {}, {3});
//   test.AddInput<float>("W", {3}, data.w);
//   test.AddInput<MLFloat16>("G", {3}, data.g_half);
//   test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
//   test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);
//   test.AddInput<MLFloat16>("FP16_W", {3}, data.w_half);
//   test.AddInput<float>("loss_scale", {1}, {1.0f});
//   // grad clipping should not take effect
//   test.AddInput<float>("grad_norm", {1}, {0.01f});
//   test.AddInput<bool>("DoUpdate", {1}, {false});

//   // Verify AdamOptimizer outputs
//   test.AddOutput<int64_t>("Update_Count_Out", {}, {3});
//   test.AddOutput<MLFloat16>("Moment_1_Out", {3}, data.m1_half);
//   test.AddOutput<MLFloat16>("Moment_2_Out", {3}, data.m2_half);
//   test.AddOutput<float>("W_Out", {3}, data.w);
//   test.AddOptionalOutputEdge<MLFloat16>();
//   test.AddOutput<MLFloat16>("FP16_W_Out", {3}, data.w_half);

//   test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
//   test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));

//   test.Run();
// }

// TEST(AdamTest, AdamOptimizerMixPrecisionTestFloatEta) {
//   OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
//   AdamTestInputOutput data;

//   test.AddInput<float>("ETA", {}, data.eta);
//   test.AddInput<int64_t>("Update_Count", {}, {3});
//   test.AddInput<float>("W", {3}, data.w);
//   test.AddInput<MLFloat16>("G", {3}, data.g_half);
//   test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
//   test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);

//   // Verify AdamOptimizer outputs
//   test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
//   test.AddOutput<MLFloat16>("Moment_1_Out", {3}, data.m1_new_half);
//   test.AddOutput<MLFloat16>("Moment_2_Out", {3}, data.m2_new_half);
//   test.AddOutput<float>("W_Out", {3}, data.w_new);

//   test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));
//   test.AddAttribute("weight_decay_mode", static_cast<int64_t>(0));

//   test.Run();
// }

// TEST(AdamTest, AdamOptimizerMixPrecisionTest_Gradient) {
//   OpTester test("AdamOptimizer", 1, onnxruntime::kMSDomain);
//   AdamTestInputOutput data;

//   test.AddInput<float>("ETA", {}, data.eta);
//   test.AddInput<int64_t>("Update_Count", {}, {3});
//   test.AddInput<float>("W", {3}, data.w);
//   test.AddInput<MLFloat16>("G", {3}, data.g_half);
//   test.AddInput<MLFloat16>("Moment_1", {3}, data.m1_half);
//   test.AddInput<MLFloat16>("Moment_2", {3}, data.m2_half);

//   // Verify AdamOptimizer outputs
//   test.AddOutput<int64_t>("Update_Count_Out", {}, {4});
//   test.AddOutput<MLFloat16>("Moment_1_Out", {3}, data.m1_new_half);
//   test.AddOutput<MLFloat16>("Moment_2_Out", {3}, data.m2_new_half);
//   test.AddOptionalOutputEdge<float>();
//   test.AddOutput<MLFloat16>("G_Out", {3}, data.g_new_half);

//   test.AddAttribute("do_bias_correction", static_cast<int64_t>(0));

//   test.Run();
// }

// #endif
}  // namespace
}  // namespace test
}  // namespace onnxruntime
