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
  int64_t step = 0;
  std::vector<TensorInfo> weight_tensor_infos{
      TensorInfo({2, 3}, {-0.1833, 0.6740, 0.3117, 0.4283, -0.3958, 0.0742})};
  std::vector<TensorInfo> gradient_tensor_infos{
      TensorInfo({2, 3}, {-0.1866, 1.0502, -0.0654, 0.7892, -0.0699, 0.0831})};
  std::vector<TensorInfo> momentum_1_tensor_infos{
      TensorInfo({2, 3}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f})};
  std::vector<TensorInfo> momentum_2_tensor_infos{
      TensorInfo({2, 3}, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f})};

  std::vector<TensorInfo> updated_weight_tensor_infos{
      TensorInfo({2, 3}, {-0.1823, 0.6729, 0.3127, 0.4273, -0.3948, 0.0732})};
  std::vector<TensorInfo> updated_momentum_1_tensor_infos{
      TensorInfo({2, 3}, {-0.0187, 0.1050, -0.0065, 0.0789, -0.0070, 0.0083})};
  std::vector<TensorInfo> updated_momentum_2_tensor_infos{
      TensorInfo({2, 3}, {3.4822e-05, 1.1029e-03, 4.2755e-06, 6.2290e-04, 4.8859e-06, 6.9078e-06})};
  AdamTestInputOutput<float> data(lr, step,
                                  weight_tensor_infos, gradient_tensor_infos,
                                  momentum_1_tensor_infos, momentum_2_tensor_infos,
                                  updated_weight_tensor_infos, updated_momentum_1_tensor_infos,
                                  updated_momentum_2_tensor_infos);

  // Default values for Torch AdamW.
  test.AddAttribute("alpha", static_cast<float>(0.9f));
  test.AddAttribute("beta", static_cast<float>(0.999f));
  test.AddAttribute("epsilon", static_cast<float>(1e-8f));
  test.AddAttribute("weight_decay", static_cast<float>(1e-2f));
  test.AddAttribute("adam_mode", static_cast<int64_t>(0));
  test.AddAttribute("correct_bias", static_cast<int64_t>(1));

  test.AddSeqInput("weights", data.WeightSeq());
  test.AddSeqInput("gradients", data.GradientSeq());
  test.AddSeqInput("momentums_1", data.Momentum_1_Seq());
  test.AddSeqInput("momentums_2", data.Momentum_2_Seq());
  test.AddInput<float>("lr", {1}, data.lr_vector);
  test.AddInput<int64_t>("step", {1}, data.step_vector);

  // Verify AdamOptimizer outputs
  float rtol = 1e-5f;
  float atol = 1e-4f;
  test.AddOutput<int64_t>("updated_flag", {1}, {1});
  test.AddOptionalTypeSeqOutput("updated_weights", &data.UpdatedWeightSeq(), rtol, atol);
  test.AddOptionalTypeSeqOutput("updated_momentums_1", &data.UpdatedMomentum_1_Seq(), rtol, atol);
  test.AddOptionalTypeSeqOutput("updated_momentums_2", &data.UpdatedMomentum_2_Seq(), rtol, atol);

  test.Run();
}

// TODO: adding more test cases.

}  // namespace
}  // namespace test
}  // namespace onnxruntime
