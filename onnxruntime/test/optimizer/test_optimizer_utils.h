// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <variant>

#include "core/common/common.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/float16.h"
#include "core/framework/framework_common.h"
#include "core/framework/ort_value.h"
#include "test/framework/test_utils.h"

namespace onnxruntime {
namespace test {

/**
 * @brief Class represent a input data (dimensions, data type and value).
 */
struct TestInputData {
  template <typename T>
  TestInputData(const std::string& name, const TensorShapeVector& dims, const std::vector<T>& values)
      : name_(name), dims_(dims), values_(values) {}

  OrtValue ToOrtValue() {
    OrtValue ortvalue;
    std::vector<int64_t> dims;
    dims.reserve(dims_.size());
    dims.insert(dims.end(), dims_.begin(), dims_.end());
    std::visit([&ortvalue, &dims](auto&& arg) {
      using T = std::decay_t<decltype(arg)>;
      if constexpr (std::is_same_v<T, std::vector<int64_t>> ||
                    std::is_same_v<T, std::vector<float>> ||
                    std::is_same_v<T, std::vector<MLFloat16>>)
        CreateMLValue<typename T::value_type>(
            TestCPUExecutionProvider()->CreatePreferredAllocators()[0], dims, arg, &ortvalue);
      else
        static_assert("Unspported types!");
    },
               values_);

    return ortvalue;
  }

  std::string GetName() const {
    return name_;
  }

 private:
  std::string name_;
  TensorShapeVector dims_;
  std::variant<std::vector<float>, std::vector<MLFloat16>, std::vector<int64_t>> values_;
};

/**
 * @brief A container for all input data.
 *
 */
struct InputContainer {
  InputContainer() = default;

  template <typename T>
  TestInputData& AddInput(const std::string& name, const TensorShapeVector dims, const std::vector<T>& values) {
    inputs_.emplace_back(TestInputData(name, dims, values));
    return inputs_.back();
  }

  template <typename T>
  TestInputData& AddInput(const std::string& name, TensorShapeVector dims,
                          std::function<
                              void(const TensorShapeVector& shape, std::vector<T>& data)>
                              func = nullptr) {
    std::vector<T> values(TensorShape(dims).Size());
    if (func) {
      func(dims, values);
    }

    inputs_.emplace_back(TestInputData(name, dims, values));
    return inputs_.back();
  }

  void ToInputMap(NameMLValMap& feeds) const {
    for (auto input : inputs_) {
      feeds.insert({input.GetName(), input.ToOrtValue()});
    }
  }

 private:
  std::vector<TestInputData> inputs_;
};

void RandomFillFloatVector(const TensorShapeVector& shape, std::vector<float>& data);

void RandomFillHalfVector(const TensorShapeVector& shape, std::vector<MLFloat16>& data);

void RandomMasks(int64_t batch, int64_t sequence_length, std::vector<int64_t>& data);

void RunModelWithData(const PathString& model_uri, const std::string session_log_id,
                      const std::string& provider_type, const InputContainer& input_container,
                      const std::vector<std::string>& output_names,
                      std::vector<OrtValue>& run_results);
}  // namespace test
}  // namespace onnxruntime
