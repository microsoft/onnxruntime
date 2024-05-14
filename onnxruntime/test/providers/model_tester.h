// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/path_string.h"
#include "core/graph/model.h"
#include "core/session/environment.h"

#include "test/providers/base_tester.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime {
namespace test {
// To use ModelTester:
//  1. Create with the path to an existing model.
//  2. Call AddInput for all the inputs
//  3. Call AddOutput with all expected outputs
//  4. Call Run
class ModelTester : public BaseTester {
 public:
  /// <summary>
  /// Create a model tester. Intended usage is a simple model that is primarily testing a specific operator, but may
  /// require additional nodes to exercise the intended code path.
  /// </summary>
  /// <param name="test_name">Name of test to use in logs and error messages.</param>
  /// <param name="model_uri">Model to load</param>
  /// <param name="onnx_opset_version">ONNX opset version for the model.
  /// Only required if testing a model with an unreleased opset version.</param>
  explicit ModelTester(std::string_view test_name, const PathString& model_uri, int onnx_opset_version = -1)
      : BaseTester{test_name, onnx_opset_version, onnxruntime::kOnnxDomain},
        model_uri_{model_uri} {
  }

  using ExpectResult = BaseTester::ExpectResult;

 private:
  void CreateModelToTest(const ModelOptions& model_options, Model*& model) override {
    ASSERT_STATUS_OK(Model::Load(model_uri_, model_, nullptr, DefaultLoggingManager().DefaultLogger(), model_options));

    model = model_.get();
  }

  const PathString model_uri_;
  std::shared_ptr<Model> model_;
};
}  // namespace test
}  // namespace onnxruntime
