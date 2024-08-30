/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/* Modifications Copyright (c) Microsoft. */

#pragma once
#include "test/providers/provider_test_utils.h"
#include "orttraining/core/session/training_session.h"
#include "orttraining/test/gradient/gradient_op_test_utils.h"

namespace onnxruntime {
namespace test {

// TODO: This class currently assumes the inputs share types and the outputs share a type.
// However there are cases like MaxPool and Gather where this is not true.
template <typename X_T, typename Y_T, typename JAC_T>
class GradientChecker {
 public:
  GradientChecker() = default;

  /// Returns in 'max_error' the maximum element-wise error for dy/dx between the
  /// computed and numeric Jacobian matrices where 'xs' and 'ys' are tensors.
  /// X_T and Y_T are the c++ types for the x and y tensors, and JAC_T is a
  /// real-valued type to store the Jacobian derivatives dy/dx.
  /// This function adds operations to the graph associated with 'scope'.
  ///
  /// Examples:
  /// if y = Square(x), where x (and so y) are DT_FLOAT,
  /// <X_T, Y_T, JAC_T> should be <float, float, float>
  ///
  /// if y = Square(x), where x (and so y) are DT_DOUBLE,
  /// <X_T, Y_T, JAC_T> should be <double, double, double>
  Status ComputeGradientError(const training::OpDef& op_def, const std::vector<TensorInfo>& x_infos,
                              const std::vector<TensorInfo>& y_infos, JAC_T* max_error,
                              const std::vector<ONNX_NAMESPACE::AttributeProto>& attributes = {},
                              // TODO: Ideally it shall check for not has_gradient cases. But some tests are failing
                              // because the gradient op does not handle the case. We have to use this flag
                              // to disable check for not having gradient cases in order to pass those test.
                              // Remove this flag when the gradient op is fixed.
                              bool check_not_have_gradient = true,
                              // Also check gradient builder for op for cases where input shapes are not available
                              bool check_not_have_shape_inferencing = false,
                              std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr);

  Status ComputeGradientError(const training::OpDef& op_def, const std::vector<TensorInfo>& x_infos,
                              const std::vector<TensorInfo>& y_infos, JAC_T* max_error,
                              std::vector<std::vector<X_T>> x_datas,
                              const std::vector<ONNX_NAMESPACE::AttributeProto>& attributes = {},
                              // TODO: Ideally it shall check for not has_gradient cases. But some tests are failing
                              // because the gradient op does not handle the case. We have to use this flag
                              // to disable check for not having gradient cases in order to pass those test.
                              // Remove this flag when the gradient op is fixed.
                              bool check_not_have_gradient = true,
                              // Also check gradient builder for op for cases where input shapes are not available
                              bool check_not_have_shape_inferencing = false,
                              std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr);

 private:
  void InitJacobians(size_t row_count, size_t col_count, std::vector<std::vector<JAC_T>>* jacobians);

  void AddDatas(GradientOpTester& op_tester,
                const std::vector<TensorInfo>& x_infos, const std::vector<TensorInfo>& y_infos,
                std::vector<std::vector<X_T>>* x_datas, std::vector<std::vector<Y_T>>* y_datas);

  std::vector<OrtValue> EvaluateFunctionAtInput(GradientOpTester& op_tester,
                                                const std::vector<TensorInfo>& x_infos,
                                                const std::vector<TensorInfo>& y_infos,
                                                std::vector<std::vector<X_T>>* x_datas,
                                                std::vector<std::vector<Y_T>>* y_datas,
                                                std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers);

  Status InitOpTesterWithGraph(GradientOpTester& op_tester,
                               const std::vector<TensorInfo>& x_infos, const std::vector<TensorInfo>& y_infos,
                               std::vector<std::vector<X_T>>* x_datas, std::vector<std::vector<Y_T>>* y_datas,
                               const std::vector<ONNX_NAMESPACE::AttributeProto>& attributes,
                               const std::unordered_map<std::string, int>& extra_domain_to_version = {});

  Status InitOpTesterWithGradGraph(GradientOpTester& op_tester,
                                   const std::vector<TensorInfo>& x_infos, const std::vector<TensorInfo>& y_infos,
                                   std::vector<std::vector<X_T>>* x_datas, std::vector<std::vector<Y_T>>* y_datas,
                                   const std::vector<ONNX_NAMESPACE::AttributeProto>& attributes);

  Status ComputeTheoreticalJacobianTranspose(
      const training::OpDef& op_def,
      const std::vector<TensorInfo>& x_infos, const std::vector<TensorInfo>& y_infos,
      std::vector<std::vector<X_T>>* x_datas, std::vector<std::vector<Y_T>>* y_datas,
      std::vector<std::vector<JAC_T>>* jacobian_ts, const std::vector<size_t>& row_strides,
      const std::vector<size_t>& col_strides, const std::vector<ONNX_NAMESPACE::AttributeProto>& attributes,
      bool add_shape = true, std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr);

  Status ComputeNumericJacobianTranspose(
      const training::OpDef& op_def,
      const std::vector<TensorInfo>& x_infos, const std::vector<TensorInfo>& y_infos, const JAC_T delta,
      std::vector<std::vector<X_T>>* x_datas, std::vector<std::vector<Y_T>>* y_datas,
      std::vector<std::vector<JAC_T>>* jacobian_ts, const std::vector<size_t>& row_strides,
      const std::vector<size_t>& col_strides, const std::vector<ONNX_NAMESPACE::AttributeProto>& attributes,
      bool add_shape = true, std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr);

  Status ComputeGradientErrorInternal(const training::OpDef& op_name,
                                      const std::vector<TensorInfo>& x_infos, const std::vector<TensorInfo>& y_infos,
                                      std::vector<std::vector<X_T>>* x_datas, std::vector<std::vector<Y_T>>* y_datas,
                                      JAC_T* max_error,
                                      const std::vector<ONNX_NAMESPACE::AttributeProto>& attributes,
                                      bool check_not_have_gradient = true,
                                      bool check_not_have_shape_inferencing = false,
                                      std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr);
};
}  // namespace test
}  // namespace onnxruntime
