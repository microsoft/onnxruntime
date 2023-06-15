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

#include "orttraining/test/gradient/gradient_checker.h"

#include <random>

#include "orttraining/core/framework/gradient_graph_builder.h"
#include "orttraining/core/graph/gradient_config.h"

#include "test/util/include/default_providers.h"
#include "test/util/include/test_random_seed.h"

namespace onnxruntime {
namespace test {

using ONNX_NAMESPACE::AttributeProto;
using training::OpDef;

namespace {

std::vector<std::unique_ptr<IExecutionProvider>> GetExecutionProviders(
    std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers = nullptr, bool cpu_only = false) {
  std::vector<std::unique_ptr<IExecutionProvider>> result;
  if (execution_providers) {
    for (auto& entry : *execution_providers) {
      if (entry->Type() == onnxruntime::kCpuExecutionProvider) {
        result.emplace_back(DefaultCpuExecutionProvider());
      } else if (entry->Type() == onnxruntime::kCudaExecutionProvider) {
        result.emplace_back(DefaultCudaExecutionProvider());
      } else if (entry->Type() == onnxruntime::kRocmExecutionProvider) {
        result.emplace_back(DefaultRocmExecutionProvider());
      } else if (entry->Type() == onnxruntime::kDnnlExecutionProvider) {
        result.emplace_back(DefaultDnnlExecutionProvider());
      } else if (entry->Type() == onnxruntime::kTensorrtExecutionProvider) {
        result.emplace_back(DefaultTensorrtExecutionProvider());
      }
    }
    return result;
  }

  if (cpu_only) {
    result.emplace_back(DefaultCpuExecutionProvider());
    return result;
  }
#ifdef USE_CUDA
  result.emplace_back(DefaultCudaExecutionProvider());
#endif
#ifdef USE_ROCM
  result.emplace_back(DefaultRocmExecutionProvider());
#endif
  result.emplace_back(DefaultCpuExecutionProvider());
  return result;
}

};  // namespace

// The jacobian transpose matrix is laid out as follows

// Say there are three inputs each of size M X N, N X K, K X J
// say there are two outputs each of size M X K , N X J

//    output size (y_shapes)  -->            | M X K  |N X J |
//     input size (x_shapes)        M X N    |        |      |
//       |                          N X K    |        |      |
//       |                          K X J    |        |      |
//       V
// The Jacobian is always a real-valued matrix.
// Given y = f(x) for tensors y and x, it contains the derivatives dy_i/dx_j for
// every pair y_i in y and x_j in x.  Note that the Jacobian is defined directly
// over the elements of tensors y and x, and doesn't depend on their shapes.
//
// If x = (x_1, x_2, ..., x_m) and y = (y_1, y_2, .., y_n) the matrix evaluated
// is actually the Jacobian transpose, defined as this mxn matrix:
// dy_1/d_x1 dy_2/dx_1 ... dy_n/dx_1
// dy_1/dx_2 dy_2/dx_2 ... dy_n/dx_2
//     .
//     .
//     .
// dy_1/dx_m dy_2/dx_m ... dy_n/dx_m
template <typename X_T, typename Y_T, typename JAC_T>
inline void GradientChecker<X_T, Y_T, JAC_T>::InitJacobians(size_t row_count, size_t col_count,
                                                            std::vector<std::vector<JAC_T>>* jacobians) {
  // the number of rows is equal to total number of scalar input values in all of input vectors
  jacobians->resize(row_count);
  // the number of cols is equal to total number of scalar output values in all of output vectors
  for (size_t i = 0; i < row_count; ++i) {
    (*jacobians)[i] = std::vector<JAC_T>(col_count, 0);
  }
}

template <typename X_T, typename Y_T, typename JAC_T>
inline std::vector<OrtValue> GradientChecker<X_T, Y_T, JAC_T>::EvaluateFunctionAtInput(
    GradientOpTester& op_tester, const std::vector<TensorInfo>& x_infos, const std::vector<TensorInfo>& y_infos,
    std::vector<std::vector<X_T>>* x_datas, std::vector<std::vector<Y_T>>* y_datas,
    std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers) {
  AddDatas(op_tester, x_infos, y_infos, x_datas, y_datas);

  // If EPs is not set, the OpTester will run over all possible EPs and keep the outputs of last run as the
  // actual output data, which is time wasting. What we need is the forward graph outputs for numeric Jacobian,
  // using CPU EP only is enough.
  std::vector<std::unique_ptr<IExecutionProvider>> eps = GetExecutionProviders(execution_providers, true);
  op_tester.Run(&eps);
  return op_tester.GetFetches();
}

template <typename X_T, typename Y_T, typename JAC_T>
inline void GradientChecker<X_T, Y_T, JAC_T>::AddDatas(GradientOpTester& op_tester, const std::vector<TensorInfo>& x_infos,
                                                       const std::vector<TensorInfo>& y_infos,
                                                       std::vector<std::vector<X_T>>* x_datas,
                                                       std::vector<std::vector<Y_T>>* y_datas) {
  op_tester.ClearData();
  for (size_t data_index = 0; data_index < x_datas->size(); ++data_index) {
    std::string name = "input" + std::to_string(data_index);
    const std::vector<X_T>& data = (*x_datas)[data_index];

    if (x_infos[data_index].data_type == DataTypeImpl::GetTensorType<int64_t>()) {
      std::vector<int64_t> int64_data(data.size());
      std::transform(data.begin(), data.end(), int64_data.begin(), [](X_T x) { return static_cast<int64_t>(x); });
      op_tester.AddInput<int64_t>(name.c_str(), x_infos[data_index].shape.AsShapeVector(), int64_data, false,
                                  &x_infos[data_index].dim_params);
    } else if (x_infos[data_index].data_type == DataTypeImpl::GetTensorType<int32_t>()) {
      std::vector<int32_t> int32_data(data.size());
      std::transform(data.begin(), data.end(), int32_data.begin(), [](X_T x) { return static_cast<int32_t>(x); });
      op_tester.AddInput<int32_t>(name.c_str(), x_infos[data_index].shape.AsShapeVector(), int32_data, false,
                                  &x_infos[data_index].dim_params);
    } else if (x_infos[data_index].data_type == DataTypeImpl::GetTensorType<bool>()) {
      std::unique_ptr<bool[]> p_data(new bool[data.size()]);
      for (size_t i = 0; i < data.size(); ++i) {
        p_data[i] = static_cast<bool>(data[i]);
      }
      op_tester.AddInput<bool>(name.c_str(), x_infos[data_index].shape.AsShapeVector(), p_data.get(), data.size(),
                               false, &x_infos[data_index].dim_params);
    } else {
      op_tester.AddInput<X_T>(name.c_str(), x_infos[data_index].shape.AsShapeVector(), data, false,
                              &x_infos[data_index].dim_params);
    }
  }

  for (size_t data_index = 0; data_index < y_infos.size(); ++data_index) {
    std::string name = "output" + std::to_string(data_index);
    const std::vector<Y_T>& data = (*y_datas)[data_index];

    if (y_infos[data_index].data_type == DataTypeImpl::GetTensorType<int64_t>()) {
      std::vector<int64_t> int64_data(data.size());
      std::transform(data.begin(), data.end(), int64_data.begin(), [](Y_T x) { return static_cast<int64_t>(x); });
      op_tester.AddOutput<int64_t>(name.c_str(), y_infos[data_index].shape.AsShapeVector(), int64_data);
    } else {
      op_tester.AddOutput<Y_T>(name.c_str(), y_infos[data_index].shape.AsShapeVector(), data);
    }
  }
}

template <typename X_T, typename Y_T, typename JAC_T>
inline Status GradientChecker<X_T, Y_T, JAC_T>::ComputeTheoreticalJacobianTranspose(
    const OpDef& op_def, const std::vector<TensorInfo>& x_infos, const std::vector<TensorInfo>& y_infos,
    std::vector<std::vector<X_T>>* x_datas, std::vector<std::vector<Y_T>>* y_datas,
    std::vector<std::vector<JAC_T>>* jacobian_ts, const std::vector<size_t>& row_strides,
    const std::vector<size_t>& col_strides, const std::vector<AttributeProto>& attributes, bool add_shape,
    std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers /* nullptr*/) {
  size_t y_num = y_infos.size();
  size_t x_num = x_infos.size();
  // build the graph once and reuse it later in the looping logic
  GradientOpTester op_tester(op_def.type.c_str(), op_def.opset_version, op_def.domain.c_str(),
                             false, &x_infos, &y_infos);
  op_tester.AddShapeToTensorData(add_shape);
  ORT_RETURN_IF_ERROR(InitOpTesterWithGradGraph(op_tester, x_infos, y_infos, x_datas, y_datas, attributes));

  // currently only supported scalar valued fns - and complex types are not supported
  for (size_t y_idx = 0; y_idx < y_num; y_idx++) {  // for each dy input
    if (!y_infos[y_idx].has_gradient) {
      continue;
    }

    const size_t dy_size = static_cast<size_t>(y_infos[y_idx].shape.Size());

    // Compute the theoretical Jacobians one row at a time by back propagating
    // '1.0' for each element of 'dy', while holding all other elements of 'dy' at zero.
    for (size_t c = 0; c < dy_size; ++c) {  // for each value in the dy input vector
      AddDatas(op_tester, x_infos, y_infos, x_datas, y_datas);

      // While calculating theoretical jacobian transpose we calculate the gradient by
      // setting back propagating one element of dY at a time and setting everything else to zero
      // as explained above. The input itself is unrolled into one big vector and the collection of
      // inputs is treated as a vector of vectors. The parameters of the function call below, y_idx and c
      // corresponding to which input (dy1, dy2..etc) and which value of the input (dy_flattened_vector[c]]
      // to perturb to 1.
      // If EPs is not set, the OpTester will run over all possible EPs and keep the outputs of last run as the
      // actual output data, which is time wasting. So if caller doesn't pass in the EPs, we will use the default
      // EPs according to the environment.
      std::vector<std::unique_ptr<IExecutionProvider>> eps = GetExecutionProviders(execution_providers);
      op_tester.Run(static_cast<int>(y_idx), static_cast<int>(c), &eps);
      auto gradients = op_tester.GetFetches();

      for (size_t x_idx = 0, grad_idx = 0; x_idx < x_num; x_idx++) {
        if (!x_infos[x_idx].has_gradient) {
          continue;
        }

        const size_t x_size = static_cast<size_t>(x_infos[x_idx].shape.Size());
        auto dx_flat = gradients[grad_idx].Get<Tensor>().Data<X_T>();
        grad_idx++;

        for (size_t r = 0; r < x_size; ++r) {
          (*jacobian_ts)[row_strides[x_idx] + r][col_strides[y_idx] + c] = dx_flat[r];
        }
      }
    }
  }

  return Status::OK();
}

template <typename X_T, typename Y_T, typename JAC_T>
inline Status GradientChecker<X_T, Y_T, JAC_T>::InitOpTesterWithGraph(
    GradientOpTester& op_tester, const std::vector<TensorInfo>& x_infos, const std::vector<TensorInfo>& y_infos,
    std::vector<std::vector<X_T>>* x_datas, std::vector<std::vector<Y_T>>* y_datas,
    const std::vector<AttributeProto>& attributes,
    const std::unordered_map<std::string, int>& extra_domain_to_version) {
  AddDatas(op_tester, x_infos, y_infos, x_datas, y_datas);
  // Currently only allows setting int attributes to zero. TODO: Expand this
  for (auto attr : attributes) {
    op_tester.AddAttributeProto(attr);
  }

  // build graph
  return op_tester.BuildAndCacheModel(extra_domain_to_version);
}

template <typename X_T, typename Y_T, typename JAC_T>
inline Status GradientChecker<X_T, Y_T, JAC_T>::InitOpTesterWithGradGraph(
    GradientOpTester& op_tester, const std::vector<TensorInfo>& x_infos, const std::vector<TensorInfo>& y_infos,
    std::vector<std::vector<X_T>>* x_datas, std::vector<std::vector<Y_T>>* y_datas,
    const std::vector<AttributeProto>& attributes) {
  std::unordered_map<std::string, int> extra_domain_to_version{{kMSDomain, 1}, {kOnnxDomain, 9}};
  ORT_RETURN_IF_ERROR(
      InitOpTesterWithGraph(op_tester, x_infos, y_infos, x_datas, y_datas, attributes, extra_domain_to_version));

  // build grad graph
  auto& model = op_tester.GetModel();
  auto& graph = model.MainGraph();

  std::unordered_set<std::string> weights_to_train;
  const auto& input_data = op_tester.GetInputData();
  const auto& output_data = op_tester.GetOutputData();

  for (size_t i = 0; i < input_data.size(); i++) {
    if (x_infos[i].has_gradient) {
      weights_to_train.insert(input_data[i].def.Name());
    }
  }

  std::unordered_set<std::string> dy_values;
  for (size_t i = 0; i < output_data.size(); i++) {
    if (y_infos[i].has_gradient) {
      dy_values.insert(output_data[i].def.Name());
    }
  }

  training::GradientGraphConfiguration gradient_graph_config;
  gradient_graph_config.set_gradients_as_graph_outputs = true;
  training::GradientGraphBuilder grad_graph_builder(&graph, dy_values, weights_to_train, "", gradient_graph_config,
                                                    logging::LoggingManager::DefaultLogger());
  return grad_graph_builder.Build();
}

template <typename X_T, typename Y_T, typename JAC_T>
inline Status GradientChecker<X_T, Y_T, JAC_T>::ComputeNumericJacobianTranspose(
    const OpDef& op_def, const std::vector<TensorInfo>& x_infos, const std::vector<TensorInfo>& y_infos,
    const JAC_T delta, std::vector<std::vector<X_T>>* x_datas, std::vector<std::vector<Y_T>>* y_datas,
    std::vector<std::vector<JAC_T>>* jacobian_ts, const std::vector<size_t>& row_strides,
    const std::vector<size_t>& col_strides, const std::vector<AttributeProto>& attributes, bool add_shape,
    std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers) {
  size_t y_num = y_infos.size();
  size_t x_num = x_infos.size();
  X_T x_delta = static_cast<X_T>(delta);

  // build the graph once and reuse it later in the looping logic
  GradientOpTester op_tester(op_def.type.c_str(), op_def.opset_version, op_def.domain.c_str(), false);
  op_tester.AddShapeToTensorData(add_shape);
  ORT_RETURN_IF_ERROR(InitOpTesterWithGraph(op_tester, x_infos, y_infos, x_datas, y_datas, attributes));

  for (size_t x_idx = 0; x_idx < x_num; ++x_idx) {
    if (!x_infos[x_idx].has_gradient) {
      continue;
    }

    const size_t x_size = static_cast<size_t>(x_infos[x_idx].shape.Size());

    // Compute the numeric Jacobian one column at a time by perturbing each
    // element of 'x_data' (positively and negatively) by 'delta', and
    // updating the jacobian with the centered difference
    for (size_t r = 0; r < x_size; ++r) {
      // Store current value of 'x' at 'r'.
      X_T v = (*x_datas)[x_idx][r];

      // Evaluate at positive delta.
      (*x_datas)[x_idx][r] = v + x_delta;
      std::vector<OrtValue> y_plus = EvaluateFunctionAtInput(op_tester, x_infos, y_infos, x_datas, y_datas, execution_providers);

      // Evaluate at negative delta.
      (*x_datas)[x_idx][r] = v - x_delta;
      std::vector<OrtValue> y_minus =
          EvaluateFunctionAtInput(op_tester, x_infos, y_infos, x_datas, y_datas, execution_providers);

      for (size_t y_idx = 0; y_idx < y_num; ++y_idx) {
        if (!y_infos[y_idx].has_gradient) {
          continue;
        }
        // Compute element-wise centered difference and store in each Jacobian.
        auto y_plus_flat = y_plus[y_idx].Get<Tensor>().Data<Y_T>();
        auto y_minus_flat = y_minus[y_idx].Get<Tensor>().Data<Y_T>();
        const size_t y_size = static_cast<size_t>(y_infos[y_idx].shape.Size());
        const Y_T scale = static_cast<Y_T>(2 * delta);
        for (size_t c = 0; c < y_size; ++c) {
          (*jacobian_ts)[row_strides[x_idx] + r][col_strides[y_idx] + c] = (y_plus_flat[c] - y_minus_flat[c]) / scale;
        }
      }

      // Restore pre-perturbation value.
      (*x_datas)[x_idx][r] = v;
    }
  }

  return Status::OK();
}

template <typename X_T, typename Y_T, typename JAC_T>
inline Status GradientChecker<X_T, Y_T, JAC_T>::ComputeGradientErrorInternal(
    const OpDef& op_def, const std::vector<TensorInfo>& x_infos, const std::vector<TensorInfo>& y_infos,
    std::vector<std::vector<X_T>>* x_datas, std::vector<std::vector<Y_T>>* y_datas, JAC_T* max_error,
    const std::vector<AttributeProto>& attributes, bool check_not_have_gradient, bool check_not_have_shape_inferencing,
    std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers /* nullptr */) {
  std::vector<size_t> row_strides(x_infos.size());
  std::vector<size_t> col_strides(y_infos.size());
  size_t row_count = 0;
  for (size_t i = 0; i < x_infos.size(); ++i) {
    row_strides[i] = row_count;
    row_count += static_cast<size_t>(x_infos[i].shape.Size());
  }

  size_t col_count = 0;
  for (size_t i = 0; i < y_infos.size(); ++i) {
    col_strides[i] = col_count;
    col_count += static_cast<size_t>(y_infos[i].shape.Size());
  }

  // Initialize numeric Jacobian to zeros.
  std::vector<std::vector<JAC_T>> jacobian_ns;
  InitJacobians(row_count, col_count, &jacobian_ns);

  // Compute numeric Jacobian.
  ORT_RETURN_IF_ERROR(ComputeNumericJacobianTranspose(op_def, x_infos, y_infos, JAC_T{1e-3f}, x_datas, y_datas,
                                                      &jacobian_ns, row_strides, col_strides, attributes, true,
                                                      execution_providers));

  // Compute the maximum error between theoretical and numeric Jacobians.
  *max_error = 0.0;

  int num_grad_builder_checks = check_not_have_shape_inferencing ? 2 : 1;
  bool add_shape = true;
  for (int i = 0; i < num_grad_builder_checks; i++, add_shape = false) {
    // It is necessary to test for inputs with or without gradient.
    // We simply set each input without gradient to test the rest inputs' gradient.
    // In the last loop it tests for the case where all inputs are with gradient.
    size_t total_gradient_variations = check_not_have_gradient ? x_infos.size() + 1 : 1;
    for (size_t x_gradient_variation = 0; x_gradient_variation < total_gradient_variations; x_gradient_variation++) {
      // Initialize theoretical Jacobians to zeros.
      std::vector<std::vector<JAC_T>> jacobian_ts;
      InitJacobians(row_count, col_count, &jacobian_ts);

      std::vector<TensorInfo> x_infos_gradient_variation = x_infos;
      if (check_not_have_gradient && x_gradient_variation < x_infos.size()) {
        x_infos_gradient_variation[x_gradient_variation].has_gradient = false;
      }

      // a gradient node cannot get created without any has_gradient node.
      if (std::all_of(x_infos_gradient_variation.cbegin(), x_infos_gradient_variation.cend(),
                      [](const TensorInfo& info) { return !info.has_gradient; })) {
        continue;
      }

      // Compute theoretical Jacobian.
      ORT_RETURN_IF_ERROR(ComputeTheoreticalJacobianTranspose(op_def, x_infos_gradient_variation, y_infos, x_datas,
                                                              y_datas, &jacobian_ts, row_strides, col_strides,
                                                              attributes, add_shape, execution_providers));

      // We have numeric jacobians regardless of has_gradient (computed once).
      // We only have theoretical jacobians for those has_gradient.
      // Theoretical jacobians are 0 for those not has_gradient.
      size_t j = 0;
      for (auto& x_info : x_infos_gradient_variation) {
        const size_t x_size = static_cast<size_t>(x_info.shape.Size());
        if (!x_info.has_gradient) {
          // TODO: These 4 test failed at following ORT_ENFORCE. need investigate before enable it.
          // GradientCheckerTest.MatMulGrad
          // GradientCheckerTest.GemmGrad
          // GradientCheckerTest.GatherNDGrad_repeat_float_data
          // GradientCheckerTest.GatherNDGrad_unique_float_data
          // auto jac_t = jacobian_ts[j];
          // ORT_ENFORCE(std::all_of(
          //    &jac_t[0], &jac_t[0] + x_info.shape.Size(), [](auto dx) { return dx == 0; }));
          j += x_size;
        } else {
          for (size_t r = 0; r < x_size; j++, r++) {
            auto jac_t = jacobian_ts[j];
            auto jac_n = jacobian_ns[j];
            for (size_t k = 0; k < jac_t.size(); k++) {
              // dy_i/dx_j for x with gradient.
              auto cur_error = std::fabs(jac_t[k] - jac_n[k]);
              // Treat any NaN as max_error and immediately return.
              // (Note that std::max may ignore NaN arguments.)
              if (std::isnan(cur_error)) {
                *max_error = cur_error;
                return Status::OK();
              }
              *max_error = std::max(*max_error, cur_error);
            }
          }
        }
      }
    }
  }

  return Status::OK();
}

template <typename X_T, typename Y_T, typename JAC_T>
inline Status GradientChecker<X_T, Y_T, JAC_T>::ComputeGradientError(
    const OpDef& op_def, const std::vector<TensorInfo>& x_infos, const std::vector<TensorInfo>& y_infos,
    JAC_T* max_error, const std::vector<AttributeProto>& attributes, bool check_not_have_gradient, /* = true*/
    bool check_not_have_shape_inferencing /* = false*/,
    std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers /* = nullptr */) {
  // TODO: Consider varying mean and variance
  constexpr float scale = 5.f;
  constexpr float mean = 0.f;
  constexpr int seed = 5566;
  // Use fixed random since numerically compute gradient is not stable.
  std::default_random_engine generator{gsl::narrow_cast<decltype(generator)::result_type>(seed)};
  std::normal_distribution<X_T> distribution{mean, scale};

  // Initialize 'x_datas' to random values.
  std::vector<std::vector<X_T>> x_datas(x_infos.size());
  for (size_t i = 0; i < x_infos.size(); i++) {
    x_datas[i].resize(static_cast<size_t>(x_infos[i].shape.Size()));

    if (x_infos[i].transformer) {
      auto transformer = *x_infos[i].transformer;
      std::generate(x_datas[i].begin(), x_datas[i].end(),
                    [&] { return transformer(static_cast<float>(distribution(generator))); });
    } else {
      std::generate(x_datas[i].begin(), x_datas[i].end(), [&] { return distribution(generator); });
    }
  }

  return ComputeGradientError(op_def, x_infos, y_infos, max_error, x_datas, attributes, check_not_have_gradient,
                              check_not_have_shape_inferencing, execution_providers);
}

template <typename X_T, typename Y_T, typename JAC_T>
inline Status GradientChecker<X_T, Y_T, JAC_T>::ComputeGradientError(
    const OpDef& op_def, const std::vector<TensorInfo>& x_infos, const std::vector<TensorInfo>& y_infos,
    JAC_T* max_error, std::vector<std::vector<X_T>> x_datas,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attributes, bool check_not_have_gradient, /* = true*/
    bool check_not_have_shape_inferencing /* = false*/,
    std::vector<std::unique_ptr<IExecutionProvider>>* execution_providers /* = nullptr */) {
  // Generate dummy placeholders with zero for y_datas
  std::vector<std::vector<Y_T>> y_datas(y_infos.size());
  for (size_t i = 0; i < y_infos.size(); i++) {
    y_datas[i].resize(static_cast<size_t>(y_infos[i].shape.Size()), 0);
  }

  // Compute gradient error.
  return ComputeGradientErrorInternal(op_def, x_infos, y_infos, &x_datas, &y_datas, max_error, attributes,
                                      check_not_have_gradient, check_not_have_shape_inferencing, execution_providers);
}

#define INSTANTIATE_GRAD_ERR_TYPE(X_T, Y_T, JAC_T) template class GradientChecker<X_T, Y_T, JAC_T>;

INSTANTIATE_GRAD_ERR_TYPE(float, float, float);
INSTANTIATE_GRAD_ERR_TYPE(double, double, double);

#undef INSTANTIATE_GRAD_ERR_TYPE

}  // namespace test
}  // namespace onnxruntime
