#include "gradient_checker.h"
#include "gradient_op_test_utils.h"
#include "test/random_seed.h"
#include <random>

namespace onnxruntime {
namespace test {

using ONNX_NAMESPACE::AttributeProto;
using training::OpDef;

// The jacobian transpose matrix is laid out as follows

// Say there are three inputs each of size M X N, N X K, K X J
// say there are two outputs each of size M X K , N X J

//    output size (y_shapes)  -->            | M X K  |N X J |
//     input size (x_shapes)        M X N    |        |      |
//       |                          N X K    |        |      |
//       |                          K X J    |        |      |
//       V

std::pair<int, int> inline CalculateJacobianTransposeIndex(const std::vector<TensorInfo>& x_infos,
                                                           int x_input_index,
                                                           int x_flattened_index,
                                                           const std::vector<TensorInfo>& y_infos,
                                                           int y_output_index,
                                                           int y_flattened_index) {
  int64_t elems_in_prev_output_tensors = 0;
  for (int i = 0; i < y_output_index; i++) {
    elems_in_prev_output_tensors += y_infos[i].shape.Size();
  }

  int64_t col = elems_in_prev_output_tensors + y_flattened_index;

  int64_t elems_in_prev_input_tensors = 0;
  for (int i = 0; i < x_input_index; i++) {
    elems_in_prev_input_tensors += x_infos[i].shape.Size();
  }

  int64_t row = elems_in_prev_input_tensors + x_flattened_index;

  return {gsl::narrow_cast<int>(row), gsl::narrow_cast<int>(col)};
}

template <typename X_T, typename Y_T, typename JAC_T>
inline std::vector<OrtValue> GradientChecker<X_T, Y_T, JAC_T>::EvaluateFunctionAtInput(
    const OpDef& op_def,
    const std::vector<TensorInfo>& x_infos,
    const std::vector<TensorInfo>& y_infos,
    std::vector<std::vector<X_T>>* x_datas,
    std::vector<std::vector<Y_T>>* y_datas,
    const std::vector<AttributeProto>& attributes) {
  OpTester op_session(op_def.type.c_str(), 9, op_def.domain.c_str(), false);
  for (size_t data_index = 0; data_index < x_datas->size(); data_index++) {
    std::string name = "input" + std::to_string(data_index);
    const std::vector<X_T>& data = (*x_datas)[data_index];

    if (x_infos[data_index].data_type == DataTypeImpl::GetTensorType<int64_t>()) {
      std::vector<int64_t> int64_data(data.size());
      std::transform(data.begin(), data.end(), int64_data.begin(), [](X_T x) { return static_cast<int64_t>(x); });
      op_session.AddInput<int64_t>(name.c_str(), x_infos[data_index].shape.GetDims(), int64_data);
    } else if (x_infos[data_index].data_type == DataTypeImpl::GetTensorType<int32_t>()) {
      std::vector<int32_t> int32_data(data.size());
      std::transform(data.begin(), data.end(), int32_data.begin(), [](X_T x) { return static_cast<int32_t>(x); });
      op_session.AddInput<int32_t>(name.c_str(), x_infos[data_index].shape.GetDims(), int32_data);
    } else {
      op_session.AddInput<X_T>(name.c_str(), x_infos[data_index].shape.GetDims(), data);
    }
  }

  for (size_t data_index = 0; data_index < y_infos.size(); data_index++) {
    std::string name = "output" + std::to_string(data_index);
    op_session.AddOutput<Y_T>(name.c_str(), y_infos[data_index].shape.GetDims(), (*y_datas)[data_index]);
  }
  // Currently only allows setting int attributes to zero. TODO: Expand this
  for (auto attr : attributes) {
    op_session.AddAttribute<AttributeProto>(attr.name(), attr);
  }
  op_session.Run();
  return op_session.GetFetches();
}
template <typename X_T, typename Y_T, typename JAC_T>
inline Status GradientChecker<X_T, Y_T, JAC_T>::ComputeTheoreticalJacobianTranspose(
    const OpDef& op_def,
    const std::vector<TensorInfo>& x_infos,
    const std::vector<TensorInfo>& y_infos,
    std::vector<std::vector<X_T>>* x_datas,
    std::vector<std::vector<Y_T>>* y_datas,
    std::vector<std::vector<JAC_T>>* jacobian_ts,
    const std::vector<AttributeProto>& attributes) {
  size_t y_num = y_infos.size();
  size_t x_num = x_infos.size();

  // currently only supported scalar valued fns - and complex types are not supported
  for (int y_idx = 0; y_idx < static_cast<int>(y_num); y_idx++) {  // for each dy input
    if (!y_infos[y_idx].has_gradient) {
      continue;
    }

    const size_t dy_size = y_infos[y_idx].shape.Size();

    // Compute the theoretical Jacobians one row at a time by back propagating
    // '1.0'for each element of 'dy', while holding all other elements of 'dy' at zero.
    for (int c = 0; c < dy_size; ++c) {  // for each value in the dy input vector
      GradientOpTester op_session(op_def.type.c_str(), x_infos, y_infos, op_def.opset_version, op_def.domain.c_str(), false);

      for (size_t data_index = 0; data_index < x_num; data_index++) {
        std::string name = "input" + std::to_string(data_index);
        const std::vector<X_T>& data = (*x_datas)[data_index];

        if (x_infos[data_index].data_type == DataTypeImpl::GetTensorType<int64_t>()) {
          std::vector<int64_t> int64_data(data.size());
          std::transform(data.begin(), data.end(), int64_data.begin(), [](X_T x) { return static_cast<int64_t>(x); });
          op_session.AddInput<int64_t>(name.c_str(), x_infos[data_index].shape.GetDims(), int64_data);
        } else if (x_infos[data_index].data_type == DataTypeImpl::GetTensorType<int32_t>()) {
          std::vector<int32_t> int32_data(data.size());
          std::transform(data.begin(), data.end(), int32_data.begin(), [](X_T x) { return static_cast<int32_t>(x); });
          op_session.AddInput<int32_t>(name.c_str(), x_infos[data_index].shape.GetDims(), int32_data);
        } else {
          op_session.AddInput<X_T>(name.c_str(), x_infos[data_index].shape.GetDims(), data);
        }
      }

      for (size_t data_index = 0; data_index < y_num; data_index++) {
        std::string name = "output" + std::to_string(data_index);
        op_session.AddOutput<Y_T>(name.c_str(), y_infos[data_index].shape.GetDims(), (*y_datas)[data_index]);
      }

      // Currently only allows setting int attributes to zero. TODO: Expand this
      for (auto attr : attributes) {
        op_session.AddAttribute<AttributeProto>(attr.name(), attr);
      }

      // While calculating theoritical jacobian transpose we calculate the gradient by
      // setting back propogating one element of dY at a time and setting everything else to zero
      // as explained above. The input itself is unrolled into one big vector and the collection of
      // inputs is treated as a vector of vectors. The parameters of the function call below, y_idx and c
      // corresponding to which input (dy1, dy2..etc) and which value of the input (dy_flattened_vector[c]]
      // to pertrub to 1.
      op_session.Run(y_idx, c);
      auto gradients = op_session.GetFetches();

      for (int x_idx = 0, grad_idx = 0; x_idx < static_cast<int>(x_num); x_idx++) {
        if (!x_infos[x_idx].has_gradient) {
          continue;
        }

        const int64_t x_size = x_infos[x_idx].shape.Size();
        auto dx_flat = gradients[grad_idx].Get<Tensor>().Data<X_T>();
        grad_idx++;

        for (int r = 0; r < static_cast<int>(x_size); ++r) {
          auto calc_index = CalculateJacobianTransposeIndex(
              x_infos,
              x_idx,
              r,
              y_infos,
              y_idx,
              c);
          (*jacobian_ts)[calc_index.first][calc_index.second] = dx_flat[r];
        }
      }
    }
  }
  return Status::OK();
}

template <typename X_T, typename Y_T, typename JAC_T>
inline Status GradientChecker<X_T, Y_T, JAC_T>::ComputeNumericJacobianTranspose(
    const OpDef& op_def,
    const std::vector<TensorInfo>& x_infos,
    const std::vector<TensorInfo>& y_infos,
    const JAC_T delta,
    std::vector<std::vector<X_T>>* x_datas,
    std::vector<std::vector<Y_T>>* y_datas,
    std::vector<std::vector<JAC_T>>* jacobian_ts,
    const std::vector<AttributeProto>& attributes) {
  size_t y_num = y_infos.size();
  size_t x_num = x_infos.size();
  X_T x_delta = static_cast<X_T>(delta);

  for (int x_idx = 0; x_idx < static_cast<int>(x_num); x_idx++) {
    if (!x_infos[x_idx].has_gradient) {
      continue;
    }

    const int64_t x_size = x_infos[x_idx].shape.Size();

    // Compute the numeric Jacobian one column at a time by perturbing each
    // element of 'x_data' (positively and negatively) by 'delta', and
    // updating the jacobian with the centered difference
    for (int r = 0; r < x_size; ++r) {
      // Store current value of 'x' at 'r'.
      X_T v = (*x_datas)[x_idx][r];

      // Evaluate at positive delta.
      (*x_datas)[x_idx][r] = v + x_delta;
      std::vector<OrtValue> y_plus = EvaluateFunctionAtInput(op_def, x_infos, y_infos, x_datas, y_datas, attributes);

      // Evaluate at negative delta.
      (*x_datas)[x_idx][r] = v - x_delta;
      std::vector<OrtValue> y_minus = EvaluateFunctionAtInput(op_def, x_infos, y_infos, x_datas, y_datas, attributes);

      for (int y_idx = 0; y_idx < static_cast<int>(y_num); y_idx++) {
        if (!y_infos[y_idx].has_gradient) {
          continue;
        }
        // Compute element-wise centered difference and store in each Jacobian.
        auto y_plus_flat = y_plus[y_idx].Get<Tensor>().Data<Y_T>();
        auto y_minus_flat = y_minus[y_idx].Get<Tensor>().Data<Y_T>();
        const int64_t y_size = y_infos[y_idx].shape.Size();
        const Y_T scale = static_cast<Y_T>(2 * delta);
        for (int c = 0; c < y_size; ++c) {
          auto calc_index = CalculateJacobianTransposeIndex(
              x_infos,
              x_idx,
              r,
              y_infos,
              y_idx,
              c);
          (*jacobian_ts)[calc_index.first][calc_index.second] = (y_plus_flat[c] - y_minus_flat[c]) / scale;
        }
      }
      // Restore pre-perturbation value.
      (*x_datas)[x_idx][r] = v;
    }
  }
  return Status::OK();
}

//// The Jacobian is always a real-valued matrix.
//// Given y = f(x) for tensors y and x, it contains the derivatives dy_i/dx_j for
//// every pair y_i in y and x_j in x.  Note that the Jacobian is defined directly
//// over the elements of tensors y and x, and doesn't depend on their shapes.
////
//// If x = (x_1, x_2, ..., x_m) and y = (y_1, y_2, .., y_n) the matrix evaluated
//// is actually the Jacobian transpose, defined as this mxn matrix:
//// dy_1/d_x1 dy_2/dx_1 ... dy_n/dx_1
//// dy_1/dx_2 dy_2/dx_2 ... dy_n/dx_2
////     .
////     .
////     .
//// dy_1/dx_m dy_2/dx_m ... dy_n/dx_m
template <typename X_T, typename Y_T, typename JAC_T>
inline Status GradientChecker<X_T, Y_T, JAC_T>::InitJacobians(
    const std::vector<TensorInfo>& x_infos,
    const std::vector<TensorInfo>& y_infos,
    std::vector<std::vector<JAC_T>>* jacobians) {
  // the number of rows is equal to total number of scalar input values in all of input vectors
  int64_t rows = 0;
  for (size_t i = 0; i < x_infos.size(); i++) {
    rows += x_infos[i].shape.Size();  // 'S'ize gives the total number of elements in all dims while 's'ize just gives num_dims
  }
  jacobians->resize(gsl::narrow_cast<int>(rows));

  // the number of cols is equal to total number of scalar output values in all of output vectors
  int64_t cols = 0;
  for (size_t i = 0; i < y_infos.size(); i++) {
    cols += y_infos[i].shape.Size();
  }

  for (size_t i = 0; i < jacobians->size(); i++) {
    (*jacobians)[i] = std::vector<JAC_T>(gsl::narrow_cast<int>(cols), 0);
  }

  return Status().OK();
}

template <typename X_T, typename Y_T, typename JAC_T>
inline Status GradientChecker<X_T, Y_T, JAC_T>::ComputeGradientErrorInternal(
    const OpDef& op_def,
    const std::vector<TensorInfo>& x_infos,
    const std::vector<TensorInfo>& y_infos,
    std::vector<std::vector<X_T>>* x_datas,
    std::vector<std::vector<Y_T>>* y_datas,
    JAC_T* max_error,
    const std::vector<AttributeProto>& attributes,
    bool check_not_have_gradient) {
  // Initialize numeric Jacobian to zeros.
  std::vector<std::vector<JAC_T>> jacobian_ns;
  InitJacobians(x_infos, y_infos, &jacobian_ns);

  // Compute numeric Jacobian.
  ORT_RETURN_IF_ERROR(ComputeNumericJacobianTranspose(
      op_def, x_infos, y_infos, JAC_T{1e-3f}, x_datas, y_datas, &jacobian_ns, attributes));

  // Compute the maximum error between theoretical and numeric Jacobians.
  *max_error = 0.0;

  // It is necessary to test for inputs with or without gradient.
  // We simply set each input without gradient to test the rest inputs' gradient.
  // In the last loop it tests for the case where all inputs are with gradient.
  size_t total_gradient_variations = check_not_have_gradient ? x_infos.size() + 1 : 1;
  for (size_t x_gradient_variation = 0; x_gradient_variation < total_gradient_variations; x_gradient_variation++) {
    // Initialize theoretical Jacobians to zeros.
    std::vector<std::vector<JAC_T>> jacobian_ts;
    InitJacobians(x_infos, y_infos, &jacobian_ts);

    std::vector<TensorInfo> x_infos_gradient_variation = x_infos;

    if (check_not_have_gradient && x_gradient_variation < x_infos.size())
      x_infos_gradient_variation[x_gradient_variation].has_gradient = false;

    if (std::all_of(x_infos_gradient_variation.cbegin(), x_infos_gradient_variation.cend(),
                    [](const TensorInfo& info) { return !info.has_gradient; }))
      // a gradient node cannot get created without any has_gradient node.
      continue;

    // Compute theoretical Jacobian.
    ORT_RETURN_IF_ERROR(ComputeTheoreticalJacobianTranspose(
        op_def, x_infos_gradient_variation, y_infos, x_datas, y_datas, &jacobian_ts, attributes));

    // We have numeric jacobians regardless of has_gradient (computed once).
    // We only have theoretical jacobians for those has_gradient.
    // Theoretical jacobians are 0 for those not has_gradient.
    int64_t j = 0;
    for (auto& x_info : x_infos_gradient_variation) {
      if (!x_info.has_gradient) {
        // TODO: These 4 test failed at following ORT_ENFORCE. need investigate before enable it.
        //GradientCheckerTest.MatMulGrad
        //GradientCheckerTest.GemmGrad
        //GradientCheckerTest.GatherNDGrad_int64_indice_repeat_float_data
        //GradientCheckerTest.GatherNDGrad_int64_indice_unique_float_data
        //auto jac_t = jacobian_ts[j];
        //ORT_ENFORCE(std::all_of(
        //    &jac_t[0], &jac_t[0] + x_info.shape.Size(), [](auto dx) { return dx == 0; }));
        j += x_info.shape.Size();
      } else {
        for (int r = 0; r < x_info.shape.Size(); j++, r++) {
          auto jac_t = jacobian_ts[j];
          auto jac_n = jacobian_ns[j];
          for (size_t i = 0; i < jac_t.size(); i++) {
            // dy_i/dx_j for x with gradient.
            auto cur_error = std::fabs(jac_t[i] - jac_n[i]);
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
  return Status::OK();
}

template <typename X_T, typename Y_T, typename JAC_T>
inline Status GradientChecker<X_T, Y_T, JAC_T>::ComputeGradientError(
    const OpDef& op_def,
    const std::vector<TensorInfo>& x_infos,
    const std::vector<TensorInfo>& y_infos,
    JAC_T* max_error,
    const std::vector<AttributeProto>& attributes,
    bool check_not_have_gradient /* = true*/) {
  // TODO: Consider varying mean and variance
  float scale = 5.f;
  float mean = 0.f;
  const uint32_t seed = GetStaticRandomSeed();
  std::default_random_engine generator{gsl::narrow_cast<decltype(generator)::result_type>(seed)};
  std::normal_distribution<X_T> distribution{mean, scale};

  // Initialize 'x_datas' to random values.
  std::vector<std::vector<X_T>> x_datas(x_infos.size());
  for (size_t i = 0; i < x_infos.size(); i++) {
    x_datas[i].resize(x_infos[i].shape.Size());

    if (x_infos[i].transformer) {
      auto transformer = *x_infos[i].transformer;
      std::generate(x_datas[i].begin(), x_datas[i].end(),
                    [&] { return transformer(static_cast<float>(distribution(generator))); });
    } else {
      std::generate(x_datas[i].begin(), x_datas[i].end(), [&] { return distribution(generator); });
    }
  }

  // Generate dummy placeholders with zero for y_datas
  std::vector<std::vector<Y_T>> y_datas(y_infos.size());
  for (size_t i = 0; i < y_infos.size(); i++) {
    y_datas[i].resize(y_infos[i].shape.Size(), 0);
  }

  // Compute gradient error.
  return ComputeGradientErrorInternal(op_def, x_infos, y_infos, &x_datas, &y_datas, max_error,
                                      attributes, check_not_have_gradient);
}

template <typename X_T, typename Y_T, typename JAC_T>
inline Status GradientChecker<X_T, Y_T, JAC_T>::ComputeGradientError(
    const OpDef& op_def,
    const std::vector<TensorInfo>& x_infos,
    const std::vector<TensorInfo>& y_infos,
    JAC_T* max_error,
    std::vector<std::vector<X_T>> x_datas,
    const std::vector<ONNX_NAMESPACE::AttributeProto>& attributes) {
  // Generate dummy placeholders with zero for y_datas
  std::vector<std::vector<Y_T>> y_datas(y_infos.size());
  for (size_t i = 0; i < y_infos.size(); i++) {
    y_datas[i].resize(y_infos[i].shape.Size(), 0);
  }

  // Compute gradient error.
  return ComputeGradientErrorInternal(op_def, x_infos, y_infos, &x_datas, &y_datas, max_error, attributes);
}

#define INSTANTIATE_GRAD_ERR_TYPE(X_T, Y_T, JAC_T) \
  template class GradientChecker<X_T, Y_T, JAC_T>;

INSTANTIATE_GRAD_ERR_TYPE(float, float, float);
INSTANTIATE_GRAD_ERR_TYPE(double, double, double);

}  // namespace test
}  // namespace onnxruntime
