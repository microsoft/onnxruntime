#include "gradient_checker.h"
#include "gradient_op_test_utils.h"
#include <random>

namespace onnxruntime {
namespace test {

// The jacobian transpose matrix is laid out as follows

// Say there are three inputs each of size M X N, N X K, K X J
// say there are two outputs each of size M X K , N X J

//    output size (y_shapes)  -->            | M X K  |N X J |
//     input size (x_shapes)        M X N    |        |      |
//       |                          N X K    |        |      |
//       |                          K X J    |        |      |
//       V

std::pair<int, int> inline CalculateJacobianTransposeIndex(const std::vector<TensorShape>& x_shapes,
                                                           int x_input_index,
                                                           int x_flattened_index,
                                                           const std::vector<TensorShape>& y_shapes,
                                                           int y_output_index,
                                                           int y_flattened_index) {
  int64_t elems_in_prev_output_tensors = 0;
  for (int i = 0; i < y_output_index; i++) {
    elems_in_prev_output_tensors += y_shapes[i].Size();
  }

  int64_t col = elems_in_prev_output_tensors + y_flattened_index;

  int64_t elems_in_prev_input_tensors = 0;
  for (int i = 0; i < x_input_index; i++) {
    elems_in_prev_input_tensors += x_shapes[i].Size();
  }

  int64_t row = elems_in_prev_input_tensors + x_flattened_index;

  return {gsl::narrow_cast<int>(row), gsl::narrow_cast<int>(col)};
}

template <typename X_T, typename Y_T, typename JAC_T>
inline std::vector<onnxruntime::MLValue> GradientChecker<X_T, Y_T, JAC_T>::EvaluateFunctionAtInput(
    const std::string& op_name,
    const std::vector<TensorShape>& x_shapes,
    const std::vector<TensorShape>& y_shapes,
    std::vector<std::vector<X_T>>* x_datas,
    std::vector<std::vector<Y_T>>* y_datas,
    std::vector<std::string> attributes) {
  OpTester op_session(op_name.c_str(), 9, "", false);
  for (int data_index = 0; data_index < x_datas->size(); data_index++) {
    std::string name = "input" + std::to_string(data_index);
    op_session.AddInput<X_T>(name.c_str(), x_shapes[data_index].GetDims(), (*x_datas)[data_index]);
  }

  for (int data_index = 0; data_index < y_shapes.size(); data_index++) {
    std::string name = "output" + std::to_string(data_index);
    op_session.AddOutput<X_T>(name.c_str(), y_shapes[data_index].GetDims(), (*y_datas)[data_index]);
  }
  // Currently only allows setting int attributes to zero. TODO: Expand this
  for (auto attr : attributes) {
    op_session.AddAttribute<int64_t>(attr, 0);
  }
  op_session.Run();
  return op_session.GetFetches();
}
template <typename X_T, typename Y_T, typename JAC_T>
inline Status GradientChecker<X_T, Y_T, JAC_T>::ComputeTheoreticalJacobianTranspose(
    std::string& op_name,
    const std::vector<TensorShape>& x_shapes,
    const std::vector<TensorShape>& y_shapes,
    std::vector<std::vector<X_T>>* x_datas,
    std::vector<std::vector<Y_T>>* y_datas,
    std::vector<std::vector<JAC_T>>* jacobian_ts,
    std::vector<std::string> attributes) {
  size_t y_num = y_shapes.size();
  size_t x_num = x_shapes.size();

  // currently only supported scalar valued fns - and complex types are not supported
  for (int y_idx = 0; y_idx < y_num; y_idx++) {  // for each dy input
    const int64_t dy_size = y_shapes[y_idx].Size();

    // Compute the theoretical Jacobians one row at a time by back propagating
    // '1.0'for each element of 'dy', while holding all other elements of 'dy' at zero.
    for (int c = 0; c < dy_size; ++c) {  // for each value in the dy input vector
      GradientOpTester op_session(op_name.c_str(), 9, "", false);

      for (int data_index = 0; data_index < x_num; data_index++) {
        std::string name = "input" + std::to_string(data_index);
        op_session.AddInput<X_T>(name.c_str(), x_shapes[data_index].GetDims(), (*x_datas)[data_index]);
      }

      for (int data_index = 0; data_index < y_num; data_index++) {
        std::string name = "output" + std::to_string(data_index);
        op_session.AddOutput<Y_T>(name.c_str(), y_shapes[data_index].GetDims(), (*y_datas)[data_index]);
      }

      // Currently only allows setting int attributes to zero. TODO: Expand this
      for (auto attr : attributes) {
        op_session.AddAttribute<int64_t>(attr, 0);
      }

      // While calculating theoritical jacobian transpose we calculate the gradient by
      // setting back propogating one element of dY at a time and setting everything else to zero
      // as explained above. The input itself is unrolled into one big vector and the collection of
      // inputs is treated as a vector of vectors. The parameters of the function call below, y_idx and c
      // corresponding to which input (dy1, dy2..etc) and which value of the input (dy_flattened_vector[c]]
      // to pertrub to 1.
      op_session.Run(y_idx, c);
      auto gradients = op_session.GetFetches();

      for (int x_idx = 0; x_idx < x_num; x_idx++) {
        const int64_t x_size = x_shapes[x_idx].Size();
        auto dx_flat = gradients[x_idx].Get<Tensor>().Data<Y_T>();
        for (int r = 0; r < x_size; ++r) {
          auto calc_index = CalculateJacobianTransposeIndex(
              x_shapes,
              x_idx,
              r,
              y_shapes,
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
    const std::string& op_name,
    const std::vector<TensorShape>& x_shapes,
    const std::vector<TensorShape>& y_shapes,
    const JAC_T delta,
    std::vector<std::vector<X_T>>* x_datas,
    std::vector<std::vector<Y_T>>* y_datas,
    std::vector<std::vector<JAC_T>>* jacobian_ts,
    std::vector<std::string> attributes) {
  size_t y_num = y_shapes.size();
  size_t x_num = x_shapes.size();
  X_T x_delta = X_T{delta};

  for (int x_idx = 0; x_idx < x_num; x_idx++) {
    const int64_t x_size = x_shapes[x_idx].Size();

    // Compute the numeric Jacobian one column at a time by perturbing each
    // element of 'x_data' (positively and negatively) by 'delta', and
    // updating the jacobian with the centered difference
    for (int r = 0; r < x_size; ++r) {
      // Store current value of 'x' at 'r'.
      X_T v = (*x_datas)[x_idx][r];

      // Evaluate at positive delta.
      (*x_datas)[x_idx][r] = v + x_delta;
      std::vector<onnxruntime::MLValue> y_plus = EvaluateFunctionAtInput(op_name, x_shapes, y_shapes, x_datas, y_datas, attributes);

      // Evaluate at negative delta.
      (*x_datas)[x_idx][r] = v - x_delta;
      std::vector<onnxruntime::MLValue> y_minus = EvaluateFunctionAtInput(op_name, x_shapes, y_shapes, x_datas, y_datas, attributes);

      for (int y_idx = 0; y_idx < y_num; y_idx++) {
        // Compute element-wise centered difference and store in each Jacobian.
        auto y_plus_flat = y_plus[y_idx].Get<Tensor>().Data<Y_T>();
        auto y_minus_flat = y_minus[y_idx].Get<Tensor>().Data<Y_T>();
        const int64_t y_size = y_shapes[y_idx].Size();
        const Y_T scale = 2 * delta;
        for (int c = 0; c < y_size; ++c) {
          auto calc_index = CalculateJacobianTransposeIndex(
              x_shapes,
              x_idx,
              r,
              y_shapes,
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
    const std::vector<TensorShape>& x_shapes,
    const std::vector<TensorShape>& y_shapes,
    std::vector<std::vector<JAC_T>>* jacobians) {
  // the number of rows is equal to total number of scalar input values in all of input vectors
  int64_t rows = 0;
  for (int i = 0; i < x_shapes.size(); i++) {
    rows += x_shapes[i].Size();  // 'S'ize gives the total number of elements in all dims while 's'ize just gives num_dims
  }
  jacobians->resize(gsl::narrow_cast<int>(rows));

  // the number of cols is equal to total number of scalar output values in all of output vectors
  int64_t cols = 0;
  for (int i = 0; i < y_shapes.size(); i++) {
    cols += y_shapes[i].Size();
  }

  for (int i = 0; i < jacobians->size(); i++) {
    (*jacobians)[i] = std::vector<JAC_T>(gsl::narrow_cast<int>(cols), 0);
  }

  return Status().OK();
}

template <typename X_T, typename Y_T, typename JAC_T>
inline Status GradientChecker<X_T, Y_T, JAC_T>::ComputeGradientErrorInternal(
    std::string& op_name,
    const std::vector<TensorShape>& x_shapes,
    const std::vector<TensorShape>& y_shapes,
    std::vector<std::vector<X_T>>* x_datas,
    std::vector<std::vector<Y_T>>* y_datas,
    JAC_T* max_error,
    std::vector<std::string> attributes) {
  // Initialize theoretical Jacobians to zeros.
  std::vector<std::vector<JAC_T>> jacobian_ts;
  InitJacobians(x_shapes, y_shapes, &jacobian_ts);

  // Compute theoretical Jacobian.
  ORT_RETURN_IF_ERROR(ComputeTheoreticalJacobianTranspose(
      op_name, x_shapes, y_shapes, x_datas, y_datas, &jacobian_ts, attributes));

  // Initialize numeric Jacobian to zeros.
  std::vector<std::vector<JAC_T>> jacobian_ns;
  InitJacobians(x_shapes, y_shapes, &jacobian_ns);

  // Compute numeric Jacobian.
  ORT_RETURN_IF_ERROR(ComputeNumericJacobianTranspose(
      op_name, x_shapes, y_shapes, JAC_T{1e-3f}, x_datas, y_datas, &jacobian_ns, attributes));

  for (int i = 0; i < jacobian_ts.size(); i++) {
    // Compute the maximum error between theoretical and numeric Jacobians.
    *max_error = 0.0;
    auto jac_t = jacobian_ts[i];
    auto jac_n = jacobian_ns[i];

    for (int r = 0; r < jacobian_ts[i].size(); ++r) {
      auto cur_error = std::fabs(jac_t[r] - jac_n[r]);
      // Treat any NaN as max_error and immediately return.
      // (Note that std::max may ignore NaN arguments.)
      if (std::isnan(cur_error)) {
        *max_error = cur_error;
        return Status::OK();
      }
      *max_error = std::max(*max_error, cur_error);
    }
  }
  return Status::OK();
}

template <typename X_T, typename Y_T, typename JAC_T>
inline Status GradientChecker<X_T, Y_T, JAC_T>::ComputeGradientError(
    std::string& op_name,
    const std::vector<TensorShape>& x_shapes,
    const std::vector<TensorShape>& y_shapes,
    JAC_T* max_error,
    std::vector<std::string> attributes) {
  // Initialize 'x_datas' to random values.
  std::vector<std::vector<X_T>> x_datas(x_shapes.size());
  for (int i = 0; i < x_shapes.size(); i++) {
    // TODO: Consider varying mean and variance
    float scale = 10.f;
    float mean = 0.f;
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();

    std::default_random_engine generator{gsl::narrow_cast<uint32_t>(seed)};
    std::normal_distribution<X_T> distribution{mean, scale};

    x_datas[i].resize(x_shapes[i].Size());
    std::for_each(x_datas[i].begin(), x_datas[i].end(),
                  [&generator, &distribution](X_T& value) { value = distribution(generator); });
  }

  // Generate dummy placeholders with zero for y_datas
  std::vector<std::vector<Y_T>> y_datas(y_shapes.size());
  for (int i = 0; i < y_shapes.size(); i++) {
    y_datas[i].resize(y_shapes[i].Size(), 0);
  }

  // Compute gradient error.
  return ComputeGradientErrorInternal(op_name, x_shapes, y_shapes, &x_datas, &y_datas, max_error, attributes);
}

#define INSTANTIATE_GRAD_ERR_TYPE(X_T, Y_T, JAC_T) \
  template class GradientChecker<X_T, Y_T, JAC_T>;

INSTANTIATE_GRAD_ERR_TYPE(float, float, float);
INSTANTIATE_GRAD_ERR_TYPE(double, double, double);
}  // namespace test
}  // namespace onnxruntime
