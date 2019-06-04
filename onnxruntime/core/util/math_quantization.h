#include "math_cpuonly.h"

namespace onnxruntime {

class QuantizeConfig {
 public:
  QuantizeConfig(int type_bits, int sign_bit, int reserve_bits, bool is_symmetric)
      : type_bits_(type_bits),
        sign_bit_(sign_bit),
        reserve_bits_(reserve_bits),
        is_symmetric_(is_symmetric) {
    ORT_ENFORCE(type_bits_ == 16 || type_bits_ == 8);
    ORT_ENFORCE(sign_bit_ == 0 || sign_bit_ == 1);
    ORT_ENFORCE(type_bits_ > reserve_bits_);
    ORT_ENFORCE(type_bits_ - reserve_bits_ > sign_bit_);
  }

  float MaxValue() const {
    return static_cast<float>((1 << (UsableBits() - SignBit())) - 1);
  }

  float MinValue() const {
    return SignBit() ? -(MaxValue() + static_cast<float>(1)) : static_cast<float>(0);
  }

  float Range() const {
    // NOTE that ModelCompiler uses 2^b instead of 2^b - 1 for full range
    // this will not be able to recover max_input which is quantized to 2^b - 1

    if (is_symmetric_) {
      return MaxValue() + 0.5f;
    } else {
      return static_cast<float>(1 << UsableBits());
    }
    // return SignBit() ? (MaxValue() + 0.5f) : static_cast<float>(1 << UsableBits());
    // return static_cast<float>((1 << UsableBits()) - 1);
  }

  int UsableBits() const {
    return type_bits_ - reserve_bits_;
  }

  int SignBit() const {
    return sign_bit_;
  }

  bool IsSymmetric() const {
    return is_symmetric_;
  }

 private:
  int type_bits_;
  int sign_bit_;
  int reserve_bits_;
  bool is_symmetric_;
};

// returns quantized matrix, base, step
inline std::tuple<Eigen::MatrixXi, Eigen::MatrixXf, Eigen::MatrixXf>
QuantizeAsymmetrically(const QuantizeConfig& qcfg, const float* data, int rows, int cols) {
  Eigen::MatrixXf input = ConstEigenMatrixMap<float>(data, rows, cols);
  auto max_input = input.colwise().maxCoeff();
  auto min_input = input.colwise().minCoeff();
  auto span = max_input - min_input;
  auto q_range = qcfg.Range();

  Eigen::MatrixXf step = (span / q_range);
  Eigen::MatrixXf base;
  if (qcfg.SignBit())
    base = (max_input + min_input + step) * 0.5f;
  else
    base = min_input;

  // normalize input value range to [0,1]
  auto normalized_input =
      (input - base.replicate(rows, 1))
          .cwiseProduct(span.cwiseInverse().replicate(rows, 1));

  Eigen::MatrixXi Q_i = (normalized_input.array() * q_range + 0.5f)
                            .floor()
                            .max(qcfg.MinValue())
                            .min(qcfg.MaxValue())
                            .matrix()
                            .template cast<int32_t>();
  return std::make_tuple(Q_i, base, step);
}

// returns quantized matrix, step
inline std::tuple<Eigen::MatrixXi, Eigen::MatrixXf>
QuantizeSymmetrically(const QuantizeConfig& qcfg, const float* data, int rows, int cols) {
  ORT_ENFORCE(qcfg.SignBit());

  Eigen::MatrixXf input = ConstEigenMatrixMap<float>(data, rows, cols);
  auto absmax_input = input.cwiseAbs().colwise().maxCoeff();

  float q_max_value = qcfg.MaxValue();
  int32_t q_max_value_int = static_cast<int32_t>(q_max_value);
  float q_max_range = q_max_value + 0.5f;
  int32_t neg_q_max_value_plus_one = -(q_max_value_int + 1);
  Eigen::MatrixXf step = (absmax_input / q_max_range).template cast<float>();

  auto Q_n = input.cwiseQuotient(absmax_input.replicate(rows, 1)).array();
  auto Q_n_cast = (Q_n * q_max_range + q_max_range + 1).template cast<int32_t>();

  Eigen::MatrixXi Q_i = (Q_n_cast +
                         neg_q_max_value_plus_one)
                            .max(neg_q_max_value_plus_one)
                            .min(q_max_value_int)
                            .matrix().template cast<int32_t>();

  return std::make_tuple(Q_i, step);
}

}  // namespace onnxruntime
