#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "core/mlas/inc/mlas_q4.h"
#include "contrib_ops/cpu/quantization/dequantize_blockwise_bnb4.h"
#include "core/util/thread_utils.h"
#include "core/framework/float16.h"

// Use the nanobind namespace
namespace nb = nanobind;


namespace nanobind::detail {
template <>
struct dtype_traits<onnxruntime::MLFloat16> {
  static constexpr dlpack::dtype value{
      (uint8_t)dlpack::dtype_code::Float,  // type code
      16,                                  // size in bits
      1                                    // lanes
  };
  static constexpr auto name = const_name("float16");
};
}  // namespace nanobind::detail

namespace onnxruntime {
namespace python {

using namespace onnxruntime;

template <typename T, int qbits>
void QuantizeMatMulNBitsBlockwise(
    nb::ndarray<uint8_t, nb::c_contig> dst,
    nb::ndarray<T, nb::c_contig> src,
    nb::ndarray<T, nb::c_contig> scale,
    nb::ndarray<uint8_t, nb::c_contig> zero_points,
    int32_t block_size,
    int32_t N,
    int32_t K,
    bool is_symmetric) {
  OrtThreadPoolParams to;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to,
                                          concurrency::ThreadPoolType::INTRA_OP);

  MlasQuantizeBlockwise<T, qbits>(
      dst.data(),
      scale.data(),
      is_symmetric ? nullptr : zero_points.data(),
      src.data(),
      block_size,
      true,  // is_col_wise
      K,
      N,
      N,  // lda
      tp.get());
}

template <typename T>
bool QuantizeQDQMatMul4BitsBlockwise(
    nb::ndarray<uint8_t, nb::c_contig> dst,
    nb::ndarray<T, nb::c_contig> src,
    nb::ndarray<T, nb::c_contig> scale,
    nb::ndarray<uint8_t, nb::c_contig> zero_points,
    int32_t quant_block_size,
    int32_t N,
    int32_t K,
    bool is_symmetric) {
  OrtThreadPoolParams to;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to,
                                          concurrency::ThreadPoolType::INTRA_OP);

  return MlasQDQQuantizeBlockwise<T, 4>(
      src.data(),
      scale.data(),
      is_symmetric ? nullptr : zero_points.data(),
      dst.data(),
      true,  // is_col_wise
      K,
      N,
      quant_block_size,
      tp.get());
}

template <typename T>
void QuantizeMatMulBnb4Blockwise(
    nb::ndarray<uint8_t, nb::c_contig> dst,
    nb::ndarray<T, nb::c_contig> src,
    nb::ndarray<T, nb::c_contig> absmax,
    int32_t block_size,
    int32_t quant_type,
    int32_t N,
    int32_t K) {
  OrtThreadPoolParams to;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to,
                                          concurrency::ThreadPoolType::INTRA_OP);

  contrib::QuantizeBlockwiseBnb4<T>(
      dst.data(),
      src.data(),
      absmax.data(),
      block_size,
      quant_type,
      N,
      K,
      tp.get());
}

void CreateQuantPybindModule(nb::module_& m) {
  m.def("quantize_matmul_4bits", &QuantizeMatMulNBitsBlockwise<float, 4>);
  m.def("quantize_matmul_4bits", &QuantizeMatMulNBitsBlockwise<MLFloat16, 4>);
  m.def("quantize_matmul_8bits", &QuantizeMatMulNBitsBlockwise<float, 8>);
  m.def("quantize_matmul_8bits", &QuantizeMatMulNBitsBlockwise<MLFloat16, 8>);
  m.def("quantize_matmul_bnb4", &QuantizeMatMulBnb4Blockwise<float>);
  m.def("quantize_matmul_bnb4", &QuantizeMatMulBnb4Blockwise<MLFloat16>);
  m.def("quantize_qdq_matmul_4bits", &QuantizeQDQMatMul4BitsBlockwise<float>);
  m.def("quantize_qdq_matmul_4bits", &QuantizeQDQMatMul4BitsBlockwise<MLFloat16>);
}

}  // namespace python
}  // namespace onnxruntime