#include <pybind11/pybind11.h>

#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_contraction_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"

namespace py = pybind11;

namespace onnxruntime {

template <typename T>
class BatchedGemmSoftmaxGemmPermute : public IKernelExplorer {
 public:
  BatchedGemmSoftmaxGemmPermute(DeviceArray& q,
                                DeviceArray& k,
                                DeviceArray& v,
                                DeviceArray& out,
                                int batchsize,
                                int seq_len,
                                int num_heads,
                                int head_dim,
                                float scale)
      : q(q.ptr()),
        k(k.ptr()),
        v(v.ptr()),
        out(out.ptr()),
        G0(batchsize),
        G1(num_heads),
        M(seq_len),
        N(seq_len),
        K(head_dim),
        O(head_dim),
        scale(scale) {}

  void Run() override {
    ORT_ENFORCE((std::is_same_v<T, half>));

    ck::tensor_operation::device::DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<
        2, 1, 1, 1, 1,
        ck::half_t,
        ck::half_t,
        ck::half_t,
        ck::half_t,
        ck::Tuple<>,  // ck::Tuple<ck::half_t>, // acc0 bias datatype
        ck::Tuple<>,
        float,
        ck::half_t,  // CShuffleDType,

        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::ScaleAndResetNaNToMinusInfinity,

        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,

        ck::tensor_operation::device::GemmSpecialization::Default,

        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default,
        ck::tensor_operation::device::TensorSpecialization::Default,

        1,

        256,  // block_size
        128,  // m_per_block
        256,  // n_per_block
        32,   // k_per_block
        64,   // Gemm1NPerBlock
        32,   // Gemm1KPerBlock
        8,    // ak1
        8,    // bk1
        2,    // b1k1
        32,   // m_per_xdl
        32,   // n_per_xdl
        1,    // m_xdl_per_wave
        8,    // n_xdl_per_wave
        2,    // Gemm1NXdlPerWave

        ck::Sequence<4, 64, 1>,  // thread_cluster_length
        ck::Sequence<1, 0, 2>,   // thread_cluster_arrange_order
        ck::Sequence<1, 0, 2>,   // src_access_order
        2,                       // src_vector_dim
        8,                       // src_scalar_per_vector
        8,                       // dst_scalar_per_vector

        1,  // add_extra_dim

        ck::Sequence<4, 64, 1>,  // thread_cluster_length
        ck::Sequence<1, 0, 2>,   // thread_cluster_arrange_order
        ck::Sequence<1, 0, 2>,   // src_access_order
        2,                       // src_vector_dim
        8,                       // src_scalar_per_vector
        8,                       // dst_scalar_per_vector

        1,  // add_extra_dim

        ck::Sequence<16, 16, 1>,  // thread_cluster_length
        ck::Sequence<0, 2, 1>,    // thread_cluster_arrange_order
        ck::Sequence<0, 2, 1>,    // src_access_order
        1,                        // src_vector_dim
        4,                        // src_scalar_per_vector
        2,                        // dst_scalar_per_vector

        0,  // add_extra_dim

        1,                                                                 // m_xdl_per_wave
        2,                                                                 // n_xdl_per_wave
        ck::Sequence<1, 32, 1, 8>,                                         // m_n_block_wave_per_xdl
        8,                                                                 // scalar_per_vector
        ck::tensor_operation::device::MaskingSpecialization::MaskDisabled  // causal_mask

        >
        batched_gemm_softmax_gemm_permute;

    auto invoker = batched_gemm_softmax_gemm_permute.MakeInvoker();
    auto argument = batched_gemm_softmax_gemm_permute.MakeArgument(
        (const ck::half_t*)q,
        (const ck::half_t*)k,
        (const ck::half_t*)v,
        (ck::half_t*)out,
        {},
        {},
        {G0, G1, M, K},
        {G1 * M * K, M * K, K, 1},
        {G0, G1, N, K},
        {G1 * N * K, N * K, K, 1},
        {G0, G1, O, N},
        {G1 * N * O, N * O, 1, O},
        {G0, G1, M, O},
        {M * G1 * O, O, G1 * O, 1},
        {},
        {},
        {},
        {},
        ck::tensor_operation::element_wise::PassThrough{},
        ck::tensor_operation::element_wise::PassThrough{},
        ck::tensor_operation::element_wise::ScaleAndResetNaNToMinusInfinity{scale},
        ck::tensor_operation::element_wise::PassThrough{},
        ck::tensor_operation::element_wise::PassThrough{});

    ORT_ENFORCE(batched_gemm_softmax_gemm_permute.IsSupportedArgument(argument));

    invoker.Run(argument, StreamConfig{nullptr, false});
  }

 private:
  void* q;
  void* k;
  void* v;
  void* out;
  int G0;
  int G1;
  int M;
  int N;
  int K;
  int O;
  float scale;
};

void InitBatchedGemmSoftmaxGemmPermute(py::module m) {
  py::class_<BatchedGemmSoftmaxGemmPermute<half>>(m, "BatchedGemmSoftmaxGemmPermute_half")
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&, int, int, int, int, float>())
      .def("SetRepeats", &BatchedGemmSoftmaxGemmPermute<half>::SetRepeats)
      .def("Profile", &BatchedGemmSoftmaxGemmPermute<half>::Profile)
      .def("Run", &BatchedGemmSoftmaxGemmPermute<half>::Run);
}

}  // namespace onnxruntime
