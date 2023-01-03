#include <pybind11/pybind11.h>

#include "python/tools/kernel_explorer/device_array.h"
#include "python/tools/kernel_explorer/kernel_explorer_interface.h"

#include "ck/tensor_operation/gpu/device/tensor_layout.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/tensor_specialization.hpp"

namespace py = pybind11;

using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;

struct AddGelu
{
    __host__ __device__ void
    operator()(ck::half_t& e, const ck::half_t& c, const ck::half_t& d) const
    {
        const ck::half_t x = c + d;

        ck::tensor_operation::element_wise::Gelu{}.template operator()<ck::half_t, ck::half_t>(e, x);
    }

    __host__ __device__ void
    operator()(float& e, const float& c, const ck::half_t& d) const
    {
        const float x = c + d;

        ck::tensor_operation::element_wise::Gelu{}.template operator()<float, float>(e, x);
    }
};

namespace onnxruntime {

template <typename T>
class BatchedGemmBiasGeluGemm : public IKernelExplorer {
 public:
  BatchedGemmBiasGeluGemm(DeviceArray& a,
                          DeviceArray& b,
                          DeviceArray& bias,
                          DeviceArray& c,
                          DeviceArray& out,
                          int M,
                          int K,
                          int N,
                          int O,
                          int batch)
      : a(a.ptr()),
        b(b.ptr()),
        bias(bias.ptr()),
        c(c.ptr()),
        out(out.ptr()),
        M(M),
        K(K),
        N(N),
        O(O),
        batch(batch) {}

  void Run() override {
    ORT_ENFORCE((std::is_same_v<T, half>));
    using F16 = ck::half_t;
    ck::tensor_operation::device::DeviceBatchedGemmMultipleDGemmMultipleD_Xdl_CShuffle<
        Row,
        Row,
        ck::Tuple<Col>,
        Row,
        ck::Tuple<>,
        Row,
        F16,
        F16,
        float,
        ck::Tuple<F16>,
        F16,
        float,
        F16,
        ck::Tuple<>,
        F16,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::Add,
        ck::tensor_operation::element_wise::PassThrough,
        ck::tensor_operation::element_wise::PassThrough,
        false,
        false,
        false,
        false,
        false,
         1,
        256,
        128,         // MPerBlock
        128,         // NPerBlock
        32,          // KPerBlock
        128,         // Gemm1NPerBlock
        32,          // Gemm1KPerBlock
        8,           // AK1
        8,           // BK1
        2,           // B1K1
        32,          // MPerXDL
        32,          // NPerXDL
        1,           // MXdlPerWave
        4,           // NXdlPerWave
        4,           // Gemm1NXdlPerWave
        ck::Sequence<4, 64, 1>, // ABlockTransfer
        ck::Sequence<1, 0, 2>,
        ck::Sequence<1, 0, 2>,
        2,
        8,
        8,
        true,
        ck::Sequence<4, 64, 1>, // BBlockTransfer
        ck::Sequence<1, 0, 2>,
        ck::Sequence<1, 0, 2>,
        2,
        8,
        8,
        true,
        ck::Sequence<8, 32, 1>, // B1BlockTransfer
        ck::Sequence<0, 2, 1>,
        ck::Sequence<0, 2, 1>,
        1,
        4,
        2,
        false,
        1,              // CShuffleMXdlPerWavePerShuffle
        2,              // CShuffleNXdlPerWavePerShuffle
        ck::Sequence<1, 32, 1, 8>, // CShuffleBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock
        8               // CShuffleBlockTransferScalarPerVector_NPerBlock
        > batched_gemm_bias_gelu_gemm;

    auto invoker = batched_gemm_bias_gelu_gemm.MakeInvoker();
    auto argument = batched_gemm_bias_gelu_gemm.MakeArgument(
        (const ck::half_t*)a,
        (const ck::half_t*)b,
        std::array<const void*, 1>{bias},
        (const ck::half_t*)c,
        {},
        (ck::half_t*)out,
        M, N, K, O, batch,
        K, N, {0}, O, {}, O,
        M*K, 0, {0}, 0, {}, M*O,
        ck::tensor_operation::element_wise::PassThrough{},
        ck::tensor_operation::element_wise::PassThrough{},
        // AddGelu{},
        ck::tensor_operation::element_wise::Add{},
        ck::tensor_operation::element_wise::PassThrough{},
        ck::tensor_operation::element_wise::PassThrough{}
        );

    ORT_ENFORCE(batched_gemm_bias_gelu_gemm.IsSupportedArgument(argument));

    invoker.Run(argument, StreamConfig{nullptr, false});
  }

 private:
  void* a;   // BxMxK
  void* b;   // BxKxN
  void* bias; // N
  void* c;   // BxNxO
  void* out; // BxMxO
  int M;
  int K;
  int N;
  int O;
  int batch;
};

void InitBatchedGemmBiasGeluGemm(py::module m) {
  py::class_<BatchedGemmBiasGeluGemm<half>>(m, "BatchedGemmBiasGeluGemm_half")
      .def(py::init<DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&, DeviceArray&, int, int, int, int, int>())
      .def("SetRepeats", &BatchedGemmBiasGeluGemm<half>::SetRepeats)
      .def("Profile", &BatchedGemmBiasGeluGemm<half>::Profile)
      .def("Run", &BatchedGemmBiasGeluGemm<half>::Run);
}

}  // namespace onnxruntime
