// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/op_kernel.h"
#include "core/mlas/inc/mlas.h"
#include "core/providers/common.h"

#ifdef USE_FBGEMM
#define FBGEMM_STATIC
#include <vector>
#include "fbgemm/Fbgemm.h"
#include "fbgemm/QuantUtils.h"
using namespace fbgemm;
#endif // USE_FBGEMM

namespace onnxruntime {

#ifdef USE_FBGEMM
// This function computes the offset values for each column which are used for compensating the remainders of quantized values
// More detailed math is avilable in the FBGEMM's blog - https://engineering.fb.com/ml-applications/fbgemm/
inline void colOffsetsWithoutZeroPtS8acc32(
    bool transpose,
    int K,
    int N,
    const int8_t* Bint8,
    int32_t* col_offsets) {
  for (int n = 0; n < N; ++n) {
    int32_t sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += transpose ? Bint8[k + n * K] : Bint8[k * N + n];
    }
    col_offsets[n] = sum;
  }
}
#endif // USE_FBGEMM

class MatMulIntegerBase : public OpKernel {
 public:
  MatMulIntegerBase(const OpKernelInfo& info) : OpKernel(info) {}

#if defined(MLAS_SUPPORTS_PACKED_GEMM_U8X8) || defined(USE_FBGEMM)
  Status PrePack(const Tensor& tensor, int input_idx, bool& is_packed) override {
    is_packed = false;

    // only pack Matrix B
    if (input_idx == 1) {
      // Only handle the common case of a 2D weight matrix. Additional matrices
      // could be handled by stacking the packed buffers.
      b_shape_ = tensor.Shape();
      if (b_shape_.NumDimensions() != 2) {
        return Status::OK();
      }

      const size_t K = static_cast<size_t>(b_shape_[0]);
      const size_t N = static_cast<size_t>(b_shape_[1]);

      const auto* b_data = static_cast<const uint8_t*>(tensor.DataRaw());
      b_is_signed_ = tensor.IsDataType<int8_t>();

#ifndef USE_FBGEMM
      const size_t packed_b_size = MlasGemmPackBSize(N, K, b_is_signed_);
      if (packed_b_size == 0) {
        return Status::OK();
      }

      auto alloc = Info().GetAllocator(0, OrtMemTypeDefault);
      auto* packed_b_data = alloc->Alloc(packed_b_size);
      packed_b_ = BufferUniquePtr(packed_b_data, BufferDeleter(alloc));
      MlasGemmPackB(N, K, b_data, N, b_is_signed_, packed_b_data);
#else // USE_FBGEMM
      size_t packed_weights_size = fbgemm::PackMatrix<fbgemm::PackBMatrix<int8_t>, int8_t>::packedBufferSize(K, N);
      if (packed_weights_size == 0) {
        return Status::OK();
      }

      // Allocate memory for packed matrix
      auto alloc = Info().GetAllocator(0, OrtMemTypeDefault);
      auto* packed_weights_data = static_cast<int8_t*>(alloc->Alloc(packed_weights_size));
      packed_b_ = BufferUniquePtr(packed_weights_data, BufferDeleter(alloc));

      // fbgemm packed B class
      std::unique_ptr<fbgemm::PackBMatrix<int8_t>> packedB(new fbgemm::PackBMatrix<int8_t>(
          fbgemm::matrix_op_t::NoTranspose, K, N, (int8_t*)b_data, N, packed_weights_data, 1));
      packed_weight_class_ = std::move(packedB);

      // Column offsets
      auto* col_offset_data = static_cast<int32_t*>(alloc->Alloc(N*sizeof(int32_t)));
      weight_col_offsets_ = BufferUniquePtr(col_offset_data, BufferDeleter(alloc));

      colOffsetsWithoutZeroPtS8acc32(
          false,
          K,
          N,
          (int8_t*)b_data,
          (int32_t*)col_offset_data);
#endif // USE_FBGEMM
      is_packed = true;
    }
    return Status::OK();
  }
#endif

 protected:
  bool b_is_signed_;
  TensorShape b_shape_;
  BufferUniquePtr packed_b_;
#ifdef USE_FBGEMM
  std::unique_ptr<fbgemm::PackBMatrix<int8_t>> packed_weight_class_;
  BufferUniquePtr weight_col_offsets_;
  mutable bool zero_offset_applied_ = false;
#endif // USE_FBGEMM
};

}  // namespace onnxruntime
