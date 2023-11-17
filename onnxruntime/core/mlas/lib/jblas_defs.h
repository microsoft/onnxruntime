/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

--*/

#pragma once

#include "jblas/jit_blas_prologue_b.h"
#include "jblas/jit_blas_wrapper.h"

namespace jblas
{
template <class GemmCore_T>
using tLauncher_Fp32_S4_F32F32 = jblas::wrapper::gemm::LauncherKBlock<
    GemmCore_T::ISA,
    GemmCore_T,
    jblas::prologue_a::gemm::ActivationKBlockBaseF32,
    jblas::prologue_b::gemm::WeightKBlockS4,
    jblas::epilogue::gemm::CompFp32BlockEpilogue,
    jblas::epilogue::gemm::AccumulatorWriteBackFp32>;

template <class GemmCore_T>
using tLauncher_Int8_S4_F32F32 = jblas::wrapper::gemm::LauncherKBlock<
    GemmCore_T::ISA,
    GemmCore_T,
    jblas::prologue_a::gemm::ActivationF32KBlockQuantize,
    jblas::prologue_b::gemm::WeightKBlockS4,
    jblas::epilogue::gemm::CompInt8BlockEpilogue,
    jblas::epilogue::gemm::AccumulatorWriteBackFp32>;

using tAVX512F = jblas::gemm::SCoreRowNAvx512f<48, 8>;
using tAMX_BF16 = jblas::gemm::HCoreRowNAmxbf16<64, 16>;
using tAVX512_FP16 = jblas::gemm::HCoreRowNAvx512fp16<96, 8>;
using tAVX_VNNI = jblas::gemm::ICoreRowNAvxvnni<48, 2>;
using tAVX512_VNNI = jblas::gemm::ICoreRowNAvx512vnni<48, 8>;
using tAMX_INT8_US = jblas::gemm::ICoreRowNAmxint8<64, 16>;
using tAMX_INT8_SS = jblas::gemm::ICoreRowNAmxint8SS<64, 16>;
using tAVX2 = jblas::gemm::SCoreRowNAvx2<48, 2>;

class ORTThreading : public jblas::parallel::IThreading
{
   public:
    ORTThreading(void* tp);
    void parallel_for(const jblas::parallel::thread_func& func) override;
    virtual void set_threads(int nthreads) override { assert(0); }
    void* mTp;
};

}  // namespace jblas
