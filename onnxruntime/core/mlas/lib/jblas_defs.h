/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

--*/

#pragma once

#include "jblas/jit_blas_prologue_b.h"
#include "jblas/jit_blas_wrapper.h"

namespace jblas
{

/*
Name conversion explaination:
Fp32:   comp type, determined by GemmCore, can be any jblas::gemm::SCorexxx(float GemmCore)
S4:     weight dtype, determined by jblas::prologue_b::gemm::WeightKBlockS4(also support other integer and float weight
classes)
F32F32: input/output dtype, determined by jblas::prologue_a::gemm::ActivationKBlockBaseF32 and
jblas::epilogue::gemm::AccumulatorWriteBackFp32.

Tips: jblas::epilogue::gemm::CompFp32BlockEpilogue is a fixed class for all fp32 accumulator GemmCores.
*/
template <class GemmCore_T>
using tLauncher_Fp32_S4_F32F32 = jblas::wrapper::gemm::LauncherKBlock<
    GemmCore_T::ISA,
    GemmCore_T,
    jblas::prologue_a::gemm::ActivationKBlockBaseF32,
    jblas::prologue_b::gemm::WeightKBlockS4,
    jblas::epilogue::gemm::CompFp32BlockEpilogue,
    jblas::epilogue::gemm::AccumulatorWriteBackFp32>;

/*
Name conversion explaination:
Int8:   comp type, determined by GemmCore, can be any jblas::gemm::ICorexxx(interger GemmCore)
S4:     weight dtype, determined by jblas::prologue_b::gemm::WeightKBlockS4(support integer weight classes only)
F32F32: input/output dtype, determined by jblas::prologue_a::gemm::ActivationKBlockBaseF32 and
jblas::epilogue::gemm::AccumulatorWriteBackFp32.

Tips: jblas::epilogue::gemm::CompInt8BlockEpilogue is a fixed class for all int32 accumulator GemmCores.
*/
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
using tAVX_VNNI = jblas::gemm::ICoreRowNAvxvnni<48, 2>;  // TODO(Yu) use 24x4 for higher efficiency
using tAVX512_VNNI = jblas::gemm::ICoreRowNAvx512vnni<48, 8>;
using tAMX_INT8_US = jblas::gemm::ICoreRowNAmxint8<64, 16>;
using tAMX_INT8_SS = jblas::gemm::ICoreRowNAmxint8SS<64, 16>;
using tAVX2 = jblas::gemm::SCoreRowNAvx2<48, 2>;  // TODO(Yu) use 24x4 for higher efficiency

class ORTThreading : public jblas::parallel::IThreading
{
   public:
    ORTThreading(void* tp);
    void parallel_for(const jblas::parallel::thread_func& func) override;
    virtual void set_threads(int nthreads) override { assert(0); }
    virtual void sync() override { assert(0); }
    void* mTp;
};

}  // namespace jblas
