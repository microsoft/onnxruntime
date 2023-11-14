//  Copyright (c) 2023 Intel Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
#pragma once
#include "jblas/jit_blas_prologue_b.h"
#include "jblas/jit_blas_wrapper.h"

namespace jblas {
template <class GemmCore_T>
using JBLAS_FP32_S4_F32F32 =
    jblas::wrapper::gemm::LauncherKBlock<GemmCore_T::ISA, GemmCore_T, jblas::prologue_a::gemm::ActivationKBlockBaseF32,
                                         jblas::prologue_b::gemm::WeightKBlockS4,
                                         jblas::epilogue::gemm::CompFp32BlockEpilogue,
                                         jblas::epilogue::gemm::AccumulatorWriteBackFp32>;

template <class GemmCore_T>
using JBLAS_INT8_S4_F32F32 = jblas::wrapper::gemm::LauncherKBlock<
    GemmCore_T::ISA, GemmCore_T, jblas::prologue_a::gemm::ActivationF32KBlockQuantize,
    jblas::prologue_b::gemm::WeightKBlockS4, jblas::epilogue::gemm::CompInt8BlockEpilogue,
    jblas::epilogue::gemm::AccumulatorWriteBackFp32>;

using Jblas_Fp32_AVX512F_F32F32 = JBLAS_FP32_S4_F32F32<jblas::gemm::SCoreRowNAvx512f<48, 8>>;
using Jblas_Fp32_AVX2_F32F32 = JBLAS_FP32_S4_F32F32<jblas::gemm::SCoreRowNAvx2<24, 4>>;
using Jblas_Int8_AVX512F_F32F32 = JBLAS_INT8_S4_F32F32<jblas::gemm::ICoreRowNAvx512vnni<48, 8>>;
using Jblas_Int8_AVX2_F32F32 = JBLAS_INT8_S4_F32F32<jblas::gemm::ICoreRowNAvxvnni<24, 4>>;
static Jblas_Fp32_AVX512F_F32F32 JblasAvx512fS4Fp32Fp32;
static Jblas_Fp32_AVX2_F32F32 JblasAvx2S4Fp32Fp32;
static Jblas_Int8_AVX512F_F32F32 JblasAvx512VnniS4Fp32Fp32;
static Jblas_Int8_AVX2_F32F32 JblasAvxVnniS4Fp32Fp32;

class ORTThreading : public jblas::parallel::IThreading {
 public:
  ORTThreading(void* tp);
  void parallel_for(const jblas::parallel::thread_func& func) override;
  virtual void set_threads(int nthreads) override { assert(0); }
  void* mTp;
};

}  // namespace jblas
