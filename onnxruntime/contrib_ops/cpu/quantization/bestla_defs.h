/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

--*/

#pragma once

#include "contrib_ops/cpu/quantization/bestla_wrapper.h"

namespace bestla {

using tAVX512F = gemm::SCoreRowNAvx512f<48, 8>;
using tAMX_BF16 = gemm::HCoreRowNAmxbf16<64, 16>;
using tAVX512_FP16 = gemm::HCoreRowNAvx512fp16<96, 8>;
using tAVX_VNNI = gemm::ICoreRowNAvxvnni<24, 4>;
using tAVX512_VNNI = gemm::ICoreRowNAvx512vnni<48, 8>;
using tAMX_INT8_US = gemm::ICoreRowNAmxint8<64, 16>;
using tAMX_INT8_SS = gemm::ICoreRowNAmxint8SS<64, 16>;
using tAVX2 = gemm::SCoreRowNAvx2<24, 4>;
using tAVX_VNNI_KBlock = gemm::ICoreRowNAvxvnniKBlock<24, 2>;
using tAVX512_VNNI_KBlock = gemm::ICoreRowNAvx512vnniKBlock<48, 4>;
using tAMX_INT8_US_KBlock = gemm::ICoreRowNAmxint8KBlock<48, 16>;
using tAMX_INT8_SS_KBlock = gemm::ICoreRowNAmxint8SSKBlock<48, 16>;

template <class GC_T, BTLA_ISA ISA_T>
using tWeiNInt = prologue_b::gemm::WeightKBlockNInteger<GC_T, ISA_T>;
template <class GC_T, BTLA_ISA ISA_T>
using tWeiNFloat = prologue_b::gemm::WeightKBlockNFloat<GC_T, ISA_T>;

class ORTThreading : public parallel::IThreading {
 public:
  explicit ORTThreading(void* tp);
  void parallel_for(const parallel::thread_func& func) const override;
  void set_threads(int nthreads) override {
    (void)(nthreads);
    assert(0);
  }
  void sync() const override { assert(0); }
  void* mTp;
};

class Platform {
 public:
  Platform();
  static Platform* get();
};

}  // namespace bestla
