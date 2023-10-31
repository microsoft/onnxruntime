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
#include <stdint.h>
enum JBLAS_CODE {
  JblasSuccess = 0,
  JblasInvalidParam = 1,
  JblasInvalidISA = 2,
  JblasRuntimeError = 4,
  JblasNotSupport = 8,
};
enum JBLAS_ISA : uint32_t {
  JblasNoSIMD = 0,
  JblasAVX,
  JblasAVX2,
  JblasAVX_VNNI,
  JblasAVX512F,
  JblasAVX512_VNNI,
  JblasAMX_BF16,
  JblasAMX_INT8,
  JblasAVX512_FP16,
};
enum class JBLAS_DTYPE : uint32_t {
  EleBitsMask = 0xff,
  EleBitsUndef = 0,
  EleBits4 = 4,
  EleBits8 = 8,
  EleBits16 = 16,
  EleBits32 = 32,
  EleBits64 = 64,
  TypeMask = 0xff00,
  TypeFloat = 0 << 8,
  TypeInt = 1 << 8,
  SubTypeMask = 0xff0000,
  SubType0 = 0 << 16,
  SubType1 = 1 << 16,
  SubType2 = 2 << 16,
  F64 = EleBits64 | TypeFloat,
  F32 = EleBits32 | TypeFloat,
  F16 = EleBits16 | TypeFloat,
  BF16 = EleBits16 | TypeFloat | SubType1,
  F8_E4M3 = EleBits8 | TypeFloat,
  F8_E5M2 = EleBits8 | TypeFloat | SubType1,
  F8_E3M4 = EleBits8 | TypeFloat | SubType2,
  S8 = EleBits8 | TypeInt,
  U8 = EleBits8 | TypeInt | SubType1,
  S4_CLIP = EleBits4 | TypeInt,
  S4_FULLRANGE = EleBits4 | TypeInt | SubType1,
  F4_E2M1 = EleBits4 | TypeFloat,
  F4_BNB = EleBits4 | TypeFloat | SubType1,
  F4_NF4 = EleBits4 | TypeFloat | SubType2,
  S32 = EleBits32 | TypeInt,
  U32 = EleBits32 | TypeInt | SubType1,
};

enum JBLAS_LAYOUT { JblasRowMajor = 101, JblasColMajor = 102 };
enum JBLAS_TRANSPOSE {
  JblasNoTrans = 111,
  JblasTrans = 112,
  JblasConjTrans = 113,
};
enum JBLAS_ELTWISEOP {
  GELU,
  SWISH,
  TANH,
  EXP,
  LOW_PRECISION_EXP,
  RELU,
  LINEAR,
};

enum class JBLAS_GEMM_CORE : uint32_t {
  // INT32=LSB|**8bits:NTile**||**8bits:PackRow**||**8bits:CompType**||**8bits:Reserve**|
  Undef = 0,
  NTILE_MASK = 0xff,
  NTILE_SHIFT = 0,
  NTILE_24 = 24,
  NTILE_48 = 48,
  NTILE_64 = 64,
  NTILE_96 = 96,
  PACKROW_MASK = 0xff00,
  PACKROW_SHIFT = 8,
  PACKROW_1 = 1 << PACKROW_SHIFT,
  PACKROW_2 = 2 << PACKROW_SHIFT,
  PACKROW_4 = 4 << PACKROW_SHIFT,
  COMP_MASK = 0xff0000,
  COMP_SHIFT = 16,
  COMP_FP32 = 0 << COMP_SHIFT,
  COMP_BF16 = 1 << COMP_SHIFT,
  COMP_FP16 = 2 << COMP_SHIFT,
  COMP_INT_START = 3 << COMP_SHIFT,
  COMP_INT8_US = COMP_INT_START,
  COMP_INT8_SS = 4 << COMP_SHIFT,
  COMP_INT8_SU = 5 << COMP_SHIFT,
  COMP_INT16_SS = 6 << COMP_SHIFT,
  COMP_INT8_US_FP32 = 7 << COMP_SHIFT,
  COMP_INT8_SS_FP32 = 8 << COMP_SHIFT,
  COMP_INT8_SU_FP32 = 9 << COMP_SHIFT,
  ISA_MASK = 0xff000000,
  ISA_SHIFT = 24,
  ISA_AVX2 = (uint32_t)JBLAS_ISA::JblasAVX2 << ISA_SHIFT,
  ISA_AVX512F = (uint32_t)JBLAS_ISA::JblasAVX512F << ISA_SHIFT,
  ISA_AVX_VNNI = (uint32_t)JBLAS_ISA::JblasAVX_VNNI << ISA_SHIFT,
  ISA_AVX512_VNNI = (uint32_t)JBLAS_ISA::JblasAVX512_VNNI << ISA_SHIFT,
  ISA_AMX_INT8 = (uint32_t)JBLAS_ISA::JblasAMX_INT8 << ISA_SHIFT,
  ISA_AMX_BF16 = (uint32_t)JBLAS_ISA::JblasAMX_BF16 << ISA_SHIFT,
  ISA_AVX512_FP16 = (uint32_t)JBLAS_ISA::JblasAVX512_FP16 << ISA_SHIFT,
  AVX2_4X24 = NTILE_24 | PACKROW_1 | COMP_FP32 | ISA_AVX2,
  AVX2_2X48 = NTILE_48 | PACKROW_1 | COMP_FP32 | ISA_AVX2,
  AVX512F_8x48 = NTILE_48 | PACKROW_1 | COMP_FP32 | ISA_AVX512F,
  AMX_BF16_16x64 = NTILE_64 | PACKROW_2 | COMP_BF16 | ISA_AMX_BF16,
  AMX_BF16_16x48 = NTILE_48 | PACKROW_2 | COMP_BF16 | ISA_AMX_BF16,
  AVX512_FP16_8x64 = NTILE_64 | PACKROW_2 | COMP_FP16 | ISA_AVX512_FP16,
  AVX512_FP16_8x96 = NTILE_96 | PACKROW_2 | COMP_FP16 | ISA_AVX512_FP16,
  AVX_VNNI_2x48 = NTILE_48 | PACKROW_4 | COMP_INT8_US | ISA_AVX_VNNI,
  AVX_VNNI_4x24 = NTILE_24 | PACKROW_4 | COMP_INT8_US | ISA_AVX_VNNI,
  AVX512_VNNI_8x48 = NTILE_48 | PACKROW_4 | COMP_INT8_US | ISA_AVX512_VNNI,
  AMX_INT8_16x64_US = NTILE_64 | PACKROW_4 | COMP_INT8_US | ISA_AMX_INT8,
  AMX_INT8_16x64_SS = NTILE_64 | PACKROW_4 | COMP_INT8_SS | ISA_AMX_INT8,
  AMX_INT8_16x48_US = NTILE_48 | PACKROW_4 | COMP_INT8_US | ISA_AMX_INT8,
  AMX_INT8_16x48_SS = NTILE_48 | PACKROW_4 | COMP_INT8_SS | ISA_AMX_INT8,
};
enum class JBLAS_PROLOGUEB_IDS : uint32_t {
  Undef = (uint32_t)-1,
  Begin = 0,
  NormalBegin = Begin,
  WeightPack = NormalBegin,
  NormalEnd,
  KBlockBegin = NormalEnd,
  WeightKBlockS8 = KBlockBegin,
  WeightKBlockS4,
  WeightKBlockF4,
  KBlockEnd,
  End,
};