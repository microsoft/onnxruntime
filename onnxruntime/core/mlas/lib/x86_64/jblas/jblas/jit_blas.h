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
  JblasAVX512_BF16,
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