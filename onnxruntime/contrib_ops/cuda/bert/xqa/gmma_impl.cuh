/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include "cuda_hint.cuh"
#include "mha_stdheaders.cuh"
#ifndef __CUDACC__
#include <cuda_runtime.h>
#endif
#include <cuda_fp16.h>
#include <cuda_fp8.h>

namespace gmma {
// cog template. Do code generation with: pip install cogapp; cog -r $filename

// clang-format off
/*[[[cog
import cog
reg_list = lambda beg,end: ", ".join([f"%{i}" for i in range(beg, end)])
acc_placeholder = lambda n: "{%s}" % reg_list(0, n//2)
acc_registers = lambda n: "\n            , ".join([f'"+f"(acc[{i}][0][0]), "+f"(acc[{i}][0][1]), "+f"(acc[{i}][1][0]), "+f"(acc[{i}][1][1])' for i in range(n//8)])
ptx_eol = "\\n"
n_list = [8, 16, 24, 32, 64, 128, 256]
for n in n_list:
    cog.outl(f'''
template<>
__device__ inline void mma_async_shmA<__nv_fp8_e4m3, {n}, false, false>(float(&acc)[{n//8}][2][2], MatDesc::Raw descA,
MatDesc::Raw descB, bool accHasVal)
{{
    if (accHasVal) {{
        asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3{ptx_eol}"
            "{acc_placeholder(n)},{ptx_eol}" // d
            "%{n//2},{ptx_eol}" //a-desc
            "%{n//2+1},{ptx_eol}" //b-desc
            "%{n//2+2}, 1, 1;{ptx_eol}"
            : {acc_registers(n)}
            : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
    }}
    else {{
        asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3{ptx_eol}"
            "{acc_placeholder(n)},{ptx_eol}" // d
            "%{n//2},{ptx_eol}" //a-desc
            "%{n//2+1},{ptx_eol}" //b-desc
            "%{n//2+2}, 1, 1;{ptx_eol}"
            : {acc_registers(n)}
            : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
    }}
}}

template<>
__device__ inline void mma_async_regA<__nv_fp8_e4m3, {n}, false, false>(float(&acc)[{n//8}][2][2], uint32_t
const(&a)[2][2][1], MatDesc::Raw descB, bool accHasVal)
{{
    if (accHasVal) {{
        asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3{ptx_eol}"
            "{acc_placeholder(n)},{ptx_eol}" // d
            "{{{reg_list(n//2, n//2 + 4)}}},{ptx_eol}" //a
            "%{n//2+4},{ptx_eol}" //b-desc
            "%{n//2+5}, 1, 1;{ptx_eol}"
            : {acc_registers(n)}
            : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]), "l"(reinterpret_cast<uint64_t
const&>(descB)), "n"(true));
    }}
    else {{
        asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k32.f32.e4m3.e4m3{ptx_eol}"
            "{acc_placeholder(n)},{ptx_eol}" // d
            "{{{reg_list(n//2, n//2 + 4)}}},{ptx_eol}" //a
            "%{n//2+4},{ptx_eol}" //b-desc
            "%{n//2+5}, 1, 1;{ptx_eol}"
            : {acc_registers(n)}
            : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]), "l"(reinterpret_cast<uint64_t
const&>(descB)), "n"(false));
    }}
}}
''')

for n in n_list:
    for transA in [0, 1]:
        for transB in [0, 1]:
            for t,s in [('half', 'f16'), ('__nv_bfloat16', 'bf16')]:
                cog.outl(f'''
template<>
__device__ inline void mma_async_shmA<{t}, {n}, {transA}, {transB}>(float(&acc)[{n//8}][2][2], MatDesc::Raw descA,
MatDesc::Raw descB, bool accHasVal)
{{
    if (accHasVal) {{
        asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.{s}.{s}{ptx_eol}"
            "{acc_placeholder(n)},{ptx_eol}" // d
            "%{n//2},{ptx_eol}" //a-desc
            "%{n//2+1},{ptx_eol}" //b-desc
            "%{n//2+2}, 1, 1, {transA}, {transB};{ptx_eol}"
            : {acc_registers(n)}
            : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
    }}
    else {{
        asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.{s}.{s}{ptx_eol}"
            "{acc_placeholder(n)},{ptx_eol}" // d
            "%{n//2},{ptx_eol}" //a-desc
            "%{n//2+1},{ptx_eol}" //b-desc
            "%{n//2+2}, 1, 1, {transA}, {transB};{ptx_eol}"
            : {acc_registers(n)}
            : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
    }}
}}
''')
                if transA == 0:
                    cog.outl(f'''
template<>
__device__ inline void mma_async_regA<{t}, {n}, {transA}, {transB}>(float(&acc)[{n//8}][2][2], uint32_t
const(&a)[2][2][1], MatDesc::Raw descB, bool accHasVal)
{{
    if (accHasVal) {{
        asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.{s}.{s}{ptx_eol}"
            "{acc_placeholder(n)},{ptx_eol}" // d
            "{{{reg_list(n//2, n//2 + 4)}}},{ptx_eol}" //a
            "%{n//2+4},{ptx_eol}" //b-desc
            "%{n//2+5}, 1, 1, {transB};{ptx_eol}"
            : {acc_registers(n)}
            : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]), "l"(reinterpret_cast<uint64_t
const&>(descB)), "n"(true));
    }}
    else {{
        asm volatile("wgmma.mma_async.sync.aligned.m64n{n}k16.f32.{s}.{s}{ptx_eol}"
            "{acc_placeholder(n)},{ptx_eol}" // d
            "{{{reg_list(n//2, n//2 + 4)}}},{ptx_eol}" //a
            "%{n//2+4},{ptx_eol}" //b-desc
            "%{n//2+5}, 1, 1, {transB};{ptx_eol}"
            : {acc_registers(n)}
            : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]), "l"(reinterpret_cast<uint64_t
const&>(descB)), "n"(false));
    }}
}}
''')
]]]*/
// clang-format on

template <>
__device__ inline void mma_async_shmA<__nv_fp8_e4m3, 8, false, false>(
    float (&acc)[1][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_fp8_e4m3, 8, false, false>(
    float (&acc)[1][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3},\n"  // d
        "{%4, %5, %6, %7},\n"  // a
        "%8,\n"                // b-desc
        "%9, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3},\n"  // d
        "{%4, %5, %6, %7},\n"  // a
        "%8,\n"                // b-desc
        "%9, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_fp8_e4m3, 16, false, false>(
    float (&acc)[2][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_fp8_e4m3, 16, false, false>(
    float (&acc)[2][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "{%8, %9, %10, %11},\n"                // a
        "%12,\n"                               // b-desc
        "%13, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "{%8, %9, %10, %11},\n"                // a
        "%12,\n"                               // b-desc
        "%13, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_fp8_e4m3, 24, false, false>(
    float (&acc)[3][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_fp8_e4m3, 24, false, false>(
    float (&acc)[3][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "{%12, %13, %14, %15},\n"                                // a
        "%16,\n"                                                 // b-desc
        "%17, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "{%12, %13, %14, %15},\n"                                // a
        "%16,\n"                                                 // b-desc
        "%17, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_fp8_e4m3, 32, false, false>(
    float (&acc)[4][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_fp8_e4m3, 32, false, false>(
    float (&acc)[4][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "{%16, %17, %18, %19},\n"                                                    // a
        "%20,\n"                                                                     // b-desc
        "%21, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "{%16, %17, %18, %19},\n"                                                    // a
        "%20,\n"                                                                     // b-desc
        "%21, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_fp8_e4m3, 64, false, false>(
    float (&acc)[8][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_fp8_e4m3, 64, false, false>(
    float (&acc)[8][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "{%32, %33, %34, %35},\n"                          // a
        "%36,\n"                                           // b-desc
        "%37, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "{%32, %33, %34, %35},\n"                          // a
        "%36,\n"                                           // b-desc
        "%37, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_fp8_e4m3, 128, false, false>(
    float (&acc)[16][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_fp8_e4m3, 128, false, false>(
    float (&acc)[16][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "{%64, %65, %66, %67},\n"                                                                                 // a
        "%68,\n"                                                                                                  // b-desc
        "%69, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "{%64, %65, %66, %67},\n"                                                                                 // a
        "%68,\n"                                                                                                  // b-desc
        "%69, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_fp8_e4m3, 256, false, false>(
    float (&acc)[32][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_fp8_e4m3, 256, false, false>(
    float (&acc)[32][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "{%128, %129, %130, %131},\n"       // a
        "%132,\n"                           // b-desc
        "%133, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k32.f32.e4m3.e4m3\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "{%128, %129, %130, %131},\n"       // a
        "%132,\n"                           // b-desc
        "%133, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 8, 0, 0>(
    float (&acc)[1][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<half, 8, 0, 0>(
    float (&acc)[1][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n"
        "{%0, %1, %2, %3},\n"  // d
        "{%4, %5, %6, %7},\n"  // a
        "%8,\n"                // b-desc
        "%9, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n"
        "{%0, %1, %2, %3},\n"  // d
        "{%4, %5, %6, %7},\n"  // a
        "%8,\n"                // b-desc
        "%9, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 8, 0, 0>(
    float (&acc)[1][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_bfloat16, 8, 0, 0>(
    float (&acc)[1][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3},\n"  // d
        "{%4, %5, %6, %7},\n"  // a
        "%8,\n"                // b-desc
        "%9, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3},\n"  // d
        "{%4, %5, %6, %7},\n"  // a
        "%8,\n"                // b-desc
        "%9, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 8, 0, 1>(
    float (&acc)[1][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<half, 8, 0, 1>(
    float (&acc)[1][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n"
        "{%0, %1, %2, %3},\n"  // d
        "{%4, %5, %6, %7},\n"  // a
        "%8,\n"                // b-desc
        "%9, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n"
        "{%0, %1, %2, %3},\n"  // d
        "{%4, %5, %6, %7},\n"  // a
        "%8,\n"                // b-desc
        "%9, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 8, 0, 1>(
    float (&acc)[1][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_bfloat16, 8, 0, 1>(
    float (&acc)[1][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3},\n"  // d
        "{%4, %5, %6, %7},\n"  // a
        "%8,\n"                // b-desc
        "%9, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3},\n"  // d
        "{%4, %5, %6, %7},\n"  // a
        "%8,\n"                // b-desc
        "%9, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 8, 1, 0>(
    float (&acc)[1][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 8, 1, 0>(
    float (&acc)[1][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 8, 1, 1>(
    float (&acc)[1][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 8, 1, 1>(
    float (&acc)[1][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3},\n"  // d
        "%4,\n"                // a-desc
        "%5,\n"                // b-desc
        "%6, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 16, 0, 0>(
    float (&acc)[2][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<half, 16, 0, 0>(
    float (&acc)[2][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "{%8, %9, %10, %11},\n"                // a
        "%12,\n"                               // b-desc
        "%13, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "{%8, %9, %10, %11},\n"                // a
        "%12,\n"                               // b-desc
        "%13, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 16, 0, 0>(
    float (&acc)[2][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_bfloat16, 16, 0, 0>(
    float (&acc)[2][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "{%8, %9, %10, %11},\n"                // a
        "%12,\n"                               // b-desc
        "%13, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "{%8, %9, %10, %11},\n"                // a
        "%12,\n"                               // b-desc
        "%13, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 16, 0, 1>(
    float (&acc)[2][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<half, 16, 0, 1>(
    float (&acc)[2][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "{%8, %9, %10, %11},\n"                // a
        "%12,\n"                               // b-desc
        "%13, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "{%8, %9, %10, %11},\n"                // a
        "%12,\n"                               // b-desc
        "%13, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 16, 0, 1>(
    float (&acc)[2][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_bfloat16, 16, 0, 1>(
    float (&acc)[2][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "{%8, %9, %10, %11},\n"                // a
        "%12,\n"                               // b-desc
        "%13, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "{%8, %9, %10, %11},\n"                // a
        "%12,\n"                               // b-desc
        "%13, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 16, 1, 0>(
    float (&acc)[2][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 16, 1, 0>(
    float (&acc)[2][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 16, 1, 1>(
    float (&acc)[2][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 16, 1, 1>(
    float (&acc)[2][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7},\n"  // d
        "%8,\n"                                // a-desc
        "%9,\n"                                // b-desc
        "%10, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 24, 0, 0>(
    float (&acc)[3][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<half, 24, 0, 0>(
    float (&acc)[3][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "{%12, %13, %14, %15},\n"                                // a
        "%16,\n"                                                 // b-desc
        "%17, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "{%12, %13, %14, %15},\n"                                // a
        "%16,\n"                                                 // b-desc
        "%17, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 24, 0, 0>(
    float (&acc)[3][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_bfloat16, 24, 0, 0>(
    float (&acc)[3][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "{%12, %13, %14, %15},\n"                                // a
        "%16,\n"                                                 // b-desc
        "%17, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "{%12, %13, %14, %15},\n"                                // a
        "%16,\n"                                                 // b-desc
        "%17, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 24, 0, 1>(
    float (&acc)[3][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<half, 24, 0, 1>(
    float (&acc)[3][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "{%12, %13, %14, %15},\n"                                // a
        "%16,\n"                                                 // b-desc
        "%17, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "{%12, %13, %14, %15},\n"                                // a
        "%16,\n"                                                 // b-desc
        "%17, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 24, 0, 1>(
    float (&acc)[3][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_bfloat16, 24, 0, 1>(
    float (&acc)[3][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "{%12, %13, %14, %15},\n"                                // a
        "%16,\n"                                                 // b-desc
        "%17, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "{%12, %13, %14, %15},\n"                                // a
        "%16,\n"                                                 // b-desc
        "%17, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 24, 1, 0>(
    float (&acc)[3][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 24, 1, 0>(
    float (&acc)[3][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 24, 1, 1>(
    float (&acc)[3][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 24, 1, 1>(
    float (&acc)[3][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11},\n"  // d
        "%12,\n"                                                 // a-desc
        "%13,\n"                                                 // b-desc
        "%14, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 32, 0, 0>(
    float (&acc)[4][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<half, 32, 0, 0>(
    float (&acc)[4][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "{%16, %17, %18, %19},\n"                                                    // a
        "%20,\n"                                                                     // b-desc
        "%21, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "{%16, %17, %18, %19},\n"                                                    // a
        "%20,\n"                                                                     // b-desc
        "%21, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 32, 0, 0>(
    float (&acc)[4][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_bfloat16, 32, 0, 0>(
    float (&acc)[4][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "{%16, %17, %18, %19},\n"                                                    // a
        "%20,\n"                                                                     // b-desc
        "%21, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "{%16, %17, %18, %19},\n"                                                    // a
        "%20,\n"                                                                     // b-desc
        "%21, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 32, 0, 1>(
    float (&acc)[4][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<half, 32, 0, 1>(
    float (&acc)[4][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "{%16, %17, %18, %19},\n"                                                    // a
        "%20,\n"                                                                     // b-desc
        "%21, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "{%16, %17, %18, %19},\n"                                                    // a
        "%20,\n"                                                                     // b-desc
        "%21, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 32, 0, 1>(
    float (&acc)[4][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_bfloat16, 32, 0, 1>(
    float (&acc)[4][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "{%16, %17, %18, %19},\n"                                                    // a
        "%20,\n"                                                                     // b-desc
        "%21, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "{%16, %17, %18, %19},\n"                                                    // a
        "%20,\n"                                                                     // b-desc
        "%21, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 32, 1, 0>(
    float (&acc)[4][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 32, 1, 0>(
    float (&acc)[4][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 32, 1, 1>(
    float (&acc)[4][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 32, 1, 1>(
    float (&acc)[4][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15},\n"  // d
        "%16,\n"                                                                     // a-desc
        "%17,\n"                                                                     // b-desc
        "%18, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 64, 0, 0>(
    float (&acc)[8][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<half, 64, 0, 0>(
    float (&acc)[8][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "{%32, %33, %34, %35},\n"                          // a
        "%36,\n"                                           // b-desc
        "%37, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "{%32, %33, %34, %35},\n"                          // a
        "%36,\n"                                           // b-desc
        "%37, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 64, 0, 0>(
    float (&acc)[8][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_bfloat16, 64, 0, 0>(
    float (&acc)[8][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "{%32, %33, %34, %35},\n"                          // a
        "%36,\n"                                           // b-desc
        "%37, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "{%32, %33, %34, %35},\n"                          // a
        "%36,\n"                                           // b-desc
        "%37, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 64, 0, 1>(
    float (&acc)[8][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<half, 64, 0, 1>(
    float (&acc)[8][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "{%32, %33, %34, %35},\n"                          // a
        "%36,\n"                                           // b-desc
        "%37, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "{%32, %33, %34, %35},\n"                          // a
        "%36,\n"                                           // b-desc
        "%37, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 64, 0, 1>(
    float (&acc)[8][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_bfloat16, 64, 0, 1>(
    float (&acc)[8][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "{%32, %33, %34, %35},\n"                          // a
        "%36,\n"                                           // b-desc
        "%37, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "{%32, %33, %34, %35},\n"                          // a
        "%36,\n"                                           // b-desc
        "%37, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 64, 1, 0>(
    float (&acc)[8][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 64, 1, 0>(
    float (&acc)[8][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 64, 1, 1>(
    float (&acc)[8][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 64, 1, 1>(
    float (&acc)[8][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31},\n"  // d
        "%32,\n"                                           // a-desc
        "%33,\n"                                           // b-desc
        "%34, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 128, 0, 0>(
    float (&acc)[16][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<half, 128, 0, 0>(
    float (&acc)[16][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "{%64, %65, %66, %67},\n"                                                                                 // a
        "%68,\n"                                                                                                  // b-desc
        "%69, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "{%64, %65, %66, %67},\n"                                                                                 // a
        "%68,\n"                                                                                                  // b-desc
        "%69, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 128, 0, 0>(
    float (&acc)[16][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_bfloat16, 128, 0, 0>(
    float (&acc)[16][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "{%64, %65, %66, %67},\n"                                                                                 // a
        "%68,\n"                                                                                                  // b-desc
        "%69, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "{%64, %65, %66, %67},\n"                                                                                 // a
        "%68,\n"                                                                                                  // b-desc
        "%69, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 128, 0, 1>(
    float (&acc)[16][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<half, 128, 0, 1>(
    float (&acc)[16][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "{%64, %65, %66, %67},\n"                                                                                 // a
        "%68,\n"                                                                                                  // b-desc
        "%69, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "{%64, %65, %66, %67},\n"                                                                                 // a
        "%68,\n"                                                                                                  // b-desc
        "%69, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 128, 0, 1>(
    float (&acc)[16][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_bfloat16, 128, 0, 1>(
    float (&acc)[16][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "{%64, %65, %66, %67},\n"                                                                                 // a
        "%68,\n"                                                                                                  // b-desc
        "%69, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "{%64, %65, %66, %67},\n"                                                                                 // a
        "%68,\n"                                                                                                  // b-desc
        "%69, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 128, 1, 0>(
    float (&acc)[16][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 128, 1, 0>(
    float (&acc)[16][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 128, 1, 1>(
    float (&acc)[16][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 128, 1, 1>(
    float (&acc)[16][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63},\n"  // d
        "%64,\n"                                                                                                  // a-desc
        "%65,\n"                                                                                                  // b-desc
        "%66, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 256, 0, 0>(
    float (&acc)[32][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<half, 256, 0, 0>(
    float (&acc)[32][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "{%128, %129, %130, %131},\n"       // a
        "%132,\n"                           // b-desc
        "%133, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "{%128, %129, %130, %131},\n"       // a
        "%132,\n"                           // b-desc
        "%133, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 256, 0, 0>(
    float (&acc)[32][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1, 0, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_bfloat16, 256, 0, 0>(
    float (&acc)[32][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "{%128, %129, %130, %131},\n"       // a
        "%132,\n"                           // b-desc
        "%133, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "{%128, %129, %130, %131},\n"       // a
        "%132,\n"                           // b-desc
        "%133, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 256, 0, 1>(
    float (&acc)[32][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<half, 256, 0, 1>(
    float (&acc)[32][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "{%128, %129, %130, %131},\n"       // a
        "%132,\n"                           // b-desc
        "%133, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "{%128, %129, %130, %131},\n"       // a
        "%132,\n"                           // b-desc
        "%133, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 256, 0, 1>(
    float (&acc)[32][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1, 0, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_regA<__nv_bfloat16, 256, 0, 1>(
    float (&acc)[32][2][2], uint32_t const (&a)[2][2][1], MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "{%128, %129, %130, %131},\n"       // a
        "%132,\n"                           // b-desc
        "%133, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "{%128, %129, %130, %131},\n"       // a
        "%132,\n"                           // b-desc
        "%133, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "r"(a[0][0][0]), "r"(a[0][1][0]), "r"(a[1][0][0]), "r"(a[1][1][0]),
          "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 256, 1, 0>(
    float (&acc)[32][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 256, 1, 0>(
    float (&acc)[32][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1, 1, 0;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<half, 256, 1, 1>(
    float (&acc)[32][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

template <>
__device__ inline void mma_async_shmA<__nv_bfloat16, 256, 1, 1>(
    float (&acc)[32][2][2], MatDesc::Raw descA, MatDesc::Raw descB, bool accHasVal) {
  if (accHasVal) {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(true));
  } else {
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n256k16.f32.bf16.bf16\n"
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, "
        "%23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, "
        "%44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, "
        "%65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, "
        "%86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, "
        "%106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, "
        "%123, %124, %125, %126, %127},\n"  // d
        "%128,\n"                           // a-desc
        "%129,\n"                           // b-desc
        "%130, 1, 1, 1, 1;\n"
        : "+f"(acc[0][0][0]), "+f"(acc[0][0][1]), "+f"(acc[0][1][0]), "+f"(acc[0][1][1]), "+f"(acc[1][0][0]),
          "+f"(acc[1][0][1]), "+f"(acc[1][1][0]), "+f"(acc[1][1][1]), "+f"(acc[2][0][0]), "+f"(acc[2][0][1]),
          "+f"(acc[2][1][0]), "+f"(acc[2][1][1]), "+f"(acc[3][0][0]), "+f"(acc[3][0][1]), "+f"(acc[3][1][0]),
          "+f"(acc[3][1][1]), "+f"(acc[4][0][0]), "+f"(acc[4][0][1]), "+f"(acc[4][1][0]), "+f"(acc[4][1][1]),
          "+f"(acc[5][0][0]), "+f"(acc[5][0][1]), "+f"(acc[5][1][0]), "+f"(acc[5][1][1]), "+f"(acc[6][0][0]),
          "+f"(acc[6][0][1]), "+f"(acc[6][1][0]), "+f"(acc[6][1][1]), "+f"(acc[7][0][0]), "+f"(acc[7][0][1]),
          "+f"(acc[7][1][0]), "+f"(acc[7][1][1]), "+f"(acc[8][0][0]), "+f"(acc[8][0][1]), "+f"(acc[8][1][0]),
          "+f"(acc[8][1][1]), "+f"(acc[9][0][0]), "+f"(acc[9][0][1]), "+f"(acc[9][1][0]), "+f"(acc[9][1][1]),
          "+f"(acc[10][0][0]), "+f"(acc[10][0][1]), "+f"(acc[10][1][0]), "+f"(acc[10][1][1]), "+f"(acc[11][0][0]),
          "+f"(acc[11][0][1]), "+f"(acc[11][1][0]), "+f"(acc[11][1][1]), "+f"(acc[12][0][0]), "+f"(acc[12][0][1]),
          "+f"(acc[12][1][0]), "+f"(acc[12][1][1]), "+f"(acc[13][0][0]), "+f"(acc[13][0][1]), "+f"(acc[13][1][0]),
          "+f"(acc[13][1][1]), "+f"(acc[14][0][0]), "+f"(acc[14][0][1]), "+f"(acc[14][1][0]), "+f"(acc[14][1][1]),
          "+f"(acc[15][0][0]), "+f"(acc[15][0][1]), "+f"(acc[15][1][0]), "+f"(acc[15][1][1]), "+f"(acc[16][0][0]),
          "+f"(acc[16][0][1]), "+f"(acc[16][1][0]), "+f"(acc[16][1][1]), "+f"(acc[17][0][0]), "+f"(acc[17][0][1]),
          "+f"(acc[17][1][0]), "+f"(acc[17][1][1]), "+f"(acc[18][0][0]), "+f"(acc[18][0][1]), "+f"(acc[18][1][0]),
          "+f"(acc[18][1][1]), "+f"(acc[19][0][0]), "+f"(acc[19][0][1]), "+f"(acc[19][1][0]), "+f"(acc[19][1][1]),
          "+f"(acc[20][0][0]), "+f"(acc[20][0][1]), "+f"(acc[20][1][0]), "+f"(acc[20][1][1]), "+f"(acc[21][0][0]),
          "+f"(acc[21][0][1]), "+f"(acc[21][1][0]), "+f"(acc[21][1][1]), "+f"(acc[22][0][0]), "+f"(acc[22][0][1]),
          "+f"(acc[22][1][0]), "+f"(acc[22][1][1]), "+f"(acc[23][0][0]), "+f"(acc[23][0][1]), "+f"(acc[23][1][0]),
          "+f"(acc[23][1][1]), "+f"(acc[24][0][0]), "+f"(acc[24][0][1]), "+f"(acc[24][1][0]), "+f"(acc[24][1][1]),
          "+f"(acc[25][0][0]), "+f"(acc[25][0][1]), "+f"(acc[25][1][0]), "+f"(acc[25][1][1]), "+f"(acc[26][0][0]),
          "+f"(acc[26][0][1]), "+f"(acc[26][1][0]), "+f"(acc[26][1][1]), "+f"(acc[27][0][0]), "+f"(acc[27][0][1]),
          "+f"(acc[27][1][0]), "+f"(acc[27][1][1]), "+f"(acc[28][0][0]), "+f"(acc[28][0][1]), "+f"(acc[28][1][0]),
          "+f"(acc[28][1][1]), "+f"(acc[29][0][0]), "+f"(acc[29][0][1]), "+f"(acc[29][1][0]), "+f"(acc[29][1][1]),
          "+f"(acc[30][0][0]), "+f"(acc[30][0][1]), "+f"(acc[30][1][0]), "+f"(acc[30][1][1]), "+f"(acc[31][0][0]),
          "+f"(acc[31][0][1]), "+f"(acc[31][1][0]), "+f"(acc[31][1][1])
        : "l"(reinterpret_cast<uint64_t const&>(descA)), "l"(reinterpret_cast<uint64_t const&>(descB)), "n"(false));
  }
}

//[[[end]]]
}  // namespace gmma
