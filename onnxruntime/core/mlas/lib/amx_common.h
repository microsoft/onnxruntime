/*++

Copyright (c) Microsoft Corporation.  All rights reserved.

Licensed under the MIT License.

Module Name:

    amx_common.h

Abstract:

    Intrinsic and inline functions for amx processing.

--*/

#pragma once

#include "mlasi.h"

#ifdef WIN32
#define tile_dpbssd(dst, src1, src2) _tile_dpbssd(dst, src1, src2)

#define tile_dpbsud(dst, src1, src2) _tile_dpbsud(dst, src1, src2)

#define tile_dpbusd(dst, src1, src2) _tile_dpbusd(dst, src1, src2)

#define tile_dpbuud(dst, src1, src2) _tile_dpbuud(dst, src1, src2)

#define tile_loadd(dst, base, stride) _tile_loadd(dst, base, stride)

#define tile_stream_loadd(dst, base, stride) _tile_stream_loadd(dst, base, stride)

#define tile_stored(dst, base, stride) _tile_stored(dst, base, stride)

void
tile_loadconfig(const void* __config)
{
    _tile_loadconfig(__config);
}
#else

#define tile_int8_dp_internal(name,dst,src1,src2)					\
  __asm__ volatile							\
  ("{"#name"\t%%tmm"#src2", %%tmm"#src1", %%tmm"#dst"|"#name"\t%%tmm"#dst", %%tmm"#src1", %%tmm"#src2"}" ::)

#define tile_dpbssd(dst,src1,src2)					\
  tile_int8_dp_internal (tdpbssd, dst, src1, src2)

#define tile_dpbsud(dst,src1,src2)					\
  tile_int8_dp_internal (tdpbsud, dst, src1, src2)

#define tile_dpbusd(dst,src1,src2)					\
  tile_int8_dp_internal (tdpbusd, dst, src1, src2)

#define tile_dpbuud(dst,src1,src2)					\
  tile_int8_dp_internal (tdpbuud, dst, src1, src2)

#define tile_loadd(dst,base,stride)		\
  tile_loadd_internal (dst, base, stride)

#define tile_loadd_internal(dst,base,stride)				\
  __asm__ volatile							\
  ("{tileloadd\t(%0,%1,1), %%tmm"#dst"|tileloadd\t%%tmm"#dst", [%0+%1*1]}" \
   :: "r" ((const void*) (base)), "r" ((long) (stride)))

#define tile_stream_loadd(dst,base,stride)		\
  tile_stream_loadd_internal (dst, base, stride)

#define tile_stream_loadd_internal(dst,base,stride)			\
  __asm__ volatile							\
  ("{tileloaddt1\t(%0,%1,1), %%tmm"#dst"|tileloaddt1\t%%tmm"#dst", [%0+%1*1]}" \
   :: "r" ((const void*) (base)), "r" ((long) (stride)))

#define tile_stored(dst,base,stride)		\
  tile_stored_internal (dst, base, stride)

#define tile_stored_internal(src,base,stride)				\
  __asm__ volatile							\
  ("{tilestored\t%%tmm"#src", (%0,%1,1)|tilestored\t[%0+%1*1], %%tmm"#src"}" \
   :: "r" ((void*) (base)), "r" ((long) (stride)) \
   : "memory")

void tile_loadconfig (const void *__config)
{
  __asm__ volatile ("ldtilecfg\t%X0" :: "m" (*((const void **)__config)));
}
#endif
