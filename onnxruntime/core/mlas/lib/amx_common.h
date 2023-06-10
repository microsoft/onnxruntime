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

#define tile_loadconfig(config)		\
 _tile_loadconfig(config)

#define tile_storeconfig(config) _tile_storeconfig(config)

#else

asm(".include \"x86_64/QgemmU8S8KernelAmxCommon.S\"");

#define tile_int8_dp_internal1(name,dst,src1,src2)		\
  __asm__ volatile						\
  (""#name" tmm"#dst", tmm"#src1", tmm"#src2"")

#define tile_dpbusd(dst,src1,src2)			\
tile_int8_dp_internal1(TdpbusdTmmTmmTmm, dst, src1, src2)

#define tile_loadd_internal1(dst,base,stride)				\
  __asm__ volatile ("TileloaddTmmMem tmm"#dst", %0, %1" \
   :: "r" ((const void*) (base)), "r" ((long) (stride)))

#define tile_loadd(dst,base,stride)		\
  tile_loadd_internal1(dst, base, stride)


#define tile_stored_internal1(dst,base,stride)				\
  __asm__ volatile ("TileStoredMemTmm tmm"#dst", %0, %1" \
   :: "r" ((const void*) (base)), "r" ((long) (stride)))

#define tile_stored(dst,base,stride)		\
tile_stored_internal1(dst, base, stride)


#define tile_loadconfig(config)		\
__asm__ volatile ("ldtilecfgMacro %0" :: "r" (((const void *)config))) \

#define tile_storeconfig(config)		\
__asm__ volatile ("sttilecfgMacro %0" :: "r" (((const void *)config))) \

#endif
