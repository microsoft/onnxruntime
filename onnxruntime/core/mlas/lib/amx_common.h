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

#define tile_loadconfig(config)						\
 _tile_loadconfig(config)

#define tile_storeconfig(config) _tile_storeconfig(config)

#else

#define tile_dpbusd_internal(dst,src1,src2)  \
__asm__ volatile (".set Payload1, 0x01\n\t"    \
	".set Payload1, Payload1 + (("#src2" & 15) ^ 15) << 3\n\t"  \
	".set ModRMByte, 0xC0\n\t" 		\
	".set ModRMByte, ModRMByte + ("#dst" << 3)\n\t"     \
	".set ModRMByte, ModRMByte + ("#src1")\n\t"     \
	".byte 0xC4, 0xE2, Payload1, 0x5E, ModRMByte\n\t")

#define tile_dpbusd(dst,src1,src2)					\
tile_dpbusd_internal(dst,src1,src2)

#define tile_loadd_internal1(dst,base,stride)				\
  __asm__ volatile (".set ModRMByte, 0x04\n\t" 		\
	".set ModRMByte, ModRMByte + ("#dst" << 3)\n\t"     \
	".byte 0xC4, 0xE2, 0x7B, 0x4B, ModRMByte, 0x18\n\t" \
   :: "a" ((const void*) (base)), "b" ((long) (stride)))

#define tile_loadd(dst,base,stride)					\
  tile_loadd_internal1(dst, base, stride)


#define tile_stored_internal1(dst,base,stride)				\
  __asm__ volatile (".set ModRMByte, 0x04\n\t" 		\
	".set ModRMByte, ModRMByte + ("#dst" << 3)\n\t"     \
	".byte 0xC4, 0xE2, 0x7A, 0x4B, ModRMByte, 0x18\n\t" \
   :: "a" ((const void*) (base)), "b" ((long) (stride)))

#define tile_stored(dst,base,stride)					\
tile_stored_internal1(dst, base, stride)


#define tile_loadconfig(config)						\
__asm__ volatile (".byte 0xC4, 0xE2, 0x78, 0x49, 0x00" :: "a" (((const void *)config)))  \

#define tile_storeconfig(config)					\
__asm__ volatile (".byte 0xC4, 0xE2, 0x79, 0x49, 0x00" :: "a" (((const void *)config)))  \

#endif
