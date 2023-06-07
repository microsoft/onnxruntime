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
//#include "./x86_64/asmmacro.h"
//#include "./x86_64/AssembleAvxVnni.h"

#ifdef WIN32
#define tile_dpbssd(dst, src1, src2) _tile_dpbssd(dst, src1, src2)

#define tile_dpbsud(dst, src1, src2) _tile_dpbsud(dst, src1, src2)

#define tile_dpbusd(dst, src1, src2) _tile_dpbusd(dst, src1, src2)

#define tile_dpbuud(dst, src1, src2) _tile_dpbuud(dst, src1, src2)

#define tile_loadd(dst, base, stride) _tile_loadd(dst, base, stride)

#define tile_stream_loadd(dst, base, stride) _tile_stream_loadd(dst, base, stride)

#define tile_stored(dst, base, stride) _tile_stored(dst, base, stride)

#define tile_storeconfig(config) _tile_storeconfig(config)

void
tile_loadconfig(const void* __config)
{
    _tile_loadconfig(__config);
}
#else

extern "C" { 
	void MlasTdpAmx(int dst, int src1, int src2); 
	void MlasTdpAmxT4T2T0(int dst, int src1, int src2); 
	void MlasTdpAmxT5T3T0(int dst, int src1, int src2); 
	void MlasTdpAmxT6T2T1(int dst, int src1, int src2); 
	void MlasTdpAmxT7T3T1(int dst, int src1, int src2); 

	void MlasLoadAmxT0(int dst, const void* base, long stride);
	void MlasLoadAmxT1(int dst, const void* base, long stride);
	void MlasLoadAmxT2(int dst, const void* base, long stride);
	void MlasLoadAmxT3(int dst, const void* base, long stride);
	void MlasLoadAmxT4(int dst, const void* base, long stride);
	void MlasLoadAmxT5(int dst, const void* base, long stride);
	void MlasLoadAmxT6(int dst, const void* base, long stride);
	void MlasLoadAmxT7(int dst, const void* base, long stride);

	void MlasStoreAmxT0(int src, const void* base, long stride);
	void MlasStoreAmxT1(int src, const void* base, long stride);
	void MlasStoreAmxT2(int src, const void* base, long stride);
	void MlasStoreAmxT3(int src, const void* base, long stride);
	void MlasStoreAmxT4(int src, const void* base, long stride);
	void MlasStoreAmxT5(int src, const void* base, long stride);
	void MlasStoreAmxT6(int src, const void* base, long stride);
	void MlasStoreAmxT7(int src, const void* base, long stride);

	void MlasLoadConfigAmx(const void *__config);
	void MlasStoreConfigAmx(void *__config);
}
//MlasTdpAmx(dst,src1,src2)

#define tile_dpbusd_t4t2t0(dst,src1,src2)			\
  MlasTdpAmxT4T2T0(dst,src1,src2) 

#define tile_dpbusd_t5t3t0(dst,src1,src2)			\
  MlasTdpAmxT5T3T0(dst,src1,src2) 

#define tile_dpbusd_t6t2t1(dst,src1,src2)			\
  MlasTdpAmxT6T2T1(dst,src1,src2) 

#define tile_dpbusd_t7t3t1(dst,src1,src2)			\
  MlasTdpAmxT7T3T1(dst,src1,src2) 

#define tile_int8_dp_internal(name,dst,src1,src2)			\
  MlasTdpAmx(dst,src1,src2) 

#define tile_loadd_t0(dst,base,stride)		\
  MlasLoadAmxT0(dst, base, stride)

#define tile_loadd_t1(dst,base,stride)		\
  MlasLoadAmxT1(dst, base, stride)

#define tile_loadd_t2(dst,base,stride)		\
  MlasLoadAmxT2(dst, base, stride)

#define tile_loadd_t3(dst,base,stride)		\
  MlasLoadAmxT3(dst, base, stride)

#define tile_loadd_t4(dst,base,stride)		\
  MlasLoadAmxT4(dst, base, stride)

#define tile_loadd_t5(dst,base,stride)		\
  MlasLoadAmxT5(dst, base, stride)

#define tile_loadd_t6(dst,base,stride)		\
  MlasLoadAmxT6(dst, base, stride)

#define tile_loadd_t7(dst,base,stride)		\
  MlasLoadAmxT7(dst, base, stride)


#define tile_stored_t0(src,base,stride)		\
  MlasStoreAmxT0(src, base, stride)

#define tile_stored_t1(src,base,stride)		\
  MlasStoreAmxT1(src, base, stride)

#define tile_stored_t2(src,base,stride)		\
  MlasStoreAmxT2(src, base, stride)

#define tile_stored_t3(src,base,stride)		\
  MlasStoreAmxT3(src, base, stride)

#define tile_stored_t4(src,base,stride)		\
  MlasStoreAmxT4(src, base, stride)

#define tile_stored_t5(src,base,stride)		\
  MlasStoreAmxT5(src, base, stride)

#define tile_stored_t6(src,base,stride)		\
  MlasStoreAmxT6(src, base, stride)

#define tile_stored_t7(src,base,stride)		\
  MlasStoreAmxT7(src, base, stride)

#define tile_loadconfig(config)		\
  MlasLoadConfigAmx(config)

#define tile_storeconfig(config)		\
  MlasStoreConfigAmx(config)


/*  __asm__ volatile							\
  ("{TdpbusdTmmTmmTmm\ttmm"#src2", tmm"#src1", tmm"#dst"|"#name"\ttmm"#dst", tmm"#src1", tmm"#src2"}" ::)
*/

#define tile_dpbssd(dst,src1,src2)					\
  tile_int8_dp_internal (tdpbssd, dst, src1, src2)

#define tile_dpbsud(dst,src1,src2)					\
  tile_int8_dp_internal (tdpbsud, dst, src1, src2)

#define tile_dpbusd(dst,src1,src2)					\
  tile_int8_dp_internal (TdpbusdTmmTmmTmm, dst, src1, src2)

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

/**
void tile_loadconfig (const void *__config)
{
  __asm__ volatile ("ldtilecfg\t%X0" :: "m" (*((const void **)__config)));
}

void tile_storeconfig (void *__config)
{
  __asm__ volatile ("sttilecfg\t%X0" : "=m" (*((void **)__config)));
}
**/

#endif
