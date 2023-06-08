/*++

Copyright (c) 2023 Intel Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

	QgemmU8S8KernelAmxCommon.s

Abstract:

    This module implements the common operators for quantized U8S8 AMX GEMM kernel.

    This implementation uses AMX instructions.

--*/

#include "asmmacro.h"
#include "AssembleAmx.h"

        .intel_syntax noprefix

/*++

Routine Description:

	Following four routines implement the tile dot product routines
	used in the Qgemm U8S8 AMX kernel.

Arguments:

    dst (rdi) - Supplies the destination tile id.

    src1 (rsi) - Supplies the first source tile id. 

    src2 (rdx) - Supplies the second source tile id.

Return Value:
	
	None

--*/
	
	.globl  C_UNDERSCORE(MlasTdpAmxT4T2T0)
C_UNDERSCORE(MlasTdpAmxT4T2T0):

	TdpbusdTmmTmmTmm tmm4, tmm2, tmm0	

	ret

	.globl  C_UNDERSCORE(MlasTdpAmxT5T3T0)
C_UNDERSCORE(MlasTdpAmxT5T3T0):

	TdpbusdTmmTmmTmm tmm5, tmm3, tmm0	

	ret

	.globl  C_UNDERSCORE(MlasTdpAmxT6T2T1)
C_UNDERSCORE(MlasTdpAmxT6T2T1):

	TdpbusdTmmTmmTmm tmm6, tmm2, tmm1	

	ret

	.globl  C_UNDERSCORE(MlasTdpAmxT7T3T1)
C_UNDERSCORE(MlasTdpAmxT7T3T1):

	TdpbusdTmmTmmTmm tmm7, tmm3, tmm1	

	ret


/*++

Routine Description:

	Following routines implement the tile load routines
	used in the Qgemm U8S8 AMX kernel.

Arguments:

	dst (rdi) - Supplies the destination tile id.

    	base (rsi) - Supplies the base address of memory location.

    	stride (rdx) - Supplies the stride for tile load.

Return Value:
	
	None

--*/

.globl  C_UNDERSCORE(MlasLoadAmxT0)
C_UNDERSCORE(MlasLoadAmxT0):

	TileloaddTmmMem tmm0, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasLoadAmxT1)
C_UNDERSCORE(MlasLoadAmxT1):

	TileloaddTmmMem tmm1, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasLoadAmxT2)
C_UNDERSCORE(MlasLoadAmxT2):

	TileloaddTmmMem tmm2, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasLoadAmxT3)
C_UNDERSCORE(MlasLoadAmxT3):

	TileloaddTmmMem tmm3, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasLoadAmxT4)
C_UNDERSCORE(MlasLoadAmxT4):

	TileloaddTmmMem tmm4, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasLoadAmxT5)
C_UNDERSCORE(MlasLoadAmxT5):

	TileloaddTmmMem tmm5, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasLoadAmxT6)
C_UNDERSCORE(MlasLoadAmxT6):

	TileloaddTmmMem tmm6, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasLoadAmxT7)
C_UNDERSCORE(MlasLoadAmxT7):

	TileloaddTmmMem tmm7, rsi, rdx

	ret


/*++

Routine Description:

	Following routines implement the tile store routines
	used in the Qgemm U8S8 AMX kernel.

Arguments:

	src (rdi) - Supplies the source tile id.

    	base (rsi) - Supplies the base address of memory location.

    	stride (rdx) - Supplies the stride for tile store.

Return Value:
	
	None

--*/

.globl  C_UNDERSCORE(MlasStoreAmxT0)
C_UNDERSCORE(MlasStoreAmxT0):

	TileStoredMemTmm tmm0, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasStoreAmxT1)
C_UNDERSCORE(MlasStoreAmxT1):

	TileStoredMemTmm tmm1, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasStoreAmxT2)
C_UNDERSCORE(MlasStoreAmxT2):

	TileStoredMemTmm tmm2, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasStoreAmxT3)
C_UNDERSCORE(MlasStoreAmxT3):

	TileStoredMemTmm tmm3, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasStoreAmxT4)
C_UNDERSCORE(MlasStoreAmxT4):

	TileStoredMemTmm tmm4, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasStoreAmxT5)
C_UNDERSCORE(MlasStoreAmxT5):

	TileStoredMemTmm tmm5, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasStoreAmxT6)
C_UNDERSCORE(MlasStoreAmxT6):

	TileStoredMemTmm tmm6, rsi, rdx

	ret

.globl  C_UNDERSCORE(MlasStoreAmxT7)
C_UNDERSCORE(MlasStoreAmxT7):

	TileStoredMemTmm tmm7, rsi, rdx

	ret

/*++

Routine Description:

	Following routine implements the tile config load routines
	used in the Qgemm U8S8 AMX kernel.

Arguments:

	src (rdi) - Supplies the memory address for the tile configuration.

Return Value:
	
	None

--*/

.globl  C_UNDERSCORE(MlasLoadConfigAmx)
C_UNDERSCORE(MlasLoadConfigAmx):

	ldtilecfgMacro rdi

	ret


/*++

Routine Description:

	Following routine implements the tile store config routines
	used in the Qgemm U8S8 AMX kernel.

Arguments:

	src (rdi) - Supplies the memory address for the tile configuration.

Return Value:
	
	None

--*/

.globl  C_UNDERSCORE(MlasStoreConfigAmx)
C_UNDERSCORE(MlasStoreConfigAmx):

	sttilecfgMacro rdi

	ret

	.end

