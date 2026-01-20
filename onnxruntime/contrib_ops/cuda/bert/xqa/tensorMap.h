#pragma once

#include <cuda.h>

uint32_t getElemBytes(CUtensorMapDataType_enum dataType);

CUtensorMap makeTensorMapForContiguousKVCache(void const* addr, CUtensorMapDataType_enum dataType, uint32_t headElems,
                                              uint32_t nbKHeads, uint32_t maxCacheLen, uint32_t beamWidth, uint32_t batchSize, uint32_t partElems,
                                              uint32_t nbTokens);

CUtensorMap makeTensorMapForPagedKVCache(void const* addr, CUtensorMapDataType_enum dataType, uint32_t headElems,
                                         uint32_t nbKHeads, uint32_t tokensPerPage, uint32_t partElems, uint32_t nbTokensPerTile);
