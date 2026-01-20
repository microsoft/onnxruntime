#include "tensorMap.h"
#include "utils.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>

uint32_t getElemBytes(CUtensorMapDataType_enum dataType) {
  switch (dataType) {
    case CU_TENSOR_MAP_DATA_TYPE_UINT8:
      return 1;
    case CU_TENSOR_MAP_DATA_TYPE_UINT16:
      return 2;
    case CU_TENSOR_MAP_DATA_TYPE_UINT32:
      return 4;
    case CU_TENSOR_MAP_DATA_TYPE_INT32:
      return 4;
    case CU_TENSOR_MAP_DATA_TYPE_UINT64:
      return 8;
    case CU_TENSOR_MAP_DATA_TYPE_INT64:
      return 8;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
      return 2;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
      return 4;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT64:
      return 8;
    case CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
      return 2;
    case CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
      return 4;
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
      return 4;
    case CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ:
      return 4;
    default:
      throw std::runtime_error("unsupported data type");
  }
}

CUtensorMap makeTensorMapForContiguousKVCache(void const* addr, CUtensorMapDataType_enum dataType, uint32_t headElems,
                                              uint32_t nbKHeads, uint32_t maxCacheLen, uint32_t beamWidth, uint32_t batchSize, uint32_t partElems,
                                              uint32_t nbTokens) {
  CUtensorMap tensorMap{};
  uint64_t const globalDims[] = {headElems, maxCacheLen, nbKHeads, 2 * beamWidth * batchSize};
  uint32_t elemBytes = getElemBytes(dataType);
  uint32_t const headBytes = elemBytes * headElems;
  uint64_t const globalStrides[] = {headBytes, headBytes * maxCacheLen, headBytes * maxCacheLen * nbKHeads};
  uint32_t const boxDims[] = {partElems, nbTokens, 1, 1};
  uint32_t const elemStrides[] = {1, 1, 1, 1};

  auto const swizzle = [&] {
    switch (partElems) {
      case 128:
        return CU_TENSOR_MAP_SWIZZLE_128B;
      case 64:
        return CU_TENSOR_MAP_SWIZZLE_64B;
      default:
        throw std::runtime_error("unsupported cache head size");
    }
  }();

  checkCu(cuTensorMapEncodeTiled(&tensorMap, dataType, 4, const_cast<void*>(addr), globalDims, globalStrides, boxDims,
                                 elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tensorMap;
}

CUtensorMap makeTensorMapForPagedKVCache(void const* addr, CUtensorMapDataType_enum dataType, uint32_t headElems,
                                         uint32_t nbKHeads, uint32_t tokensPerPage, uint32_t partElems, uint32_t nbTokensPerTile) {
  CUtensorMap tensorMap{};
  uint32_t elemBytes = getElemBytes(dataType);
// VLLM Layout
#if PAGED_KV_CACHE_LAYOUT == 1
  uint64_t const globalDims[] = {headElems, nbKHeads, tokensPerPage, 1U << 31};
  uint32_t const headBytes = elemBytes * headElems;
  uint64_t const globalStrides[] = {headBytes, headBytes * nbKHeads, headBytes * nbKHeads * tokensPerPage};
  uint32_t const partBytes = partElems * elemBytes;
  uint32_t const boxDims[] = {partElems, 1, mha::min(tokensPerPage, nbTokensPerTile), 1};
  uint32_t const elemStrides[] = {1, 1, 1, 1};
  // XQA Original Layout
#else
  uint64_t const globalDims[] = {headElems, tokensPerPage, nbKHeads, 1U << 31};
  uint32_t const headBytes = elemBytes * headElems;
  uint64_t const globalStrides[] = {headBytes, headBytes * tokensPerPage, headBytes * tokensPerPage * nbKHeads};
  uint32_t const partBytes = partElems * elemBytes;
  uint32_t const boxDims[] = {partElems, mha::min(tokensPerPage, nbTokensPerTile), 1, 1};
  uint32_t const elemStrides[] = {1, 1, 1, 1};
#endif

  auto const swizzle = [&] {
    switch (partBytes) {
      case 128:
        return CU_TENSOR_MAP_SWIZZLE_128B;
      case 64:
        return CU_TENSOR_MAP_SWIZZLE_64B;
      default:
        throw std::runtime_error("unsupported cache head size");
    }
  }();

  checkCu(cuTensorMapEncodeTiled(&tensorMap, dataType, 4, const_cast<void*>(addr), globalDims, globalStrides, boxDims,
                                 elemStrides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                 CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return tensorMap;
}
