/*All contributions by Facebook :
Copyright(c) 2016 Facebook Inc.
==============================================================================*/
/* Modifications Copyright (c) Microsoft. */

#pragma once
#include "core/providers/cuda/cuda_common.h"
#include "cufft.h"
#include "cufftXt.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

//key
struct FFTState {
  int64_t signal_ndim;
  int64_t signal_dims[5];
  cudaDataType itype;
  cudaDataType otype;
  int64_t batch_size;
  cudaDataType exec_type;
};

//value
struct CufftPlanInfo {
  cufftHandle plan;
  size_t ws_size_t;
};

// Hashing machinery for Params
// Fowler–Noll–Vo hash function
// see https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
template <typename T>
struct ParamsHash {
  // Params must be a POD because we read out its memory
  // contenst as char* when hashing

  static_assert(std::is_pod<T>::value, "Params is not POD");
  size_t operator()(const T& params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&params);
    uint32_t value = 0x811C9DC5;
    for (int i = 0; i < (int)sizeof(T); ++i) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return (size_t)value;
  }
};

template <typename T>
struct ParamsEqual {
  // Params must be a POD because we read out its memory
  // contenst as char* when comparing

  static_assert(std::is_pod<T>::value, "Params is not POD");

  bool operator()(const T& a, const T& b) const {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
    auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
    return memcmp(ptr1, ptr2, sizeof(T)) == 0;
  }
};

class CuFFTPlanCache {
 public:
  CufftPlanInfo TryEmplaceValue(FFTState& key) {
    std::lock_guard<std::mutex> lock(mutex);

    auto it = map.find(key);
    if (it == map.end()) {
      CufftPlanInfo plan_info = CreatePlanInfo(key);
      map.emplace(key, plan_info);
      return plan_info;
    } else {
      return it->second;
    }
  }

  int64_t GetCacheSize() { return map.size(); }

  std::mutex mutex;

 private:
  CufftPlanInfo CreatePlanInfo(FFTState& key) {
    cufftHandle plan;
    size_t ws_size_t;
    CufftPlanInfo plan_info;

    CUFFT_CALL_THROW(cufftCreate(&plan));

    CUFFT_CALL_THROW(cufftXtMakePlanMany(plan, static_cast<int>(key.signal_ndim), reinterpret_cast<long long int*>(key.signal_dims),
                                         /* inembed */ nullptr, /* base_istride */ 1, /* idist */ 1, key.itype,
                                         /* onembed */ nullptr, /* base_ostride */ 1, /* odist */ 1, key.otype,
                                         key.batch_size, &ws_size_t, key.exec_type));

    plan_info.plan = plan;
    plan_info.ws_size_t = ws_size_t;

    return plan_info;
  }

  std::unordered_map<FFTState, CufftPlanInfo, ParamsHash<FFTState>, ParamsEqual<FFTState>> map;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
