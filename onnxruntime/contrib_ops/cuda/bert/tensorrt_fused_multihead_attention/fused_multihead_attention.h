/*
 * Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <memory>
#include <mutex>
#include <set>
#include <cstdint>
#include <utility>
#include <unordered_map>
#include <vector>
#include <cuda_runtime_api.h>
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/cudaDriverWrapper.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/fused_multihead_attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename TKernelMeta, typename TKernelParam>
class TFusedMultiHeadAttentionXMMAKernel {
 public:
  using KernelMeta = TKernelMeta;
  using KernelParam = TKernelParam;
  inline uint64_t hashID(unsigned int s, unsigned int d) const {
    return (uint64_t)s << 32 | d;
  }
  virtual uint64_t hashID(const KernelMeta& kernelMeta) const {
    return hashID(kernelMeta.mS, kernelMeta.mD);
  }

  TFusedMultiHeadAttentionXMMAKernel(
      const TKernelMeta* pMetaStart, unsigned int nMetaCount, Data_type type, unsigned int sm)
      : mDataType(type), mKernelMeta(pMetaStart), mKernelMetaCount(nMetaCount), mSM(sm) {
  }

  void loadXMMAKernels(uint32_t smVersion) {
    for (uint32_t i = 0; i < mKernelMetaCount; ++i) {
      const auto& kernelMeta = mKernelMeta[i];
      const auto kernelKey = hashID(kernelMeta);
      if (kernelMeta.mSM == smVersion &&
          kernelMeta.mDataType == mDataType &&
          mFunctions.find(kernelKey) == mFunctions.end()) {
        constexpr uint32_t DEFAULT_SMEM_SIZE = 48 * 1024;
        if (kernelMeta.mSharedMemBytes >= DEFAULT_SMEM_SIZE) {
          int32_t deviceID{0};
          cudaGetDevice(&deviceID);
          int32_t sharedMemPerMultiprocessor{0};
          if (cudaDeviceGetAttribute(
                  &sharedMemPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceID) != cudaSuccess ||
              sharedMemPerMultiprocessor < static_cast<int32_t>(kernelMeta.mSharedMemBytes)) {
            // skip load function because not enough shared memory to launch the kernel
            printf("skip loading trt fused attention kernel %s because no enough shared memory",
                   kernelMeta.mFuncName);
            continue;
          }
        }
        CUmodule hmod{0};
        auto findModuleIter = mModules.find(kernelMeta.mCubin);
        if (findModuleIter != mModules.end()) {
          hmod = findModuleIter->second;
        } else {
          cuErrCheck(mDriver.cuModuleLoadData(&hmod, kernelMeta.mCubin), mDriver);
          mModules.insert(std::make_pair(kernelMeta.mCubin, hmod));
        }
        FusedMultiHeadAttentionKernelInfo funcInfo;
        funcInfo.mMetaInfoIndex = i;
        cuErrCheck(mDriver.cuModuleGetFunction(&funcInfo.mDeviceFunction, hmod, kernelMeta.mFuncName), mDriver);
        if (kernelMeta.mSharedMemBytes >= DEFAULT_SMEM_SIZE) {
          if (CUDA_SUCCESS != mDriver.cuFuncSetAttribute(funcInfo.mDeviceFunction,
                                                         CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                                         kernelMeta.mSharedMemBytes)) {
            // some chip may not have enough shared memory to launch the kernel
            printf("skip loading trt fused attention kernel %s because no enough shared memory",
                   kernelMeta.mFuncName);
            continue;
          }
        }
        mFunctions.insert({kernelKey, funcInfo});
        const int s = static_cast<int>(kernelMeta.mS);
        if (mValidSequences.find(s) == mValidSequences.end()) {
          mValidSequences.insert(s);
        }
      }
    }
  }

  void loadXMMAKernels() {
    if (!mFunctions.empty()) {
      return;
    }

    loadXMMAKernels(mSM);
    
    // sm_86 chips prefer sm_86 kernel, but can also use sm_80 kernel if sm_86 not exist.
    // sm_89 will reuse sm_80 kernels
    if (mSM == kSM_86 || mSM == kSM_89) {
      loadXMMAKernels(kSM_80);
    }
  }

  bool isValid(int s) const {
    return (mValidSequences.find(s) != mValidSequences.end());
  }

  virtual void run(TKernelParam& params, cudaStream_t ss, bool flash_attention = false, bool causal_mask = false) const {
    ORT_UNUSED_PARAMETER(flash_attention);
    ORT_UNUSED_PARAMETER(causal_mask);

    const auto findIter = mFunctions.find(hashID(params.s, params.d));
    ORT_ENFORCE(findIter != mFunctions.end());

    const auto& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
    const CUfunction func = findIter->second.mDeviceFunction;

    void* kernelParams[] = {&params, nullptr};
    cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                                      kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
               mDriver);
  }

  virtual ~TFusedMultiHeadAttentionXMMAKernel() = default;

 protected:
  CUDADriverWrapper mDriver;

  Data_type mDataType;
  const TKernelMeta* mKernelMeta;
  unsigned int mKernelMetaCount;
  unsigned int mSM;
  std::unordered_map<const unsigned char*, CUmodule> mModules;
  struct FusedMultiHeadAttentionKernelInfo {
    unsigned int mMetaInfoIndex;
    CUfunction mDeviceFunction;
  };
  std::unordered_map<uint64_t, FusedMultiHeadAttentionKernelInfo> mFunctions;
  std::set<int> mValidSequences;
};

template <typename TFusedMHAKernelList>
class TFusedMHAKernelFactory {
 public:
  const TFusedMHAKernelList* getXMMAKernels(const typename TFusedMHAKernelList::KernelMeta* pKernelList,
                                            unsigned int nbKernels, Data_type type, unsigned int sm) {
    static std::mutex s_mutex;
    std::lock_guard<std::mutex> lg(s_mutex);

    const auto id = hashID(type, sm);
    const auto findIter = mKernels.find(id);
    if (findIter == mKernels.end()) {
      GSL_SUPPRESS(r.11)
      std::unique_ptr<TFusedMHAKernelList> newKernel(new TFusedMHAKernelList{pKernelList, nbKernels, type, sm});
      newKernel->loadXMMAKernels();
      TFusedMHAKernelList* ret = newKernel.get();
      mKernels.emplace(id, std::move(newKernel));      
      return ret;
    }
    return findIter->second.get();
  }

  static TFusedMHAKernelFactory<TFusedMHAKernelList>& Get() {
    static TFusedMHAKernelFactory<TFusedMHAKernelList> s_factory;
    return s_factory;
  }

 private:
  TFusedMHAKernelFactory() = default;

  inline uint64_t hashID(Data_type type, uint32_t sm) const {
    // use deviceID in hasID for multi GPU support before driver support context-less loading of cubin
    int32_t deviceID{0};
    CUDA_CALL_THROW(cudaGetDevice(&deviceID));
    ORT_ENFORCE((deviceID & 0xFFFF) == deviceID);
    return (uint64_t)type << 48 | (uint64_t)deviceID << 32 | sm;
  }

  std::unordered_map<uint64_t, const std::unique_ptr<TFusedMHAKernelList>> mKernels;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
