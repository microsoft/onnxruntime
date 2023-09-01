/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/cudaDriverWrapper.h"
#include "contrib_ops/cuda/bert/tensorrt_fused_multihead_attention/fused_multihead_attention_common.h"
#include <cuda_runtime_api.h>
#include <memory>
#include <mutex>
#include <set>
#include <stdint.h>
#include <unordered_map>
#include <vector>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename TKernelMeta, typename TKernelParam>
class TSharedCubinKernel {
 public:
  using KernelMeta = TKernelMeta;
  using KernelParam = TKernelParam;

  virtual uint64_t hashID(KernelMeta const& kernelMeta) const = 0;
  virtual uint64_t hashID(TKernelParam const& param) const = 0;

  TSharedCubinKernel(TKernelMeta const* pMetaStart, int32_t nMetaCount, Data_type type, int32_t sm)
      : mDataType(type), mKernelMeta(pMetaStart), mKernelMetaCount(nMetaCount), mSM(sm) {
  }

  void loadCubinKernels(int32_t smVersion) {
    for (int32_t i = 0; i < mKernelMetaCount; ++i) {
      auto const& kernelMeta = mKernelMeta[i];
      auto const kernelKey = hashID(kernelMeta);
      if (kernelMeta.mSM == smVersion &&
          kernelMeta.mDataType == mDataType &&
          mFunctions.find(kernelKey) == mFunctions.end()) {
        constexpr int32_t DEFAULT_SMEM_SIZE = 48 * 1024;
        if (kernelMeta.mSharedMemBytes >= DEFAULT_SMEM_SIZE) {
          int32_t deviceID{0};
          cudaGetDevice(&deviceID);
          int32_t sharedMemPerMultiprocessor{0};
          if (cudaDeviceGetAttribute(
                  &sharedMemPerMultiprocessor, cudaDevAttrMaxSharedMemoryPerBlockOptin, deviceID) != cudaSuccess ||
              sharedMemPerMultiprocessor < kernelMeta.mSharedMemBytes) {
            // skip load function because not enough shared memory to launch the kernel
            printf("skip loading trt attention kernel %s because no enough shared memory",
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
          if (mDriver.cuFuncSetAttribute(funcInfo.mDeviceFunction,
                                         CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                         kernelMeta.mSharedMemBytes) != CUDA_SUCCESS) {
            // some chip may not have enough shared memory to launch the kernel
            printf("skip loading trt attention kernel %s because no enough shared memory",
                   kernelMeta.mFuncName);
            continue;
          }
        }
        mFunctions.insert({kernelKey, funcInfo});
      }
    }
  }

  void loadCubinKernels() {
    if (!mFunctions.empty()) {
      return;
    }

    loadCubinKernels(mSM);
  }

  bool isValid(int32_t /*s*/) const {
    return !mFunctions.empty();
  }

  virtual void dumpHashId(TKernelParam const& param, std::ostringstream& message) const = 0;

  virtual int32_t getSForUnroll(TKernelParam const& param) const = 0;

  virtual void run(TKernelParam& params, cudaStream_t ss) const {
    ORT_ENFORCE(!params.interleaved);  // interleaved is for int8

    auto const findIter = mFunctions.find(hashID(params));
    if (findIter == mFunctions.end()) {
      std::ostringstream errMsg;
      errMsg << "Could not find kernel for:\n";
      dumpHashId(params, errMsg);
      errMsg << "\t Compiled on CUDA " << CUDA_VERSION << "\n"
             << "\t Current SM version: " << mSM << "\n"
             << "\t SM versions enabled during compilation: 75, 80, 86, 89"
             << "\n";
      ORT_ENFORCE(findIter != mFunctions.end(), errMsg.str().c_str());
    }

    auto const& kernelMeta = mKernelMeta[findIter->second.mMetaInfoIndex];
    CUfunction const func = findIter->second.mDeviceFunction;

    void* kernelParams[] = {&params, nullptr};
    if (!params.force_unroll) {
      cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, 1, kernelMeta.mThreadsPerCTA, 1, 1,
                                        kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
                 mDriver);
    } else {
      int32_t unroll = (getSForUnroll(params) + kernelMeta.mUnrollStep - 1) / kernelMeta.mUnrollStep;
      cuErrCheck(mDriver.cuLaunchKernel(func, params.h, params.b, unroll, kernelMeta.mThreadsPerCTA, 1, 1,
                                        kernelMeta.mSharedMemBytes, ss, kernelParams, nullptr),
                 mDriver);
    }
  }

  virtual ~TSharedCubinKernel() = default;

 protected:
  CUDADriverWrapper mDriver;

  Data_type mDataType;
  TKernelMeta const* mKernelMeta;
  int32_t mKernelMetaCount;
  int32_t mSM;
  std::unordered_map<unsigned char const*, CUmodule> mModules;
  struct FusedMultiHeadAttentionKernelInfo {
    int32_t mMetaInfoIndex;
    CUfunction mDeviceFunction;
  };
  std::unordered_map<uint64_t, FusedMultiHeadAttentionKernelInfo> mFunctions;
};

template <typename TKernelList>
class TSharedCubinKernelFactory {
 public:
  TKernelList const* getCubinKernels(
      typename TKernelList::KernelMeta const* pKernelList, int32_t nbKernels, Data_type type, int32_t sm) {
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lg(sMutex);

    auto const id = hashID(type, sm);
    auto const findIter = mKernels.find(id);
    if (findIter == mKernels.end()) {
      auto newKernel = std::make_unique<TKernelList>(pKernelList, nbKernels, type, sm);
      newKernel->loadCubinKernels();
      auto const insert_result = mKernels.insert(std::make_pair(id, std::move(newKernel)));
      return insert_result.first->second.get();
    }
    return findIter->second.get();
  }

  static TSharedCubinKernelFactory<TKernelList>& Get() {
    static TSharedCubinKernelFactory<TKernelList> gFactory;
    return gFactory;
  }

 private:
  TSharedCubinKernelFactory() = default;

  inline uint64_t hashID(Data_type type, int32_t sm) const {
    // use deviceID in hasID for multi GPU support before driver support context-less loading of cubin
    int32_t deviceID{0};
    CUDA_CALL_THROW(cudaGetDevice(&deviceID));
    ORT_ENFORCE((deviceID & 0xFFFF) == deviceID);
    return (uint64_t)type << 48 | (uint64_t)deviceID << 32 | sm;
  }

  std::unordered_map<uint64_t, std::unique_ptr<TKernelList> const> mKernels;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
