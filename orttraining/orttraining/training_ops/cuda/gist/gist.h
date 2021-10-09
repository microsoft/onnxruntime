// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class GistBinarizeEncoderOp final : public CudaKernel {
 public:
  GistBinarizeEncoderOp(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistBinarizeDecoderOp final : public CudaKernel {
 public:
  GistBinarizeDecoderOp(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistPack1EncoderOp final : public CudaKernel {
 public:
  static constexpr int GIST_PACK1_FACTOR = 8;
  GistPack1EncoderOp(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistPack1DecoderOp final : public CudaKernel {
 public:
  static constexpr int GIST_PACK1_FACTOR = 8;
  GistPack1DecoderOp(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistPack8EncoderOp final : public CudaKernel {
 public:
  GistPack8EncoderOp(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistPack8DecoderOp final : public CudaKernel {
 public:
  GistPack8DecoderOp(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistPack16EncoderOp final : public CudaKernel {
 public:
  GistPack16EncoderOp(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistPack16DecoderOp final : public CudaKernel {
 public:
  GistPack16DecoderOp(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistPackMsfp15EncoderOp final : public CudaKernel {
 public:
  GistPackMsfp15EncoderOp(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

template <typename T>
class GistPackMsfp15DecoderOp final : public CudaKernel {
 public:
  GistPackMsfp15DecoderOp(const OpKernelInfo& info) : CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace onnxruntime
