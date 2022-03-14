// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv_winograd_helper.h"
#include <core/providers/opencl/opencl_utils.h>
#include <cmath>
// #include <memory.h>

namespace onnxruntime {

/*
1D: AT*((G*g)(BT*d))
2D: AT*((G*g*GT)(BT*d*B))*A
https://github.com/andravin/wincnn
*/
WinogradHelper::WinogradHelper(AllocatorPtr& cpu_alloc, int compute_unit, int kernel_size) : cpu_alloc_{cpu_alloc} {
  ORT_ENFORCE(compute_unit > 0 && kernel_size > 0);

  constexpr std::array<float, 12> wino_init_data{1, 0, 0, 1, 1, 1, 1, -1, 1, 0, 0, 1};

  unit_ = compute_unit;
  kernel_size_ = kernel_size;

  wino_size_ = compute_unit + kernel_size - 1;
  // G_ is used to transform conv_weight
  G_ = Tensor::Create(DataTypeImpl::GetType<float>(), {kernel_size, wino_size_}, cpu_alloc_);
  std::copy_n(wino_init_data.data(), wino_init_data.size(), G_->MutableData<float>());
}

/*
transform weight size: unit*unit*ROUND_UP(oc, 4)*ROUND_UP(ic, 4)
*/
std::unique_ptr<Tensor> WinogradHelper::AllocWeightTensor(int ochannel, int ichannel, int unit_ci, int unit_co) {
  int ciC4 = opencl::CeilDiv(ichannel, unit_ci);
  int coC4 = opencl::CeilDiv(ochannel, unit_co);
  return Tensor::Create(DataTypeImpl::GetType<float>(), {16LL, coC4, ciC4, unit_ci, unit_co}, cpu_alloc_);
}

// Winograd 4x4 For Conv 3x3
inline void WeightTransform4x4_3x3(
    const Tensor* src,
    Tensor* dst,
    const Tensor* G,
    const TensorShapeVector& dst_strides,
    int in_channel, int out_channel,
    int oz_index, int alpha_index) {
  int ic_stride = 9;
  int oc_stride = in_channel * ic_stride;
  int unit_co = 4;
  int unit_ci = 4;

  float GgGt[16];
  float Gg[12];
  const float* g = G->Data<float>();
  for (int oz = 0; oz < out_channel; ++oz) {
    auto srcOz = src->Data<float>() + oz * oc_stride;

    int ozC4 = oz / unit_co;
    int mx = oz % unit_co;

    auto dstOz = dst->MutableData<float>() + dst_strides[oz_index] * ozC4 + mx;
    for (int sz = 0; sz < in_channel; ++sz) {
      int szC4 = sz / unit_ci;
      int my = sz % unit_ci;
      const auto* srcSz = srcOz + ic_stride * sz;
      const float* k0 = srcSz;
      const float* k1 = k0 + 3;
      const float* k2 = k1 + 3;

      // M = G * K
      Gg[0] = k0[0] * g[0] + k1[0] * g[1] + k2[0] * g[2];
      Gg[1] = k0[1] * g[0] + k1[1] * g[1] + k2[1] * g[2];
      Gg[2] = k0[2] * g[0] + k1[2] * g[1] + k2[2] * g[2];
      Gg[3] = k0[0] * g[3] + k1[0] * g[4] + k2[0] * g[5];
      Gg[4] = k0[1] * g[3] + k1[1] * g[4] + k2[1] * g[5];
      Gg[5] = k0[2] * g[3] + k1[2] * g[4] + k2[2] * g[5];
      Gg[6] = k0[0] * g[6] + k1[0] * g[7] + k2[0] * g[8];
      Gg[7] = k0[1] * g[6] + k1[1] * g[7] + k2[1] * g[8];
      Gg[8] = k0[2] * g[6] + k1[2] * g[7] + k2[2] * g[8];
      Gg[9] = k0[0] * g[9] + k1[0] * g[10] + k2[0] * g[11];
      Gg[10] = k0[1] * g[9] + k1[1] * g[10] + k2[1] * g[11];
      Gg[11] = k0[2] * g[9] + k1[2] * g[10] + k2[2] * g[11];

      // K_Transform = M*GT
      const float* gt0 = g;
      const float* gt1 = gt0 + 1;
      const float* gt2 = gt1 + 1;
      GgGt[0] = Gg[0] * gt0[0] + Gg[1] * gt1[0] + Gg[2] * gt2[0];
      GgGt[1] = Gg[0] * gt0[3 * 1] + Gg[1] * gt1[3 * 1] + Gg[2] * gt2[3 * 1];
      GgGt[2] = Gg[0] * gt0[3 * 2] + Gg[1] * gt1[3 * 2] + Gg[2] * gt2[3 * 2];
      GgGt[3] = Gg[0] * gt0[3 * 3] + Gg[1] * gt1[3 * 3] + Gg[2] * gt2[3 * 3];
      GgGt[4] = Gg[3] * gt0[3 * 0] + Gg[4] * gt1[3 * 0] + Gg[5] * gt2[3 * 0];
      GgGt[5] = Gg[3] * gt0[3 * 1] + Gg[4] * gt1[3 * 1] + Gg[5] * gt2[3 * 1];
      GgGt[6] = Gg[3] * gt0[3 * 2] + Gg[4] * gt1[3 * 2] + Gg[5] * gt2[3 * 2];
      GgGt[7] = Gg[3] * gt0[3 * 3] + Gg[4] * gt1[3 * 3] + Gg[5] * gt2[3 * 3];
      GgGt[8] = Gg[6] * gt0[0] + Gg[7] * gt1[0] + Gg[8] * gt2[0];
      GgGt[9] = Gg[6] * gt0[3 * 1] + Gg[7] * gt1[3 * 1] + Gg[8] * gt2[3 * 1];
      GgGt[10] = Gg[6] * gt0[3 * 2] + Gg[7] * gt1[3 * 2] + Gg[8] * gt2[3 * 2];
      GgGt[11] = Gg[6] * gt0[3 * 3] + Gg[7] * gt1[3 * 3] + Gg[8] * gt2[3 * 3];
      GgGt[12] = Gg[9] * gt0[3 * 0] + Gg[10] * gt1[0] + Gg[11] * gt2[0];
      GgGt[13] = Gg[9] * gt0[3 * 1] + Gg[10] * gt1[3 * 1] + Gg[11] * gt2[3 * 1];
      GgGt[14] = Gg[9] * gt0[3 * 2] + Gg[10] * gt1[3 * 2] + Gg[11] * gt2[3 * 2];
      GgGt[15] = Gg[9] * gt0[3 * 3] + Gg[10] * gt1[3 * 3] + Gg[11] * gt2[3 * 3];

      auto dstSz = dstOz + szC4 * dst_strides[2] + unit_co * my;
      // [alpha][alpha][oc4][ic4][16]
      for (int i = 0; i < 16; ++i) {
        *dstSz = GgGt[i];
        dstSz += dst_strides[alpha_index];
      }
    }
  }
}

// TODO: improve!
TensorShapeVector GetStrides(const Tensor* matrix) {
  auto dims = matrix->Shape();
  TensorShapeVector strides;
  int count = 1;
  for (auto iter : dims.AsShapeVector()) {
    count *= iter;
  }
  for (auto iter : dims.AsShapeVector()) {
    count /= iter;
    strides.push_back(count);
  }
  return strides;
}

/*
transform weight from [oc][ic][kh][kw] to [unit][unit][co4][ci4][16]
*/
std::unique_ptr<Tensor> WinogradHelper::TransformWeight(const Tensor* source, int output_channel, int input_channel) {
  auto dst = AllocWeightTensor(output_channel, input_channel, 4, 4);
  const auto& dst_shape = dst->Shape();
  auto dst_strides = GetStrides(dst.get());

  int ci = input_channel;
  int co = output_channel;
  int unitCi = dst_shape[3];
  int unitCo = dst_shape[4];
  std::fill_n(dst->MutableData<float>(), dst_shape.Size(), 0.0);

  if (unitCi == 4 && unitCo == 4 && kernel_size_ == 3) {
    WeightTransform4x4_3x3(source, dst.get(), G_.get(), dst_strides, ci, co, 1, 0);
  } else {
    ORT_THROW("only surpport F(2,3)");
  }
  return dst;
}

}  // namespace onnxruntime
