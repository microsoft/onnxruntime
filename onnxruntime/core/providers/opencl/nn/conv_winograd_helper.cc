#include "conv_winograd_helper.h"
#include <core/providers/opencl/opencl_utils.h>
#include <cmath>
// #include <memory.h>

namespace onnxruntime {

/*
get cmatrix stride
*/
FBshape GetStridesY(DirectBufferPtr matrix) {
  auto dims = matrix->shape;
  FBshape strides;
  int count = 1;
  for (auto iter : dims) {
    count *= iter;
  }
  for (auto iter : dims) {
    count /= iter;
    strides.push_back(count);
  }
  return strides;
}

/*
1D: AT*((G*g)(BT*d))
2D: AT*((G*g*GT)(BT*d*B))*A
https://github.com/andravin/wincnn
*/
WinogradHelper::WinogradHelper(int computeUnit, int kernelSize) {
  ORT_ENFORCE(computeUnit > 0 && kernelSize > 0);
  unit_ = computeUnit;
  kernel_size_ = kernelSize;

  wino_size_ = computeUnit + kernelSize - 1;
  // G_ is used to transform conv_weight
  G_ = DirectBufferPtr(new DirectBuffer);
  G_->Create(kernelSize, wino_size_);
  // TODO only for F(2,3)
  G_->Fill({1, 0, 0, 1, 1, 1, 1, -1, 1, 0, 0, 1.0});
}

/*
transform weight size: unit*unit*ROUND_UP(oc, 4)*ROUND_UP(ic, 4)
*/
DirectBufferPtr WinogradHelper::AllocWeightTensor(int ochannel, int ichannel, int unitCi, int unitCo) {
  int ciC4 = opencl::CeilDiv(ichannel, unitCi);
  int coC4 = opencl::CeilDiv(ochannel, unitCo);
  DirectBufferPtr p = DirectBufferPtr(new DirectBuffer);
  p->Create({16, coC4, ciC4, unitCi, unitCo});
  return p;
}

// Winograd 4x4 For Conv 3x3
static inline void WeightTransform4x4_3x3(const float* src, float* dst, DirectBufferPtr G, const FBshape& weight_dest_strides,
                                          int in_channel, int out_channel, int oz_index, int alpha_index) {
  int ic_stride = 9;
  int oc_stride = in_channel * ic_stride;
  int unit_co = 4;
  int unit_ci = 4;

  float GgGt[16];
  float Gg[12];
  const float* g = G->buff.get();
  for (int oz = 0; oz < out_channel; ++oz) {
    auto srcOz = src + oz * oc_stride;

    int ozC4 = oz / unit_co;
    int mx = oz % unit_co;

    auto dstOz = dst + weight_dest_strides[oz_index] * ozC4 + mx;
    for (int sz = 0; sz < in_channel; ++sz) {
      int szC4 = sz / unit_ci;
      int my = sz % unit_ci;
      auto srcSz = srcOz + ic_stride * sz;
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

      auto dstSz = dstOz + szC4 * weight_dest_strides[2] + unit_co * my;
      // [alpha][alpha][oc4][ic4][16]
      for (int i = 0; i < 16; ++i) {
        *dstSz = GgGt[i];
        dstSz += weight_dest_strides[alpha_index];
      }
    }
  }
}

/*
transform weight from [oc][ic][kh][kw] to [unit][unit][co4][ci4][16]
*/
DirectBufferPtr WinogradHelper::TransformWeight(const float* source, int output_channel, int input_channel) {
  DirectBufferPtr weightDest_ptr = AllocWeightTensor(output_channel, input_channel, 4, 4);
  auto weight_dest_data = weightDest_ptr->buff.get();
  auto& weight_dest_dims = weightDest_ptr->shape;

  auto weight_dest_strides = GetStridesY(weightDest_ptr);

  int ci = input_channel;
  int co = output_channel;
  int unitCi = weight_dest_dims[3];
  int unitCo = weight_dest_dims[4];
  if (ci % unitCi != 0 || co % unitCo != 0) {
    int result = 1;
    for (int index = 0; index < weight_dest_dims.size(); ++index) {
      result *= weight_dest_dims[index];
    }
    ::memset(weight_dest_data, 0, result * sizeof(float));
  }

  if (unitCi == 4 && unitCo == 4 && kernel_size_ == 3) {
    WeightTransform4x4_3x3(source, weight_dest_data, G_,
                           weight_dest_strides, ci, co, 1, 0);
  } else {
    ORT_THROW("only surpport F(2,3) yet");
  }
  return weightDest_ptr;
}

}  // namespace onnxruntime
