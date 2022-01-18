// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef WINOGRAD_GENERATOR_H_
#define WINOGRAD_GENERATOR_H_

#include <memory>
#include <tuple>
#include "core/framework/tensor.h"

namespace onnxruntime {
using DimsVector = std::vector<int>;
typedef std::tuple<std::shared_ptr<float>, DimsVector> CMatrix;
CMatrix CMatrixCreate(int w, int h);
CMatrix CMatrixCreate(DimsVector dims);
DimsVector CMatrixGetStrides(CMatrix& matrix);

class WinogradGenerator {
 public:
  WinogradGenerator(int computeUnit, int kernelSize, float interp = 0.5f, bool transform_inner = false);
  ~WinogradGenerator() = default;

  CMatrix A() const {
    return A_;
  }
  CMatrix B() const {
    return B_;
  }
  CMatrix G() const {
    return G_;
  }

  CMatrix allocTransformWeight(int batch, int channel, int height, int width, int unitCi, int unitCo);
  void transformWeight(CMatrix& dest, const float* source, int batch, int channel, int height, int width);

 private:
  CMatrix A_;
  CMatrix G_;
  CMatrix B_;
  int unit_;
  int kernel_size_;
  bool transform_inner_;
};

}  // namespace TNN_NS

#endif  //WINOGRAD_GENERATOR_H_
