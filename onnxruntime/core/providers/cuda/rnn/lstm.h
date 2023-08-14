// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cudnn_rnn_base.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
class LSTM final : public CudnnRnnBase<T> {
 public:
  LSTM(const OpKernelInfo& info) : CudnnRnnBase<T>(info) {
    CudnnRnnBase<T>::SetRNNMode(CUDNN_LSTM);

    // ONNX W layout is W[iofc], WB[iofc], mapping to RNNLinLayerMatrixParams the linLayerID is 0, 3, 1, 2
    CudnnRnnBase<T>::W_lin_layer_id_.assign({0, 3, 1, 2});
    // ONNX R layout is R[iofc], RB[iofc], mapping to RNNLinLayerMatrixParams the linLayerID is 4, 7, 5, 6
    CudnnRnnBase<T>::R_lin_layer_id_.assign({4, 7, 5, 6});
    // ONNX B layout is Wb[iofc], Rb[iofc], mapping to RNNLinLayerMatrixParams
    // the linLayerID is 0, 3, 1, 2, 4, 7, 5, 6, we can reuse it from W_lin_layer_id & R_lin_layer_id

    ORT_THROW_IF_ERROR(CudnnRnnBase<T>::CacheCudnnRnnWeights(info));
  }
};

}  // namespace cuda
}  // namespace onnxruntime
