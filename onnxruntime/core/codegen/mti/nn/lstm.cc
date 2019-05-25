// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/codegen/mti/nn/lstm.h"

#include "core/codegen/mti/math/binary_ops.h"
#include "core/codegen/mti/math/unary_ops.h"
#include "core/codegen/mti/math/matmul_ops.h"
#include "core/codegen/mti/math/reduce_ops.h"
#include "core/codegen/mti/mti_tvm_utils.h"
#include "core/codegen/mti/tensor/reshape_ops.h"
#include "core/codegen/mti/tensor/split.h"

namespace onnxruntime {
namespace tvm_codegen {

/*
`X` - input tensor
`i` - input gate
`o` - output gate
`f` - forget gate
`c` - cell gate
`t` - time step (t-1 means previous time step)

`W[iofc]` - W parameter weight matrix for input, output, forget, and cell gates
`R[iofc]` - R recurrence weight matrix for input, output, forget, and cell gates
`Wb[iofc]` - W bias vectors for input, output, forget, and cell gates
`Rb[iofc]` - R bias vectors for input, output, forget, and cell gates
`P[iof]`  - P peephole weight vector for input, output, and forget gates
`WB[iofc]` - W parameter weight matrix for backward input, output, forget, and cell gates
`RB[iofc]` - R recurrence weight matrix for backward input, output, forget, and cell gates
`WBb[iofc]` - W bias vectors for backward input, output, forget, and cell gates
`RBb[iofc]` - R bias vectors for backward input, output, forget, and cell gates
`PB[iof]`  - P peephole weight vector for backward input, output, and forget gates

`H` - Hidden state
`num_directions` - 2 if direction == bidirectional else 1

Equations (Default: f=Sigmoid, g=Tanh, h=Tanh):
  it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
  ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
  ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
  Ct = ft (.) Ct-1 + it (.) ct
  ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
  Ht = ot (.) h(Ct)
*/

void LSTM_cell(
    const LSTMAttributes& lstm_attrs,
    const tvm::Tensor& X,
    const tvm::Tensor& W,
    const tvm::Tensor& R,
    const tvm::Tensor& B,
    bool has_B,
    const tvm::Tensor& prev_H,
    const tvm::Tensor& prev_C,
    const tvm::Tensor& P,
    bool has_P,
    tvm::Tensor& Y_h,
    tvm::Tensor& Y_c) {
  // Input projection: Xt*(W[iofc]^T) for forward direction or Xt*(WB[iofc]^T) for reverse direction
  // (batch_size, input_size) * trans(4 * hidden_size, input_size) => (batch_size, 4 * hidden_size)
  tvm::Tensor input_proj = MatMul2D(X, W, /*trans_a*/ false, /*trans_b*/ true);

  // Hidden projection: Ht-1*(R[iofc]^T) for forward direction or Ht-1*(RB[iofc]^T) for reverse direction
  // (batch_size, hidden_size) * trans(4 * hidden_size, hidden_size) => (batch_size, 4 * hidden_size)
  tvm::Tensor hidden_proj = MatMul2D(prev_H, R, /*trans_a*/ false, /*trans_b*/ true);

  // (batch_size, 4 * hidden_size)
  tvm::Tensor sum_proj = Add(input_proj, hidden_proj);

  // Concatenation of [Wb[iofc], Rb[iofc]] or [WBb[iofc], RBb[iofc]]
  if (has_B) {
    // (8 * hidden_size) -> (2, 4 * hidden_size) -> (1, 4 * hidden_size), should be done in const folding
    tvm::Tensor reduce_B =
        ReduceSum(Reshape(B, {2, 4 * static_cast<int>(lstm_attrs.hidden_size)}), {0}, /*keep_dims*/ true);
    // (batch_size, 4 * hidden_size) via broadcasting reduce_B
    sum_proj = Add(sum_proj, reduce_B);
  }

  std::vector<int64_t> iofc_sum_split_sizes(4, lstm_attrs.hidden_size);
  // Split sum_proj into iofc, where each gate proj is of (batch_size, hidden_size)
  tvm::Array<tvm::Tensor> iofc_sum_projs = Split(sum_proj, ToTvmArray(iofc_sum_split_sizes), /*axis*/ 1);
  MTI_ASSERT(iofc_sum_projs.size() == 4);
  tvm::Tensor i_proj = iofc_sum_projs[0],
              o_proj = iofc_sum_projs[1],
              f_proj = iofc_sum_projs[2],
              c_proj = iofc_sum_projs[3];

  tvm::Tensor P_i, P_o, P_f;
  if (has_P) {
    std::vector<int64_t> iof_p_split_sizes(3, lstm_attrs.hidden_size);
    // Split P into P_i, P_o, P_f, in const pre-processing (P_i, P_f might be merged?)
    // where each P_[iof] has the shape of (hidden_size)
    tvm::Array<tvm::Tensor> iof_P_projs = Split(P, ToTvmArray(iof_p_split_sizes), /*axis*/ 0);
    MTI_ASSERT(iof_P_projs.size() == 3);
    P_i = iof_P_projs[0],
    P_o = iof_P_projs[1],
    P_f = iof_P_projs[2];

    // (batch_size, hidden_size) via broadcasting P_[if]
    i_proj = Add(i_proj, Mul(P_i, prev_C));
    f_proj = Add(f_proj, Mul(P_f, prev_C));
  }

  // TODO: handle more general cases for activations f, h, g and activation_alpha and
  // activation_beta. We may consider to move some code such as ActivationInfo from deep_cpu_lstm
  // into a common header file, because the code can be used here.

  // Note that by default f = Sigmoid, g = Tanh, h = Tanh

  // it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
  // shape: (batch_size, hidden_size)
  tvm::Tensor i_t = Sigmoid(i_proj);
  // ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
  // shape: (batch_size, hidden_size)
  tvm::Tensor f_t = Sigmoid(f_proj);
  // ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
  // shape: (batch_size, hidden_size)
  tvm::Tensor c_t = Tanh(c_proj);

  // Ct = ft (.) Ct-1 + it (.) ct
  // shape: (batch_size, hidden_size)
  Y_c = Add(Mul(f_t, prev_C), Mul(i_t, c_t), Y_c->op->name);

  // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
  // shape: (batch_size, hidden_size)
  if (has_P) {
    o_proj = Add(o_proj, Mul(P_o, Y_c));
  }
  // ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
  // shape: (batch_size, hidden_size)
  o_proj = Sigmoid(o_proj);
  // Ht = ot (.) h(Ct)
  // shape: (batch_size, hidden_size)
  Y_h = Mul(o_proj, Tanh(Y_c), Y_h->op->name);
}

}  // namespace tvm_codegen
}  // namespace onnxruntime
