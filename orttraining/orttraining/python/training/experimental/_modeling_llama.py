from typing import List

import onnxscript
from onnxscript import opset18

custom_opset = onnxscript.values.Opset(domain="com.microsoft", version=1)
aten_opset = onnxscript.values.Opset(domain="org.pytorch.aten", version=1)


@onnxscript.script(custom_opset, default_opset=opset18)
def scaled_dot_product_efficient_attention(
    query, key, value, attn_bias, compute_log_sumexp: bool, dropout_p: float, is_causal: bool
):
    # Observed arguments from FX graph:
    # %_scaled_dot_product_efficient_attention : [num_users=4] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention.default](
    #    args = (%clone, %clone_1, %clone_2, %expand, True, 0.0, True), kwargs = {}
    # )
    # %getitem : [num_users=2] = call_function[target=operator.getitem](args = (%_scaled_dot_product_efficient_attention, 0), kwargs = {})
    # %getitem_1 : [num_users=1] = call_function[target=operator.getitem](args = (%_scaled_dot_product_efficient_attention, 1), kwargs = {})
    # %getitem_2 : [num_users=1] = call_function[target=operator.getitem](args = (%_scaled_dot_product_efficient_attention, 2), kwargs = {})
    # %getitem_3 : [num_users=1] = call_function[target=operator.getitem](args = (%_scaled_dot_product_efficient_attention, 3), kwargs = {})
    # (%clone, %clone_1, %clone_2, %expand, True, 0.0, True)
    #
    # Official signature:
    # func: _scaled_dot_product_efficient_attention(
    #   Tensor query,
    #   Tensor key,
    #   Tensor value,
    #   Tensor? attn_bias,
    #   bool compute_log_sumexp,
    #   float dropout_p=0.0,
    #   bool is_causal=False,
    #   float? scale=None
    # ) -> (Tensor output, Tensor log_sumexp, Tensor philox_seed, Tensor philox_offset)
    output, log_sumexp, philox_seed, philox_offset = aten_opset.ATen(
        query,
        key,
        value,
        attn_bias,
        compute_log_sumexp,
        dropout_p,
        is_causal,
        1.0,
        operator="_scaled_dot_product_efficient_attention",
    )

    return output, log_sumexp, philox_seed, philox_offset


@onnxscript.script(aten_opset, default_opset=opset18)
def scaled_dot_product_efficient_attention_backward(
    grad,
    query,
    key,
    value,
    attn_bias,
    output,
    logsumexp,
    philox_seed,
    philox_offset,
    dropout_p,
    grad_input_mask: List[bool],
    is_causal: bool,
):
    # Observed arguments from FX graph:
    # %_scaled_dot_product_efficient_attention_backward : [num_users=3] = call_function[target=torch.ops.aten._scaled_dot_product_efficient_attention_backward.default](
    # args = (
    #    %transpose_4,
    #    %clone,
    #    %clone_1,
    #    %clone_2,
    #    %expand,
    #    %detach_13,
    #    %getitem_1,
    #    %getitem_2,
    #    %getitem_3,
    #    0.0,
    #    [True, True, True, False],
    #    True
    # ), kwargs = {})
    #
    # Official signature:
    # - func: _scaled_dot_product_efficient_attention_backward(Tensor grad_out_, Tensor query, Tensor key, Tensor value, Tensor attn_bias, Tensor out, Tensor logsumexp, Tensor philox_seed, Tensor philox_offset, float dropout_p, bool[4] grad_input_mask, bool is_causal=False, *, float? scale=None) -> (Tensor, Tensor, Tensor, Tensor)
    #   device_check: NoCheck
    #   dispatch:
    #     CUDA: _scaled_dot_product_efficient_attention_backward_cuda
    #   tags: nondeterministic_seeded
    grad_query, grad_key, grad_value, grad_attn_bias = aten_opset.ATen(
        grad,
        query,
        key,
        value,
        attn_bias,
        output,
        logsumexp,
        philox_seed,
        philox_offset,
        dropout_p,
        grad_input_mask,
        is_causal,
        1.0,
        operator="_scaled_dot_product_efficient_attention_backward",
    )
    return grad_query, grad_key, grad_value, grad_attn_bias
