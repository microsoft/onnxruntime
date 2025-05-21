# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from typing import ClassVar

import torch
from onnxscript import ir

try:
    import whisper
except ImportError:
    whisper = None

_custom_patches = []

if whisper:
    if whisper.model.SDPA_AVAILABLE and whisper.model.MultiHeadAttention.use_sdpa:

        class PatchedMultiHeadAttention(torch.nn.Module):
            _PATCHED_CLASS_ = whisper.model.MultiHeadAttention
            _PATCHES_: ClassVar = ["qkv_attention"]

            @classmethod
            def _qkv_attentation_causal(cls, q, k, v, mask, condition):
                return torch.cond(
                    condition,
                    lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True),
                    lambda q, k, v: torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False),
                    [q, k, v],
                )

            def qkv_attention(self, q, k, v, mask=None):
                # q.shape == k.shape == v.shape == torch.Size([2, 2, 384])
                # mask.shape == torch.Size([448, 448])
                q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
                k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
                v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

                # a = scaled_dot_product_attention(q, k, v, is_causal=mask is not None and n_ctx > 1)
                if mask is None:
                    a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
                else:
                    a = self._qkv_attentation_causal(q, k, v, mask, q.shape[2] > 1)
                out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
                qk = None
                # out.shape == torch.Size([2, 2, 384])
                return out, qk

        # The patch does this:
        # whisper.model.MultiHeadAttention.qkv_attention = patched_MultiHeadAttention.qkv_attention
        # whisper.model.MultiHeadAttention._qkv_attentation_causal = patched_MultiHeadAttention._qkv_attentation_causal
        _custom_patches.append(patched_MultiHeadAttention)


def optimize_for_ort(
    model: ir.Model,
    config_name: str | None = None,
    *,
    debug: bool = False,
) -> tuple[ir.Model, dict[str, int]]:
    from onnxscript.rewriter import rewrite
    from onnxscript.rewriter.ort_fusions import (  # fused_matmul_rule_sets,; group_normalization_merge_silu,
        instance_to_group_normalization,
        softmax,
    )
    from onnxscript.rewriter.ort_fusions._core import fuse_xformers

    model, fusion_count = fuse_xformers(model, debug=debug)
    # Apply the ORT pattern rewrite rules.
    rewrite_rules = [
        *softmax.rules.rules,
        *instance_to_group_normalization.rules.rules,
        # *fused_matmul_rule_sets.fused_matmul_rule_sets(),
    ]
    rewrite(model, rewrite_rules)
    return model, fusion_count
