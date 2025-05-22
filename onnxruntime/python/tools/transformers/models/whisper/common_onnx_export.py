# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from typing import ClassVar

import torch

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

            def qkv_attention(self, q, k, v, mask=None):
                # q.shape == k.shape == v.shape == torch.Size([2, 2, 384])
                # mask.shape == torch.Size([448, 448])
                q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
                k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
                v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

                # a = scaled_dot_product_attention(q, k, v, is_causal=mask is not None and n_ctx > 1)
                if torch.compiler.is_exporting():
                    # torchscript is only tracing one branch, even if does not match
                    # the implementation of oepnai-whisper, it matches what needs to be exported.
                    a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                else:
                    # NOTE: dynamo intends to skip tracing the function
                    # but it's a false alarm, so we need to use
                    # dont_skip_tracing to force tracing
                    @torch._dynamo.dont_skip_tracing
                    def _true_qkv_attentation(q, k, v):
                        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
                        return a

                    @torch._dynamo.dont_skip_tracing
                    def _false_qkv_attentation(q, k, v):
                        a = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
                        return a

                    a = torch.cond(
                        (q.shape[2] > torch.tensor(1, device=q.device)),
                        _true_qkv_attentation,
                        _false_qkv_attentation,
                        [q, k, v],
                    )
                out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
                qk = None
                # out.shape == torch.Size([2, 2, 384])
                return out, qk

        # The patch does this:
        # whisper.model.MultiHeadAttention.qkv_attention = PatchedMultiHeadAttention.qkv_attention
        _custom_patches.append(PatchedMultiHeadAttention)
