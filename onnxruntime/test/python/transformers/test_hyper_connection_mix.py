# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Numeric check for com.microsoft.HyperConnectionMix against a torch reference.

The reference mirrors the unrolled subgraph the builder used to emit:
``make_hc_post`` -> ``make_hc_pre`` -> ``make_rmsnorm``, including the two
activation-dtype round trips, so the comparison is meant to be tight.
"""

# Tensor dimensions keep the upper-case names the schema doc gives them.
# ruff: noqa: N806

import unittest

import numpy as np
import torch
from onnx import TensorProto as TP  # noqa: N817
from onnx import helper

import onnxruntime as ort


def has_cuda():
    return "CUDAExecutionProvider" in ort.get_available_providers()


def build_graph(io_elem, hc, dim, iters, eps, hc_eps, sk_eps, post_alpha):
    mix = (2 + hc) * hc
    node = helper.make_node(
        "HyperConnectionMix",
        ["x", "residual", "post_mix", "comb_mix", "fn", "scale", "base", "norm_weight"],
        ["residual_out", "post_mix_out", "comb_mix_out", "layer_input"],
        domain="com.microsoft",
        sinkhorn_iterations=iters,
        epsilon=eps,
        hc_epsilon=hc_eps,
        sinkhorn_epsilon=sk_eps,
        post_alpha=post_alpha,
    )
    F = TP.FLOAT
    nodes = [node]
    out_elem = io_elem
    if io_elem == TP.BFLOAT16:
        # numpy cannot hold bfloat16, so widen the two activation outputs in-graph.
        nodes += [
            helper.make_node("Cast", [n], [n + "_f"], to=F, name=n + "/cast") for n in ("residual_out", "layer_input")
        ]
        out_elem = F
    ro = "residual_out_f" if io_elem == TP.BFLOAT16 else "residual_out"
    li = "layer_input_f" if io_elem == TP.BFLOAT16 else "layer_input"
    g = helper.make_graph(
        nodes,
        "hcmix",
        [
            helper.make_tensor_value_info("x", io_elem, ["B", "S", dim]),
            helper.make_tensor_value_info("residual", io_elem, ["B", "S", hc, dim]),
            helper.make_tensor_value_info("post_mix", F, ["B", "S", hc]),
            helper.make_tensor_value_info("comb_mix", F, ["B", "S", hc, hc]),
            helper.make_tensor_value_info("fn", F, [hc * dim, mix]),
            helper.make_tensor_value_info("scale", F, [3]),
            helper.make_tensor_value_info("base", F, [mix]),
            helper.make_tensor_value_info("norm_weight", F, [dim]),
        ],
        [
            helper.make_tensor_value_info(ro, out_elem, ["B", "S", hc, dim]),
            helper.make_tensor_value_info("post_mix_out", F, ["B", "S", hc]),
            helper.make_tensor_value_info("comb_mix_out", F, ["B", "S", hc, hc]),
            helper.make_tensor_value_info(li, out_elem, ["B", "S", dim]),
        ],
    )
    m = helper.make_model(
        g,
        opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)],
    )
    m.ir_version = 10
    return m


def sinkhorn(y, iters, eps):
    def norm(t, axis):
        return t / (t.sum(axis, keepdim=True) + eps)

    y = norm(y, -2)
    for _ in range(iters - 1):
        y = norm(norm(y, -1), -2)
    return y


def reference(
    x,
    residual,
    post_mix,
    comb_mix,
    fn,
    scale,
    base,
    norm_weight,
    io_dtype,
    hc,
    dim,
    iters,
    eps,
    hc_eps,
    sk_eps,
    post_alpha,
):
    # --- make_hc_post -----------------------------------------------------
    term1 = post_mix.unsqueeze(-1) * x.float().unsqueeze(2)  # [B,S,hc,dim]
    term2 = torch.einsum("bsgh,bsgd->bshd", comb_mix, residual.float())
    residual_out = (term1 + term2).to(io_dtype)

    # --- make_hc_pre ------------------------------------------------------
    xf = residual_out.float()
    xflat = xf.reshape(*xf.shape[:2], hc * dim)
    rs = torch.rsqrt(xflat.pow(2).mean(-1, keepdim=True) + eps)
    mixes = (xflat @ fn) * rs  # [B,S,mix]

    pre = torch.sigmoid(mixes[..., :hc] * scale[0] + base[:hc]) + hc_eps
    post_out = torch.sigmoid(mixes[..., hc : 2 * hc] * scale[1] + base[hc : 2 * hc]) * post_alpha
    comb = (mixes[..., 2 * hc :] * scale[2] + base[2 * hc :]).reshape(*xf.shape[:2], hc, hc)
    comb = torch.softmax(comb, -1) + hc_eps
    comb_out = sinkhorn(comb, iters, sk_eps)

    y = (pre.unsqueeze(-1) * xf).sum(2).to(io_dtype)  # [B,S,dim]

    # --- make_rmsnorm -----------------------------------------------------
    yf = y.float()
    li = (yf * torch.rsqrt(yf.pow(2).mean(-1, keepdim=True) + eps) * norm_weight).to(io_dtype)
    return residual_out, post_out, comb_out, li


@unittest.skipUnless(has_cuda(), "HyperConnectionMix is a CUDA-only kernel")
class TestHyperConnectionMix(unittest.TestCase):
    def _run(
        self,
        np_dtype,
        io_elem,
        hc=4,
        dim=256,
        B=2,
        S=3,
        iters=20,
        eps=1e-6,
        hc_eps=1e-6,
        post_alpha=2.0,
        tol=None,
    ):
        mix = (2 + hc) * hc
        g = torch.Generator().manual_seed(0)
        t = {"generator": g, "dtype": torch.float32}
        x = torch.randn(B, S, dim, **t)
        residual = torch.randn(B, S, hc, dim, **t)
        post_mix = torch.rand(B, S, hc, **t) + 0.2
        comb_mix = torch.rand(B, S, hc, hc, **t) + 0.2
        fn = torch.randn(hc * dim, mix, **t) * (1.0 / (hc * dim) ** 0.5)
        scale = torch.rand(3, **t) + 0.5
        base = torch.randn(mix, **t) * 0.1
        norm_weight = torch.rand(dim, **t) + 0.5

        torch_dtype = {np.float32: torch.float32, np.float16: torch.float16}.get(np_dtype)
        if torch_dtype is None:
            torch_dtype = torch.bfloat16
        x = x.to(torch_dtype)
        residual = residual.to(torch_dtype)

        ref = reference(
            x,
            residual,
            post_mix,
            comb_mix,
            fn,
            scale,
            base,
            norm_weight,
            torch_dtype,
            hc,
            dim,
            iters,
            eps,
            hc_eps,
            hc_eps,
            post_alpha,
        )

        model = build_graph(io_elem, hc, dim, iters, eps, hc_eps, hc_eps, post_alpha)
        sess = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])

        def _bf16(n):
            u = np.ascontiguousarray(n, dtype=np.float32).view(np.uint32)
            return ((u + 0x8000 + ((u >> 16) & 1)) >> 16).astype(np.uint16)

        def to_ort(v):
            n = v.float().numpy()
            return _bf16(n) if v.dtype is torch.bfloat16 else n.astype(np_dtype)

        feeds = {
            "x": to_ort(x),
            "residual": to_ort(residual),
            "post_mix": post_mix.numpy(),
            "comb_mix": comb_mix.numpy(),
            "fn": fn.numpy(),
            "scale": scale.numpy(),
            "base": base.numpy(),
            "norm_weight": norm_weight.numpy(),
        }
        if torch_dtype is torch.bfloat16:
            maker = getattr(ort.OrtValue, "ortvalue_from_numpy_with_onnx_type", None)
            if maker is None:
                self.skipTest("this onnxruntime build cannot feed BFLOAT16 from numpy")
            io = ort.IOBinding(sess)
            for k, v in feeds.items():
                elem = io_elem if k in ("x", "residual") else TP.FLOAT
                io.bind_ortvalue_input(k, maker(v, elem))
            for name in [o.name for o in sess.get_outputs()]:
                io.bind_output(name, "cpu")
            sess.run_with_iobinding(io)
            outs = [o.numpy() for o in io.get_outputs()]
        else:
            outs = sess.run(None, feeds)

        names = ["residual_out", "post_mix_out", "comb_mix_out", "layer_input"]
        for name, raw, want in zip(names, outs, ref, strict=False):
            widened = (raw.astype(np.uint32) << 16).view(np.float32) if raw.dtype == np.uint16 else raw
            got = np.asarray(widened, dtype=np.float32)
            w = want.float().numpy()
            d = np.abs(got - w)
            denom = max(np.abs(w).max(), 1e-6)
            rel = d.max() / denom
            lim = tol if tol is not None else (2e-6 if np_dtype is np.float32 else 6e-3)
            self.assertLessEqual(rel, lim, f"{name}: max|d|={d.max():.3e} rel={rel:.3e}")

    def _run_all_hc(self, np_dtype, io_elem):
        for hc in (1, 2, 3, 4):
            with self.subTest(hc=hc):
                self._run(np_dtype, io_elem, hc=hc)

    def test_float32(self):
        self._run_all_hc(np.float32, TP.FLOAT)

    def test_float16(self):
        self._run_all_hc(np.float16, TP.FLOAT16)

    def test_bfloat16(self):
        self._run_all_hc(None, TP.BFLOAT16)

    def test_odd_dim_with_64_tokens(self):
        self._run(np.float32, TP.FLOAT, hc=3, dim=257, B=2, S=32)

    def test_token_count_beyond_grid_y_limit(self):
        self._run(np.float32, TP.FLOAT, hc=3, dim=8, B=256, S=256)


if __name__ == "__main__":
    unittest.main()
