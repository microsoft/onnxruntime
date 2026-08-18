# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Numeric parity for com.microsoft.GatedLatentPool against the subgraph it replaces.

The reference below is a direct transcription of ``make_compressor`` in the DeepSeek-V4
builder, including the two places the graph rounds to the activation dtype.
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


def rt(x, dt):
    """The activation round trip the graph performs with a pair of Cast nodes."""
    return x.to(dt).to(torch.float32) if dt is not None else x


def block_scale(x, block, limit, floor):
    a = x.reshape(-1, block).abs().amax(-1, keepdim=True)
    r = (a / limit).clamp(min=floor)
    s = torch.pow(torch.tensor(2.0), torch.ceil(torch.log(r) / float(np.log(2.0))))
    return torch.where(a > 0, s, torch.ones_like(s))


def fp8_round_trip(x, block=64):
    s = block_scale(x, block, 448.0, 1e-30)
    q = (x.reshape(-1, block) / s).clamp(-448.0, 448.0)
    q = q.to(torch.float8_e4m3fn).to(torch.float32)
    return (q * s).reshape(x.shape)


def fp4_round_trip(x, block=32):
    s = block_scale(x, block, 6.0, 1e-38)
    v = (x.reshape(-1, block) / s).clamp(-6.0, 6.0)
    u = v.abs()
    step = torch.where(u < 2.0, 0.5, torch.where(u < 4.0, 1.0, 2.0))
    q = torch.sign(v) * step * torch.ceil(u / step - 0.5)
    return (q * s).reshape(x.shape)


def hadamard(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h * (n**-0.5)


def reference(kv, sc, pkv, psc, ape, nw, cos, sin, past_lens, cfg, dt):
    r, co, d, rd, L, eps = cfg["r"], cfg["co"], cfg["d"], cfg["rd"], cfg["L"], cfg["eps"]
    quant, rotate = cfg["quant"], cfg["rotate"]
    B, S, _ = kv.shape
    span = co * r
    nd = d - rd
    J = (S - 1) // r + 2
    full_kv = torch.cat([pkv, kv], 1)
    full_sc = torch.cat([psc, sc], 1)
    n_full = span + S
    H = hadamard(d)

    rows = torch.zeros(B, J, d)
    for b in range(B):
        past = int(past_lens[b])
        total, base, first = past + S, past - span, past // r
        for j in range(J):
            pos0 = (first + j) * r
            pool_kv = torch.zeros(span, d)
            pool_sc = torch.zeros(span, d)
            valid = torch.zeros(span, dtype=torch.bool)
            for s in range(span):
                if co == 2 and s < r:
                    k, pos, fo, ok = s, pos0 + s - r, 0, pos0 + s - r >= 0
                elif co == 2:
                    k, pos, fo, ok = s - r, pos0 + s - r, d, True
                else:
                    k, pos, fo, ok = s, pos0 + s, 0, True
                idx = pos - base
                valid[s] = ok and 0 <= idx < n_full and pos < total
                cl = min(max(idx, 0), n_full - 1)
                pool_kv[s] = full_kv[b, cl, fo : fo + d]
                pool_sc[s] = full_sc[b, cl, fo : fo + d] + ape[k, fo : fo + d]

            m = torch.where(valid[:, None], pool_sc, torch.tensor(-1e30))
            w = torch.softmax(m, 0)
            pooled = rt((pool_kv * w).sum(0), dt)

            normed = pooled * torch.rsqrt(pooled.pow(2).mean() + eps) * nw
            nope, rope = normed[:nd].clone(), normed[nd:].clone()
            rot = torch.empty_like(rope)
            rot[0::2], rot[1::2] = -rope[1::2], rope[0::2]
            ps = min(max(pos0, 0), L - 1)
            rope = rope * cos[ps] + rot * sin[ps]
            if quant:
                nope = fp8_round_trip(nope)
            row = torch.cat([nope, rope])
            if rotate:
                row = fp4_round_trip(rt(row @ H, dt))
            rows[b, j] = rt(row, dt)

    full = torch.cat([pkv, kv], 1)
    full_s = torch.cat([psc, sc], 1)
    first_slot = torch.tensor([[int(p) // r] for p in past_lens], dtype=torch.int64)
    last_slot = torch.tensor([[(int(p) + S - 1) // r] for p in past_lens], dtype=torch.int64)
    return (
        rows,
        first_slot,
        last_slot,
        torch.tensor(J),
        full[:, S : S + span],
        full_s[:, S : S + span],
    )


def build_graph(cfg, onnx_dt, proj_dt=TP.FLOAT, fused=False):
    r, co, d, rd, L = cfg["r"], cfg["co"], cfg["d"], cfg["rd"], cfg["L"]
    span, feat = co * r, co * d
    names = ["kv", "sc", "pkv", "psc", "ape", "nw", "cos", "sin", "past_lens"]
    shapes = [
        ["B", "S", feat],
        ["B", "S", feat],
        ["B", span, feat],
        ["B", span, feat],
        [r, feat],
        [d],
        [L, rd],
        [L, rd],
        ["B"],
    ]
    types = [TP.FLOAT] * 8 + [TP.INT64]
    inputs = [helper.make_tensor_value_info(n, t, s) for n, t, s in zip(names, types, shapes, strict=False)]

    nodes = []
    names = list(names)
    if fused:
        # One GEMM's worth of output: kv in the low half of each row, the gate in the high half.
        nodes.append(helper.make_node("Concat", ["kv", "sc"], ["kvsc"], name="cat", axis=-1))
        names[0], names[1] = "kvsc", ""
    if proj_dt != TP.FLOAT:
        # numpy cannot feed bfloat16, so the narrowing the real graph's MatMul does is
        # reproduced with a Cast the kernel then reads through.
        for i in (0, 1):
            if not names[i]:
                continue
            nodes.append(helper.make_node("Cast", [names[i]], [f"{names[i]}_p"], name=f"cast{names[i]}", to=proj_dt))
            names[i] = f"{names[i]}_p"

    outs = ["rows", "first_slot", "last_slot", "row_count", "state_kv", "state_sc"]
    node = helper.make_node(
        "GatedLatentPool",
        names,
        outs,
        name="comp",
        domain="com.microsoft",
        ratio=r,
        window_multiplier=co,
        head_dim=d,
        rope_head_dim=rd,
        max_seq_len=L,
        epsilon=cfg["eps"],
        simulate_fp8=int(cfg["quant"]),
        simulate_rotated_fp4=int(cfg["rotate"]),
        dtype=int(onnx_dt),
    )
    nodes.append(node)
    rows_name = "rows"
    if onnx_dt == TP.BFLOAT16:  # numpy cannot hold bfloat16
        nodes.append(helper.make_node("Cast", ["rows"], ["rows_f"], name="castrows", to=TP.FLOAT))
        rows_name = "rows_f"

    graph_outs = [
        helper.make_tensor_value_info(rows_name, TP.FLOAT if onnx_dt == TP.BFLOAT16 else onnx_dt, ["B", "J", d]),
        helper.make_tensor_value_info("first_slot", TP.INT64, ["B", 1]),
        helper.make_tensor_value_info("last_slot", TP.INT64, ["B", 1]),
        helper.make_tensor_value_info("row_count", TP.INT64, []),
        helper.make_tensor_value_info("state_kv", TP.FLOAT, ["B", span, feat]),
        helper.make_tensor_value_info("state_sc", TP.FLOAT, ["B", span, feat]),
    ]
    graph = helper.make_graph(nodes, "gated_latent_pool", inputs, graph_outs)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    return model, rows_name


# 1e-5 for float32: the ratio-128 windows sum 128 terms, and the reference sums them in a
# different order.
DTYPES = [
    (None, TP.FLOAT, 1e-5),
    (torch.float16, TP.FLOAT16, 6e-3),
    (torch.bfloat16, TP.BFLOAT16, 6e-3),
]

PROJECTIONS = [(TP.FLOAT, False, "f32"), (TP.BFLOAT16, False, "bf16"), (TP.BFLOAT16, True, "bf16+cat")]

BATCH = 2


@unittest.skipUnless(has_cuda(), "GatedLatentPool is a CUDA-only kernel")
class TestGatedLatentPool(unittest.TestCase):
    def _run_case(self, cfg):
        torch.manual_seed(0)
        span = cfg["co"] * cfg["r"]
        feat, d, rd, L = cfg["co"] * cfg["d"], cfg["d"], cfg["rd"], cfg["L"]
        B = BATCH
        for S, past in cfg.get("sp", ((7, [0, 5]), (1, [11, 3]))):
            kv = torch.randn(B, S, feat)
            sc = torch.randn(B, S, feat)
            pkv = torch.randn(B, span, feat)
            psc = torch.randn(B, span, feat)
            ape = torch.randn(cfg["r"], feat) * 0.5
            nw = torch.randn(d).abs() + 0.5
            ang = torch.outer(
                torch.arange(L, dtype=torch.float32),
                1.0 / (10000.0 ** (torch.arange(0, rd, 2).float() / rd)),
            )
            cos = ang.cos().repeat_interleave(2, -1)
            sin = ang.sin().repeat_interleave(2, -1)
            past_lens = torch.tensor(past, dtype=torch.int64)

            feeds = {
                "kv": kv.numpy(),
                "sc": sc.numpy(),
                "pkv": pkv.numpy(),
                "psc": psc.numpy(),
                "ape": ape.numpy(),
                "nw": nw.numpy(),
                "cos": cos.numpy(),
                "sin": sin.numpy(),
                "past_lens": past_lens.numpy(),
            }
            # The reference depends on the activation dtype and on how the projection was
            # narrowed, but not on whether the two halves arrived fused, so it is shared.
            expected = {}
            for tdt, odt, tol in DTYPES:
                for pdt, fused, pname in PROJECTIONS:
                    model, rows_name = build_graph(cfg, odt, pdt, fused)
                    names = [rows_name, "first_slot", "last_slot", "row_count", "state_kv", "state_sc"]
                    sess = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
                    got = sess.run(names, feeds)

                    key = (tdt, pdt)
                    if key not in expected:
                        pkv_in, psc_in = (
                            (kv, sc) if pdt == TP.FLOAT else (rt(kv, torch.bfloat16), rt(sc, torch.bfloat16))
                        )
                        expected[key] = reference(pkv_in, psc_in, pkv, psc, ape, nw, cos, sin, past_lens, cfg, tdt)
                    exp = expected[key]

                    tag = (
                        f"r={cfg['r']} co={cfg['co']} quant={cfg['quant']} "
                        f"rotate={cfg['rotate']} S={S} proj={pname} dtype={odt}"
                    )
                    with self.subTest(case=tag):
                        for i, name in enumerate(
                            ["first_slot", "last_slot", "row_count", "state_kv", "state_sc"], start=1
                        ):
                            np.testing.assert_allclose(
                                got[i], exp[i].numpy(), atol=1e-6, rtol=0, err_msg=f"{name} mismatch"
                            )
                        # The FP8/FP4 grids are step functions, so a 1-ULP difference anywhere
                        # upstream moves a value a whole step. The kernel cannot be bit-identical
                        # to a reference built from cuBLAS GEMMs and ORT's own tree reductions, so
                        # what matters is that such flips stay rare rather than that the max
                        # difference is small.
                        ad = np.abs(got[0].astype(np.float32) - exp[0].numpy())
                        frac = float((ad > tol).mean())
                        self.assertLessEqual(
                            frac,
                            5e-3,
                            f"{frac * 100:.3f}% of rows differ by more than {tol} (max |d| = {ad.max():.3e})",
                        )

    def test_ratio4_window2_fp8(self):
        self._run_case({"r": 4, "co": 2, "d": 128, "rd": 64, "L": 64, "eps": 1e-6, "quant": True, "rotate": False})

    def test_odd_rope_head_dim_rejected(self):
        cfg = {"r": 4, "co": 1, "d": 128, "rd": 63, "L": 64, "eps": 1e-6, "quant": False, "rotate": False}
        model, _ = build_graph(cfg, TP.FLOAT)
        with self.assertRaisesRegex(Exception, "rope_head_dim must be even"):
            ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])

    def test_ratio8_window1_fp8(self):
        self._run_case({"r": 8, "co": 1, "d": 128, "rd": 64, "L": 64, "eps": 1e-6, "quant": True, "rotate": False})

    def test_ratio4_window2_rotated_fp4(self):
        self._run_case({"r": 4, "co": 2, "d": 128, "rd": 64, "L": 64, "eps": 1e-6, "quant": False, "rotate": True})

    # The shapes DeepSeek-V4-Flash actually uses, at a context long enough that the rolling
    # state is full and the window sits far from the origin.
    def test_model_shapes_ratio4_window2_fp8(self):
        self._run_case(
            {
                "r": 4,
                "co": 2,
                "d": 512,
                "rd": 64,
                "L": 4096,
                "eps": 1e-6,
                "quant": True,
                "rotate": False,
                "sp": ((1, [1000, 1001]), (1024, [0, 3])),
            }
        )

    def test_model_shapes_ratio128_window1_fp8(self):
        self._run_case(
            {
                "r": 128,
                "co": 1,
                "d": 512,
                "rd": 64,
                "L": 4096,
                "eps": 1e-6,
                "quant": True,
                "rotate": False,
                "sp": ((1, [1000, 1001]), (1024, [0, 3])),
            }
        )

    def test_model_shapes_rotated_fp4(self):
        self._run_case(
            {
                "r": 4,
                "co": 2,
                "d": 128,
                "rd": 64,
                "L": 4096,
                "eps": 1e-6,
                "quant": False,
                "rotate": True,
                "sp": ((1, [1000, 1001]), (1024, [0, 3])),
            }
        )


if __name__ == "__main__":
    unittest.main()
