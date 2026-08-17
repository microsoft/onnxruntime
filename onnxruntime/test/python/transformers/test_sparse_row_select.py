# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Numeric parity for com.microsoft.SparseRowSelect against the subgraph it replaces.

The reference below is a direct transcription of ``make_indexer`` in the DeepSeek-V4 builder,
including the two places the graph rounds to the activation dtype.

The operator returns the selected rows in ascending order while ``TopK`` returns them by
descending score, so the comparison is on the selected *set* -- which is all attention over a
gather list can depend on.
"""

# Tensor dimensions keep the upper-case names the schema doc gives them.
# ruff: noqa: N806

import unittest

import numpy as np
import onnx
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


def reference(q_raw, cos, sin, rows, first_slot, last_slot, past_cache, w_raw, past_lens, cfg, dt):
    NH, HD, rd, ratio = cfg["nh"], cfg["hd"], cfg["rd"], cfg["ratio"]
    k, L, rotate = cfg["k"], cfg["L"], cfg["rotate"]
    B, S, _ = q_raw.shape
    C = past_cache.shape[1]
    nd = HD - rd
    scale = (HD**-0.5) * NH**-0.5

    q = rt(q_raw, dt).reshape(B, S, NH, HD)
    rope = q[..., nd:].clone()
    rot = torch.empty_like(rope)
    rot[..., 0::2], rot[..., 1::2] = -rope[..., 1::2], rope[..., 0::2]
    q = torch.cat([q[..., :nd], rope * cos[:, :, None] + rot * sin[:, :, None]], -1)
    if rotate:
        q = fp4_round_trip(rt(q @ hadamard(HD), dt))

    present = rt(past_cache, dt).clone()
    rows_f = rt(rows, dt)
    for b in range(B):
        first, last = int(first_slot[b, 0]), int(last_slot[b, 0])
        for c in range(first, min(last, C - 1) + 1):
            if c < 0:
                continue
            sel = c - first
            present[b, c] = rows_f[b, min(max(sel, 0), rows.shape[1] - 1)]

    # A query at absolute position past_lens[b] + s can reach only the first
    # (past_lens[b] + s + 1) / ratio rows, so scoring stops at the widest of those. Rows past
    # that point never enter a top-k and on a long-context export they are almost the whole
    # cache; dropping them from the reference is what keeps this test tractable.
    n_max = min(int((int(past_lens.max()) + S - 1 + 1) // ratio), C)
    n_max = max(n_max, 1)

    w = rt(w_raw, dt) * scale
    e = torch.einsum("bshd,btd->bsht", q, present[:, :n_max])
    score = (torch.relu(e) * w[..., None]).sum(2)

    selection = torch.full((B, S, k), -1, dtype=torch.int64)
    for b in range(B):
        for s in range(S):
            n_vis = min((int(past_lens[b]) + s + 1) // ratio, C)
            if n_vis <= k:
                chosen = list(range(n_vis))
            else:
                chosen = sorted(torch.topk(score[b, s, :n_vis], k).indices.tolist())
            for i, c in enumerate(chosen):
                selection[b, s, i] = c + L
    return selection, present


def build_graph(cfg, onnx_dt):
    NH, HD, rd, L, k, C = cfg["nh"], cfg["hd"], cfg["rd"], cfg["L"], cfg["k"], cfg["C"]
    # The T-typed inputs are fed as float and cast inside, so the harness never has to hold a
    # bfloat16 array (numpy cannot).
    names = ["query", "cos", "sin", "rows", "first_slot", "last_slot", "past_cache", "weights", "past_lens"]
    shapes = [
        ["B", "S", NH * HD],
        ["B", "S", rd],
        ["B", "S", rd],
        ["B", "J", HD],
        ["B", 1],
        ["B", 1],
        ["B", C, HD],
        ["B", "S", NH],
        ["B"],
    ]
    types = [TP.FLOAT] * 4 + [TP.INT64] * 2 + [TP.FLOAT] * 2 + [TP.INT64]
    inputs = [helper.make_tensor_value_info(n, t, s) for n, t, s in zip(names, types, shapes, strict=False)]

    nodes = []
    wired = list(names)
    for i, n in ((0, "query"), (3, "rows"), (6, "past_cache"), (7, "weights")):
        if onnx_dt != TP.FLOAT:
            nodes.append(helper.make_node("Cast", [n], [f"{n}_t"], name=f"cast_{n}", to=onnx_dt))
            wired[i] = f"{n}_t"

    nodes.append(
        helper.make_node(
            "SparseRowSelect",
            wired,
            ["selection", "present"],
            name="idx",
            domain="com.microsoft",
            num_heads=NH,
            head_dim=HD,
            rope_head_dim=rd,
            ratio=cfg["ratio"],
            topk=k,
            row_id_offset=L,
            simulate_rotated_fp4=int(cfg["rotate"]),
            scale=(HD**-0.5) * NH**-0.5,
        )
    )

    present_name = "present"
    if onnx_dt != TP.FLOAT:
        nodes.append(helper.make_node("Cast", ["present"], ["present_f"], name="castp", to=TP.FLOAT))
        present_name = "present_f"

    graph_outs = [
        helper.make_tensor_value_info("selection", TP.INT64, ["B", "S", k]),
        helper.make_tensor_value_info(present_name, TP.FLOAT, ["B", C, HD]),
    ]
    graph = helper.make_graph(nodes, "sparse_row_select", inputs, graph_outs)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    return model, present_name


DTYPES = [(None, TP.FLOAT), (torch.float16, TP.FLOAT16), (torch.bfloat16, TP.BFLOAT16)]

BATCH = 2


@unittest.skipUnless(has_cuda(), "SparseRowSelect is a CUDA-only kernel")
class TestSparseRowSelect(unittest.TestCase):
    def _run_case(self, cfg):
        torch.manual_seed(0)
        NH, HD, rd, ratio, L = cfg["nh"], cfg["hd"], cfg["rd"], cfg["ratio"], cfg["L"]
        C = cfg["C"] = L // ratio + 2
        B = BATCH
        for S, past in cfg["sp"]:
            J = (S - 1) // ratio + 2
            q_raw = torch.randn(B, S, NH * HD)
            rows = torch.randn(B, J, HD)
            past_cache = torch.randn(B, C, HD)
            w_raw = torch.randn(B, S, NH)
            past_lens = torch.tensor(past, dtype=torch.int64)
            first_slot = torch.tensor([[int(p) // ratio] for p in past], dtype=torch.int64)
            last_slot = torch.tensor([[(int(p) + S - 1) // ratio] for p in past], dtype=torch.int64)
            pos = past_lens[:, None] + torch.arange(S)
            ang = torch.outer(
                torch.arange(L, dtype=torch.float32),
                1.0 / (10000.0 ** (torch.arange(0, rd, 2).float() / rd)),
            )
            cos = ang.cos().repeat_interleave(2, -1)[pos]
            sin = ang.sin().repeat_interleave(2, -1)[pos]

            feeds = {
                "query": q_raw.numpy(),
                "cos": cos.numpy(),
                "sin": sin.numpy(),
                "rows": rows.numpy(),
                "first_slot": first_slot.numpy(),
                "last_slot": last_slot.numpy(),
                "past_cache": past_cache.numpy(),
                "weights": w_raw.numpy(),
                "past_lens": past_lens.numpy(),
            }

            for tdt, odt in DTYPES:
                tag = f"nh={NH} hd={HD} k={cfg['k']} rotate={cfg['rotate']} S={S} {onnx.TensorProto.DataType.Name(odt)}"
                with self.subTest(case=tag):
                    model, present_name = build_graph(cfg, odt)
                    sess = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
                    got_sel, got_present = sess.run(["selection", present_name], feeds)
                    exp_sel, exp_present = reference(
                        q_raw, cos, sin, rows, first_slot, last_slot, past_cache, w_raw, past_lens, cfg, tdt
                    )

                    # Slots above ``last_slot`` are unspecified by the operator: no step has ever
                    # written them and no query can reach them, so the kernel stops there rather
                    # than copying a 256K-capable cache forward every step. The live extent is
                    # still graded exactly, and it spans both the rows carried over from
                    # ``past_cache`` and the rows written this step.
                    cache_bad = 0.0
                    for b in range(B):
                        live = int(last_slot[b, 0]) + 1
                        cache_bad = max(
                            cache_bad,
                            float(np.abs(got_present[b, :live] - exp_present[b, :live].numpy()).max()),
                        )
                    self.assertEqual(cache_bad, 0.0, f"present cache differs by {cache_bad:.3e}")

                    # A row whose score sits within rounding distance of the k-th best can land
                    # on either side of the cut, so what is graded is how often that happens, not
                    # whether it ever does.
                    miss = 0
                    total = 0
                    for b in range(B):
                        for s in range(S):
                            a = {int(v) for v in got_sel[b, s] if v >= 0}
                            e = {int(v) for v in exp_sel[b, s].tolist() if v >= 0}
                            self.assertEqual(len(a), len(e), f"kept {len(a)} rows, expected {len(e)} at b={b} s={s}")
                            miss += len(e - a)
                            total += max(len(e), 1)
                    frac = miss / total
                    self.assertLessEqual(frac, 5e-3, f"selection differs on {frac:.2%} of rows")

    def test_small_topk_binds(self):
        self._run_case(
            {
                "nh": 4,
                "hd": 32,
                "rd": 16,
                "ratio": 4,
                "L": 64,
                "k": 8,
                "rotate": True,
                "sp": ((5, [0, 7]), (1, [40, 33])),
            }
        )

    def test_small_topk_exceeds_visible(self):
        self._run_case(
            {
                "nh": 4,
                "hd": 32,
                "rd": 16,
                "ratio": 4,
                "L": 64,
                "k": 18,
                "rotate": True,
                "sp": ((5, [0, 7]), (1, [40, 33])),
            }
        )

    def test_small_no_rotation(self):
        self._run_case(
            {"nh": 4, "hd": 32, "rd": 16, "ratio": 4, "L": 64, "k": 8, "rotate": False, "sp": ((1, [40, 33]),)}
        )

    # The shapes DeepSeek-V4-Flash actually uses, at a context long enough that the top-k
    # binds.  S=64 is above the operator's fused-scorer cutoff, so it is what covers the
    # cuBLAS scoring path; the shorter steps take the fused one.
    def test_model_shapes(self):
        self._run_case(
            {
                "nh": 64,
                "hd": 128,
                "rd": 64,
                "ratio": 4,
                "L": 4096,
                "k": 512,
                "rotate": True,
                "sp": ((1, [4000, 4003]), (8, [3000, 2000]), (64, [3000, 2000])),
            }
        )

    # A long-context export serving a short prompt: 65,538 rows of capacity against a few
    # hundred live ones.  That gap is what the operator's scoring clamp removes, so without a
    # case here the clamp is never exercised and a bad bound reads as a pass.
    def test_long_context_capacity_clamp(self):
        self._run_case(
            {
                "nh": 64,
                "hd": 128,
                "rd": 64,
                "ratio": 4,
                "L": 262144,
                "k": 512,
                "rotate": True,
                "sp": ((8, [0, 0]), (8, [16384, 8192])),
            }
        )


if __name__ == "__main__":
    unittest.main()
