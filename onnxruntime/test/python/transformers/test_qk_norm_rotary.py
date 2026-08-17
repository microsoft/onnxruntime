# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Numeric parity for the fused LLM building blocks in com.microsoft:

  SwiGLU, SinkhornNormalize, QKNormRotaryEmbedding, InverseRotaryRegroup, MoERouter

Every reference here is a transcription of the schema doc in contrib_defs.cc, including the
places the fused kernel rounds to the activation dtype. The T-typed inputs are fed as float
and cast inside the graph, so the harness never has to hold a bfloat16 numpy array.
"""

# Tensor dimensions keep the upper-case names the schema doc gives them.

import unittest

import numpy as np
import torch
from onnx import TensorProto as TP  # noqa: N817
from onnx import helper

import onnxruntime as ort


def has_cuda():
    return "CUDAExecutionProvider" in ort.get_available_providers()


TORCH_OF = {TP.FLOAT: torch.float32, TP.FLOAT16: torch.float16, TP.BFLOAT16: torch.bfloat16}
NAME_OF = {TP.FLOAT: "float32", TP.FLOAT16: "float16", TP.BFLOAT16: "bfloat16"}
ELEM_TYPES = [TP.FLOAT, TP.FLOAT16, TP.BFLOAT16]


def rt(x, elem):
    """The round trip a Cast pair to the activation dtype performs."""
    return x.to(TORCH_OF[elem]).to(torch.float32)


def cast_in(nodes, name, elem):
    """Narrow a float graph input to the activation dtype the kernel is registered for."""
    if elem == TP.FLOAT:
        return name
    nodes.append(helper.make_node("Cast", [name], [name + "_t"], to=elem, name="cast_in_" + name))
    return name + "_t"


def cast_out(nodes, name, elem):
    if elem == TP.FLOAT:
        return name
    nodes.append(helper.make_node("Cast", [name], [name + "_f"], to=TP.FLOAT, name="cast_out_" + name))
    return name + "_f"


def make_model(nodes, inputs, outputs, name):
    graph = helper.make_graph(nodes, name, inputs, outputs)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    model.ir_version = 10
    return model


def run(model, feeds, out_names=None):
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CUDAExecutionProvider"])
    return sess.run(out_names, feeds)


def block_scale(x, block, limit, floor):
    """The graph's Log/Div/Ceil/Pow chain: a power-of-two scale per block of `block`."""
    a = x.reshape(-1, block).abs().amax(-1, keepdim=True)
    r = (a / limit).clamp(min=floor)
    s = torch.pow(torch.tensor(2.0), torch.ceil(torch.log(r) / float(np.log(2.0))))
    return torch.where(a > 0, s, torch.ones_like(s))


def fp8_round_trip(x, block=64):
    s = block_scale(x, block, 448.0, 1e-30)
    q = (x.reshape(-1, block) / s).clamp(-448.0, 448.0)
    q = q.to(torch.float8_e4m3fn).to(torch.float32)
    return (q * s).reshape(x.shape)


# Tolerances: both sides round to the same activation dtype, so the only gap is the fp32
# intermediate, and a couple of ULPs of the output dtype covers it.
TOL = {TP.FLOAT: 2e-5, TP.FLOAT16: 6e-3, TP.BFLOAT16: 4e-2}


class FusedOpTestCase(unittest.TestCase):
    def assert_close(self, got, want, tol, tag, max_frac=0.0):
        got = np.asarray(got, dtype=np.float32)
        want = np.asarray(want, dtype=np.float32)
        self.assertEqual(got.shape, want.shape, f"{tag}: shape {got.shape} != {want.shape}")
        ad = np.abs(got - want)
        if max_frac <= 0.0:
            self.assertLessEqual(float(ad.max()), tol, f"{tag}: max |d| = {ad.max():.3e}")
            return
        # A simulated FP8 grid is a step function, so a 1-ULP difference in the reduction that
        # feeds it moves a value a whole step. Rare flips are expected; a systematic error is
        # not, which is what the mean guards.
        frac = float((ad > tol).mean())
        self.assertLessEqual(frac, max_frac, f"{tag}: {frac * 100:.3f}% over {tol} (max {ad.max():.3e})")
        self.assertLessEqual(float(ad.mean()), tol, f"{tag}: mean |d| = {ad.mean():.3e}")


# --------------------------------------------------------------------------------------
# QKNormRotaryEmbedding
# --------------------------------------------------------------------------------------


def rope_forward(x, cos, sin):
    """x[nope + t] = x[t] * cos[t] + (t odd ? x[t - 1] : -x[t + 1]) * sin[t]."""
    rot = torch.empty_like(x)
    rot[..., 0::2] = -x[..., 1::2]
    rot[..., 1::2] = x[..., 0::2]
    return x * cos + rot * sin


def rope_inverse(x, cos, sin):
    """The signed swap flips sign, which is the transpose of the forward rotation."""
    rot = torch.empty_like(x)
    rot[..., 0::2] = x[..., 1::2]
    rot[..., 1::2] = -x[..., 0::2]
    return x * cos + rot * sin


def qk_norm_rotary_reference(query, kv, kv_weight, cos, sin, cfg, elem):
    nh, hd, rd, eps = cfg["nh"], cfg["hd"], cfg["rd"], cfg["eps"]
    nope = hd - rd
    b, s, _ = query.shape

    q = rt(query, elem).reshape(b, s, nh, hd)
    # The graph rounds the reciprocal to T and then multiplies in T; both roundings are kept.
    rs = rt(torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps), elem)
    q = rt(q * rs, elem)
    if rd:
        q = torch.cat([q[..., :nope], rope_forward(q[..., nope:], cos[:, :, None], sin[:, :, None])], -1)
    q = rt(q, elem)

    k = rt(kv, elem)
    # The norm itself stays in fp32; the cast that follows it does not.
    k = rt(k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps) * kv_weight, elem)
    if rd:
        k = torch.cat([k[..., :nope], rope_forward(k[..., nope:], cos, sin)], -1)
    if cfg["fp8"]:
        k = torch.cat([rt(fp8_round_trip(k[..., :nope].contiguous()), elem), k[..., nope:]], -1)
    return q, rt(k, elem)


def build_qk_norm_rotary(cfg, elem, b, s):
    nh, hd, rd = cfg["nh"], cfg["hd"], cfg["rd"]
    nodes = []
    inputs = [
        helper.make_tensor_value_info("query", TP.FLOAT, [b, s, nh * hd]),
        helper.make_tensor_value_info("kv", TP.FLOAT, [b, s, hd]),
        helper.make_tensor_value_info("kv_weight", TP.FLOAT, [hd]),
        helper.make_tensor_value_info("cos", TP.FLOAT, [b, s, rd]),
        helper.make_tensor_value_info("sin", TP.FLOAT, [b, s, rd]),
    ]
    nodes.append(
        helper.make_node(
            "QKNormRotaryEmbedding",
            [cast_in(nodes, "query", elem), cast_in(nodes, "kv", elem), "kv_weight", "cos", "sin"],
            ["query_out", "kv_out"],
            name="qknorm",
            domain="com.microsoft",
            num_heads=nh,
            head_dim=hd,
            rope_head_dim=rd,
            epsilon=cfg["eps"],
            simulate_fp8=int(cfg["fp8"]),
        )
    )
    q_out = cast_out(nodes, "query_out", elem)
    k_out = cast_out(nodes, "kv_out", elem)
    outputs = [
        helper.make_tensor_value_info(q_out, TP.FLOAT, [b, s, nh, hd]),
        helper.make_tensor_value_info(k_out, TP.FLOAT, [b, s, hd]),
    ]
    return make_model(nodes, inputs, outputs, "qk_norm_rotary"), [q_out, k_out]


def rotary_tables(tokens, rd, offset=0):
    ang = torch.outer(
        torch.arange(offset, offset + tokens, dtype=torch.float32),
        1.0 / (10000.0 ** (torch.arange(0, rd, 2).float() / rd)),
    )
    return ang.cos().repeat_interleave(2, -1), ang.sin().repeat_interleave(2, -1)


@unittest.skipUnless(has_cuda(), "QKNormRotaryEmbedding is a CUDA-only kernel")
class TestQKNormRotaryEmbedding(FusedOpTestCase):
    def _run(self, elem, fp8, rd=64):
        torch.manual_seed(0)
        b, s = 2, 3
        cfg = {"nh": 8, "hd": 128, "rd": rd, "eps": 1e-6, "fp8": fp8}
        query = torch.randn(b, s, cfg["nh"] * cfg["hd"])
        kv = torch.randn(b, s, cfg["hd"])
        kv_weight = torch.rand(cfg["hd"]) + 0.5
        cos, sin = rotary_tables(b * s, rd) if rd else (torch.zeros(b * s, 0), torch.zeros(b * s, 0))
        cos = cos.reshape(b, s, rd)
        sin = sin.reshape(b, s, rd)

        model, names = build_qk_norm_rotary(cfg, elem, b, s)
        feeds = {
            "query": query.numpy(),
            "kv": kv.numpy(),
            "kv_weight": kv_weight.numpy(),
            "cos": cos.numpy(),
            "sin": sin.numpy(),
        }
        got_q, got_k = run(model, feeds, names)
        want_q, want_k = qk_norm_rotary_reference(query, kv, kv_weight, cos, sin, cfg, elem)

        tag = f"qknorm fp8={fp8} rd={rd} {NAME_OF[elem]}"
        self.assert_close(got_q, want_q.numpy(), TOL[elem], tag + " query")
        self.assert_close(got_k, want_k.numpy(), TOL[elem], tag + " kv", max_frac=5e-3 if fp8 else 0.0)

    def test_without_fp8(self):
        for elem in ELEM_TYPES:
            with self.subTest(dtype=NAME_OF[elem]):
                self._run(elem, fp8=False)

    def test_with_fp8(self):
        for elem in ELEM_TYPES:
            with self.subTest(dtype=NAME_OF[elem]):
                self._run(elem, fp8=True)

    def test_no_rotary_slice(self):
        for elem in ELEM_TYPES:
            with self.subTest(dtype=NAME_OF[elem]):
                self._run(elem, fp8=False, rd=0)

    def test_query_rows_are_unit_rms(self):
        """The query norm is weightless, so every head comes out with unit mean square."""
        torch.manual_seed(2)
        b, s = 1, 4
        cfg = {"nh": 4, "hd": 128, "rd": 0, "eps": 1e-6, "fp8": False}
        query = torch.randn(b, s, cfg["nh"] * cfg["hd"]) * 7.0
        model, names = build_qk_norm_rotary(cfg, TP.FLOAT, b, s)
        got = run(
            model,
            {
                "query": query.numpy(),
                "kv": torch.randn(b, s, cfg["hd"]).numpy(),
                "kv_weight": torch.ones(cfg["hd"]).numpy(),
                "cos": np.zeros((b, s, 0), dtype=np.float32),
                "sin": np.zeros((b, s, 0), dtype=np.float32),
            },
            names,
        )[0]
        ms = (got.astype(np.float64) ** 2).mean(-1)
        np.testing.assert_allclose(ms, 1.0, atol=1e-5)


# --------------------------------------------------------------------------------------
# InverseRotaryRegroup
# --------------------------------------------------------------------------------------


def inverse_rotary_regroup_reference(x, cos, sin, cfg, elem):
    nh, hd, rd, groups = cfg["nh"], cfg["hd"], cfg["rd"], cfg["groups"]
    nope = hd - rd
    tokens = x.shape[0]
    v = rt(x, elem).reshape(tokens, nh, hd)
    if rd:
        v = torch.cat([v[..., :nope], rope_inverse(v[..., nope:], cos[:, None, :], sin[:, None, :])], -1)
    # reshape / transpose / reshape: both views index the same flat channel h * head_dim + c.
    v = v.reshape(tokens, groups, nh * hd // groups).permute(1, 0, 2).contiguous()
    return rt(v, elem)


def build_inverse_rotary_regroup(cfg, elem, tokens):
    nh, hd, rd, groups = cfg["nh"], cfg["hd"], cfg["rd"], cfg["groups"]
    nodes = []
    inputs = [
        helper.make_tensor_value_info("x", TP.FLOAT, [tokens, nh * hd]),
        helper.make_tensor_value_info("cos", TP.FLOAT, [tokens, rd]),
        helper.make_tensor_value_info("sin", TP.FLOAT, [tokens, rd]),
    ]
    nodes.append(
        helper.make_node(
            "InverseRotaryRegroup",
            [cast_in(nodes, "x", elem), "cos", "sin"],
            ["y"],
            name="invrope",
            domain="com.microsoft",
            num_heads=nh,
            head_dim=hd,
            rope_head_dim=rd,
            num_groups=groups,
        )
    )
    out = cast_out(nodes, "y", elem)
    outputs = [helper.make_tensor_value_info(out, TP.FLOAT, [groups, tokens, nh * hd // groups])]
    return make_model(nodes, inputs, outputs, "inverse_rotary_regroup")


@unittest.skipUnless(has_cuda(), "InverseRotaryRegroup is a CUDA-only kernel")
class TestInverseRotaryRegroup(FusedOpTestCase):
    def test_reference(self):
        torch.manual_seed(0)
        tokens = 6
        for elem in ELEM_TYPES:
            for rd, groups in ((64, 1), (64, 4), (0, 2)):
                with self.subTest(dtype=NAME_OF[elem], rope_head_dim=rd, groups=groups):
                    cfg = {"nh": 8, "hd": 128, "rd": rd, "groups": groups}
                    x = torch.randn(tokens, cfg["nh"] * cfg["hd"])
                    cos, sin = rotary_tables(tokens, rd) if rd else (torch.zeros(tokens, 0),) * 2
                    got = run(
                        build_inverse_rotary_regroup(cfg, elem, tokens),
                        {"x": x.numpy(), "cos": cos.numpy(), "sin": sin.numpy()},
                    )[0]
                    want = inverse_rotary_regroup_reference(x, cos, sin, cfg, elem)
                    self.assert_close(got, want.numpy(), TOL[elem], f"invrope rd={rd} g={groups}")

    def test_regroup_is_reshape_transpose_reshape(self):
        """With no rotary slice the operator is pure addressing, so it must be bit-exact."""
        torch.manual_seed(1)
        tokens = 5
        for elem in ELEM_TYPES:
            for groups in (1, 2, 4, 8):
                with self.subTest(dtype=NAME_OF[elem], groups=groups):
                    cfg = {"nh": 8, "hd": 32, "rd": 0, "groups": groups}
                    x = torch.randn(tokens, cfg["nh"] * cfg["hd"])
                    got = run(
                        build_inverse_rotary_regroup(cfg, elem, tokens),
                        {
                            "x": x.numpy(),
                            "cos": np.zeros((tokens, 0), dtype=np.float32),
                            "sin": np.zeros((tokens, 0), dtype=np.float32),
                        },
                    )[0]
                    narrowed = rt(x, elem)
                    want = narrowed.reshape(tokens, groups, -1).permute(1, 0, 2).contiguous()
                    np.testing.assert_array_equal(got, want.numpy())

    def test_inverts_qk_norm_rotary_embedding(self):
        """QKNormRotaryEmbedding -> Reshape -> InverseRotaryRegroup is the identity on the query."""
        torch.manual_seed(3)
        tokens = 6
        nh, hd, rd = 8, 128, 64
        cfg = {"nh": nh, "hd": hd, "rd": rd, "eps": 1e-6, "fp8": False}
        query = torch.randn(1, tokens, nh * hd)
        cos, sin = rotary_tables(tokens, rd)

        nodes = [
            helper.make_node("Unsqueeze", ["cos", "axis0"], ["cos3"], name="u0"),
            helper.make_node("Unsqueeze", ["sin", "axis0"], ["sin3"], name="u1"),
            helper.make_node(
                "QKNormRotaryEmbedding",
                ["query", "kv", "kv_weight", "cos3", "sin3"],
                ["query_out", "kv_out"],
                name="qknorm",
                domain="com.microsoft",
                num_heads=nh,
                head_dim=hd,
                rope_head_dim=rd,
                epsilon=cfg["eps"],
                simulate_fp8=0,
            ),
            helper.make_node("Reshape", ["query_out", "flat"], ["q2d"], name="rs"),
            helper.make_node(
                "InverseRotaryRegroup",
                ["q2d", "cos", "sin"],
                ["y"],
                name="invrope",
                domain="com.microsoft",
                num_heads=nh,
                head_dim=hd,
                rope_head_dim=rd,
                num_groups=1,
            ),
        ]
        initializers = [
            helper.make_tensor("axis0", TP.INT64, [1], [0]),
            helper.make_tensor("flat", TP.INT64, [2], [tokens, nh * hd]),
        ]
        inputs = [
            helper.make_tensor_value_info("query", TP.FLOAT, [1, tokens, nh * hd]),
            helper.make_tensor_value_info("kv", TP.FLOAT, [1, tokens, hd]),
            helper.make_tensor_value_info("kv_weight", TP.FLOAT, [hd]),
            helper.make_tensor_value_info("cos", TP.FLOAT, [tokens, rd]),
            helper.make_tensor_value_info("sin", TP.FLOAT, [tokens, rd]),
        ]
        outputs = [helper.make_tensor_value_info("y", TP.FLOAT, [1, tokens, nh * hd])]
        graph = helper.make_graph(nodes, "roundtrip", inputs, outputs, initializer=initializers)
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
        )
        model.ir_version = 10

        got = run(
            model,
            {
                "query": query.numpy(),
                "kv": torch.randn(1, tokens, hd).numpy(),
                "kv_weight": torch.ones(hd).numpy(),
                "cos": cos.numpy(),
                "sin": sin.numpy(),
            },
        )[0]

        # What survives the round trip is the normalized query, not the raw one.
        q = query.reshape(1, tokens, nh, hd)
        want = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + cfg["eps"])
        self.assert_close(got, want.reshape(1, tokens, nh * hd).numpy(), 1e-5, "rotary round trip")


if __name__ == "__main__":
    unittest.main()
