# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import math
import os
import unittest

import torch
from parity_utilities import parse_arguments, create_ort_session, compare_outputs
from torch import nn
import numpy


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    return q_embed

class RotaryEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, position_ids, cos_cache, sin_cache, past_key) -> torch.Tensor:
        seq_len = q.shape[-2]
        if past_key is not None:
            seq_len += past_key.shape[-2]
        cos = cos_cache[:,:,:seq_len,...].to(q.dtype)
        sin = sin_cache[:,:,:seq_len,...].to(q.dtype)

        cos = cos.reshape(cos.shape[2], cos.shape[3])
        sin = sin.reshape(sin.shape[2], sin.shape[3])
        cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
        q_embed = (q * cos) + (rotate_half(q) * sin)
        return q_embed


    @staticmethod
    def symbolic(g: torch.Graph, q, position_ids, cos_cache, sin_cache, past_key) -> (torch.Value, torch.Value):
        if past_key is None:
            return g.op('com.microsoft::RotaryEmbedding', q, position_ids, cos_cache, sin_cache)
        else:
            return g.op('com.microsoft::RotaryEmbedding', q, position_ids, cos_cache, sin_cache, past_key)

class TestRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.emb = LlamaRotaryEmbedding(dim)

    def forward(self, x, position_ids, past_key=None):
        return RotaryEmbedding.apply(x, position_ids, self.emb.cos_cached, self.emb.sin_cached, past_key)


def create_inputs(batch_size, num_heads, seqlen, hidden_size, dtype, device, past_seqlen=0):
    in_tensor = torch.randn(batch_size, num_heads, seqlen, hidden_size).to(dtype).to(device)
    position_ids = torch.stack([torch.arange(0, seqlen) for _ in range(batch_size)]).to(device)
    inputs = {"x": in_tensor, "position_ids": position_ids}
    if past_seqlen > 0:
        past_key = torch.randn(batch_size, num_heads, past_seqlen, hidden_size).to(dtype)
        inputs['past_key']=past_key
        seqlen_with_past = seqlen + past_seqlen
        inputs['position_ids'] = torch.stack([torch.tensor([seqlen_with_past - 1], dtype=torch.int64) for _ in range(batch_size)]).to(device)

    return inputs

def onnxruntime_inference(ort_session, inputs):
    ort_inputs = {k: numpy.ascontiguousarray(v.cpu().numpy()) for k,v in inputs.items()}
    ort_outputs = ort_session.run(None, ort_inputs)
    return ort_outputs


def run_parity(
    model,
    onnx_model_path,
    batch_size,
    hidden_size,
    sequence_length,
    num_heads,
    float16,
    device,
    past_key_seqlen=0,
    test_cases=100,
    verbose=False,
    tolerance=None,
):
    passed_cases = 0
    max_diffs = []
    printed = False  # print only one sample
    ort_session = create_ort_session(onnx_model_path, device.type == "cuda", verbose=verbose)
    for _i in range(test_cases):
        inputs = create_inputs(batch_size, num_heads, sequence_length, hidden_size, dtype=torch.float16 if float16 else torch.float32, device=device, past_seqlen=past_key_seqlen)

        with torch.no_grad():
            torch_outputs = model(**inputs)
            if not isinstance(torch_outputs, (list, tuple)):
                torch_outputs = [torch_outputs,]

        ort_outputs = onnxruntime_inference(ort_session, inputs)

        if tolerance is None:
            tolerance = 2e-02 if float16 else 1e-05
        is_all_close, max_diff = compare_outputs(torch_outputs, ort_outputs, atol=tolerance, verbose=verbose)
        max_diffs.append(max_diff)
        if is_all_close:
            passed_cases += 1
        elif verbose and not printed:
            printed = True
            numpy.set_printoptions(precision=10, floatmode="fixed")
            torch.set_printoptions(precision=10)
            print("input", inputs)
            print('diff: ', max_diff)
            print("torch_outputs", torch_outputs)
            print("ort_outputs", ort_outputs)

    max_diff = max(max_diffs)
    diff_count = len([i for i in max_diffs if i > 0])
    success_flag = "[FAILED]" if passed_cases < test_cases else "[OK]"
    print(f"{success_flag} Passed_cases={passed_cases}/{test_cases}; Max_diff={max_diff}; Diff_count={diff_count}")
    return test_cases - passed_cases

def run(
    batch_size,
    float16,
    hidden_size,
    device,
    test_cases,
    sequence_length=2,
    num_heads=16,
    past_key_seqlen=0,
    verbose=False,
):
    test_name = f"device={device}, float16={float16}, batch_size={batch_size}, sequence_length={sequence_length}, hidden_size={hidden_size}, num_heads={num_heads}, past_key={past_key_seqlen}"
    print(f"\nTesting: {test_name}")

    model = TestRotaryEmbedding(hidden_size)
    model.eval()
    model.to(device)
    if float16:
        model.half()

    inputs = create_inputs(batch_size, num_heads, sequence_length, hidden_size, dtype=torch.float16 if float16 else torch.float32, device=device, past_seqlen=past_key_seqlen)

    # Do not re-use onnx file from previous test since weights of model are random.
    onnx_model_path = "./temp/rotary_emb_{}_{}.onnx".format(sequence_length, "fp16" if float16 else "fp32")

    torch.onnx.export(
        model,
        args=inputs,
        f=onnx_model_path,
        input_names=tuple(inputs.keys()),
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
    )

    num_failure = run_parity(
        model,
        onnx_model_path,
        batch_size,
        hidden_size,
        sequence_length,
        num_heads,
        float16,
        device,
        past_key_seqlen,
        test_cases,
        verbose,
    )

    # clean up onnx file
    os.remove(onnx_model_path)

    return num_failure, test_name


class TestParity(unittest.TestCase):
    verbose = True
    def setUp(self):
        self.test_cases = 100  # Number of test cases per test run
        self.sequence_length = 2
        self.hidden_size = 768
        self.num_heads=8

    def run_test(
        self,
        batch_size,
        float16,
        device,
        seq_len,
        past_key_seqlen=0,
        enable_assert=True,
        verbose=True,
    ):
        if float16 and device.type == "cpu":  # CPU does not support FP16
            return
        num_failure, test_name = run(
            batch_size,
            float16,
            hidden_size=self.hidden_size,
            device=device,
            test_cases=self.test_cases,
            sequence_length=seq_len,
            num_heads=self.num_heads,
            past_key_seqlen=past_key_seqlen,
            verbose=verbose,
        )
        if enable_assert:
            self.assertTrue(num_failure == 0, "Failed: " + test_name)

    def run_one(self, device, verbose=False):
        for batch_size in [1,2,4]:
            self.run_test(
                batch_size,
                float16=False,
                device=device,
                seq_len=15,
                past_key_seqlen=0,
                verbose=verbose,
            )

            self.run_test(
                batch_size,
                float16=False,
                device=device,
                seq_len=1,
                past_key_seqlen=15,
                verbose=verbose,
            )

            self.run_test(
                batch_size,
                float16=True,
                device=device,
                seq_len=15,
                past_key_seqlen=0,
                verbose=verbose,
            )

            self.run_test(
                batch_size,
                float16=True,
                device=device,
                seq_len=1,
                past_key_seqlen=15,
                verbose=verbose,
            )

    def test_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("test requires GPU and torch+cuda")
        else:
            gpu = torch.device("cuda")
            self.run_one(gpu, verbose=self.verbose)


if __name__ == "__main__":
    args, remaining_args = parse_arguments(namespace_filter=unittest)

    TestParity.verbose = args.log_verbose

    unittest.main(argv=remaining_args)
