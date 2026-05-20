# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Patch a HuggingFace causal-LM ONNX export so it only returns the last
token's logits.  Standard LLM-serving optimization (HF transformers calls
this `logits_to_keep=1`) that has two benefits:

1. **Saves compute** on the LM head matmul: instead of projecting
   the full hidden state of shape (B, S_q, hidden) through a
   (hidden, vocab) weight matrix, we slice the hidden state to
   (B, 1, hidden) first.  At long context this is a huge win
   (e.g. at 128 K and vocab 65 536, cuts 8.4 GFLOP per layer to
   65.5 KFLOP for the LM head alone).

2. **Dodges int32 overflow** in CUDA Cast / pointwise kernels at
   long context.  ORT's Cast (and many element-wise kernels) iterate
   with int32 element counters; element count = B * S_q * vocab.
   For LFM2.5 (vocab = 65536) the limit is `B * S_q * 65536 < 2^31`
   → S_q < 32768.  Without this slice the final logits Cast hits an
   illegal-memory-access at any S_q > 32K.  With this slice the
   element count drops to `B * 1 * vocab` regardless of S_q, so the
   model runs at full 128 K context.

Usage:
    python -m onnxruntime.quantization.turboquant_kv.last_token_logits \\
        path/to/model.onnx -o path/to/model_lasttok.onnx

Notes:
- Output `logits` shape changes from (B, S_q, vocab) to (B, 1, vocab).
  Code that uses only `logits[:, -1, :]` for greedy decoding is
  unaffected (just access `logits[:, 0, :]` instead).
- This is NOT TurboQuant-specific — it applies to any causal LM ONNX
  export — but it lives next to the TQ tools because the long-context
  TQ benchmark needs it.  Should probably move to a sibling tool dir
  if other consumers want it.
"""

from __future__ import annotations

import argparse
import logging

import onnx
from onnx import TensorProto, helper

logger = logging.getLogger(__name__)


def patch_to_last_token(in_path: str, out_path: str) -> None:
    m = onnx.load(in_path, load_external_data=True)

    # Find the LM head: a MatMul / MatMulNBits node whose name contains "lm_head".
    lm_head = None
    for n in m.graph.node:
        if "lm_head" in n.name and n.op_type in ("MatMulNBits", "MatMul"):
            lm_head = n
            break
    if lm_head is None:
        raise RuntimeError(
            "could not find an lm_head MatMul / MatMulNBits node in the graph"
        )
    logger.info("found LM head: %s op=%s input[0]=%s",
                lm_head.name, lm_head.op_type, lm_head.input[0])

    # Add Slice initializers + node, taking dim 1 of the hidden state from -1 to end.
    starts = helper.make_tensor("__last_tok_starts", TensorProto.INT64, [1], [-1])
    ends   = helper.make_tensor("__last_tok_ends",   TensorProto.INT64, [1], [9223372036854775807])
    axes   = helper.make_tensor("__last_tok_axes",   TensorProto.INT64, [1], [1])
    m.graph.initializer.extend([starts, ends, axes])

    sliced = "__last_tok_hidden"
    slice_node = helper.make_node(
        "Slice",
        inputs=[lm_head.input[0], "__last_tok_starts", "__last_tok_ends", "__last_tok_axes"],
        outputs=[sliced],
        name="/model/graph_outputs/logits/LastTokenSlice",
    )
    m.graph.node.extend([slice_node])
    lm_head.input[0] = sliced

    # Update the graph output type info: dim 1 is now 1 instead of 'sequence_length'.
    for o in m.graph.output:
        if o.name == "logits":
            if len(o.type.tensor_type.shape.dim) >= 2:
                o.type.tensor_type.shape.dim[1].ClearField("dim_param")
                o.type.tensor_type.shape.dim[1].dim_value = 1
            break

    logger.info("saving %s", out_path)
    onnx.save(m, out_path,
              save_as_external_data=True,
              all_tensors_to_one_file=True,
              location=out_path.split("/")[-1] + "_data")


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="onnxruntime.quantization.turboquant_kv.last_token_logits",
        description="Patch a causal-LM ONNX export to only return the last token's logits.",
    )
    parser.add_argument("input", help="input model.onnx")
    parser.add_argument("-o", "--output", required=True, help="output model.onnx")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    patch_to_last_token(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
