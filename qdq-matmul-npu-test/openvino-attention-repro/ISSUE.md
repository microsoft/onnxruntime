# OpenVINO NPU mask-QDQ/Add/Softmax changes output for an all-zero mask

## Bug

A model with `QuantizeLinear(mask) -> DequantizeLinear(mask) ->
Add(scores, mask) -> Softmax` disagrees with ORT CPU when compiled for the
OpenVINO NPU through Windows ML. Removing the mask QDQ (leaving
`Add -> Softmax`) produces the expected NPU result. The mask is **all zeros**
in both models, so the float value added to the scores is unchanged.

Varying *only* the mask QDQ scale, while maintaining the same dequantized
zeros and identical ORT CPU outputs, varies the NPU output. This is a
four-operator, weight-free, static-shape reproducer; it does not rely on a
Gemma model, MatMul, causal attention, calibration artifacts, or CPU fallback.

## Steps to reproduce

1. Use a Windows ML Python environment with the Intel OpenVINO EP and an
   available Intel NPU, plus NumPy and ONNX.
2. Run `python repro.py --output-dir output` from this directory. If catalog
   discovery is unavailable, pass `--provider-library` with the installed
   OpenVINO WinML plugin DLL.
3. Inspect `output\results.json` and the generated ONNX models. All four
   compiled models contain exactly one `EPContext`; CPU fallback is disabled.

## Expected versus actual

All four CPU outputs are bitwise identical. Their NPU outputs should agree up
to normal device precision, regardless of the mask-QDQ scale. On an Intel
OpenVINO EP 1.8 / SDK 2026.3 NPU with ORT 1.30.0, the float-mask control
has CPU/NPU MAE `2.76e-5`; the QDQ-mask scale-0.5 variant matches that NPU
control bit-for-bit. The scale-1 and scale-2 variants instead have MAE
`0.032262` and `0.049413`. The mask is all zero in every run.

The discrepancy appears when lowering the mask-QDQ/Add/Softmax graph;
the exact internal pass or kernel has not been determined. Could the
OpenVINO NPU EP/compiler preserve ONNX QDQ semantics across this pattern?

The attached/source artifacts should be reviewed before sharing. Generated
ONNX models and inputs contain **no trained weights**; compiled EP contexts
are device-specific and unnecessary to reproduce.
