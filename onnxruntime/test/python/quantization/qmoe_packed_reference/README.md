# Packed-Code Reference Tests

CPU-only format and numerical references, not a production quantizer or kernel.
Generic 1..8-bit packing coverage does not enable those widths in QMoE.

## Run

Requires `numpy`, `onnx`, and `pytest`; optional `ml_dtypes` enables BF16 tests.
No installed ONNX Runtime or GPU is needed. From this directory:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider --rootdir=. --confcutdir=. test_reference.py
```

The collection boundaries avoid importing the enclosing native ORT package.
Existing Python build/test CI copies this directory, including the golden JSON
fixtures, to the build output and runs the pytest suite after the quantization
unittest suite when contrib ops are enabled. No feature-specific workflow is needed.
For the optional external reference check, set `ONNX_PACK_PROPOSAL_CHECKOUT` to
a trusted, unmodified checkout at the revision pinned in the test. The test
checks HEAD, executes that checkout's Python references, and otherwise skips;
it never downloads code or requires draft operators in the installed ONNX.

## Validation Boundaries

Generic unpack ignores unused high padding bits; canonical-padding validation
is separate. The INT3 profile requires zero padding and explicit zero-point
codes equal to 4. Weights unpack with count K; block zero points use count B.
Call `validate_int3` before evaluating the ONNX decoding reference: that graph
expresses arithmetic, not profile validation.

The Python reference accepts tails, empty dimensions, and FP32 scales for
format/numerical testing. The QMoE INT3 schema and C++ host validator instead
require block-aligned model dimensions and matching FP16/BF16 activation and
scale types. These references do not imply provider execution support.