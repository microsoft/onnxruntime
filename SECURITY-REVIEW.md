# PackedAttention Token Offset Review

## Summary

ONNX Runtime GPU 1.26.0 does not validate `PackedAttention` `token_offset`
values before the unfused CUDA backend uses them as padded-buffer indices.
Out-of-range values can cause CUDA out-of-bounds writes during Q/K/V
materialization and an out-of-bounds read during output compaction.

The reported output-compaction read is a genuine memory-safety defect, but it
is not a separate root cause from the previously reported write defect. Both
are consequences of the same unvalidated `token_offset` input.

## Severity

**Medium, confidence 9/10.**

Exploitation requires the CUDA Execution Provider, selection of the unfused
`PackedAttention` backend, control of the model or `token_offset` input, and
access to the model output.

## Evidence

- `AddBiasTransposeQKVPacked` uses each offset to select Q/K/V workspace
  destinations.
- `TransposeRemovePadding` uses each offset to select a source row that is
  copied to the returned output.
- Version 1.26.0 performs no value validation before these accesses.
- The operator's host validation checks tensor shapes but not offset values.

The source establishes an out-of-bounds read primitive. The stronger claim
that output contains unrelated cross-request or cross-tenant data is plausible
but not demonstrated by the supplied report. The report does not include a
repeatable sentinel-recovery experiment or proof that returned bytes originated
from another allocation.

## Duplicate Assessment

The read and write sinks should be tracked as one issue because they share:

- the same attacker-controlled input;
- the same missing range check;
- the same CUDA operator and backend;
- the same required remediation.

Different access directions and impact descriptions do not create independent
root causes.

## Fix Status

- ONNX Runtime GPU 1.26.0 is affected.
- Current `main` guards the final output-compaction read, but does not perform
  operator-level validation before the earlier write sink.
- Local commit `eeffc2d6c9` on branch
  `fix/packed-attention-token-validation`:
  - validates all offsets before unfused execution;
  - returns `INVALID_ARGUMENT` outside CUDA graph capture;
  - guards the Q/K/V write sink during capture;
  - guards the final read and zeroes invalid output rows;
  - adds negative and upper-bound offset regressions.

## Recommended Resolution

Treat the disclosure report as a duplicate of the unvalidated
`PackedAttention` `token_offset` issue. Land the operator-level validation and
retain inline CUDA guards for graph-capture safety.
