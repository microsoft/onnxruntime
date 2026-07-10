# onnx-ir / onnxscript-rewriter fusions

This package incrementally re-implements the graph fusions under
`onnxruntime/python/tools/transformers` — historically built on the proto-based
`OnnxModel` helper API (`onnx_model.py`) — using
[`onnx-ir`](https://github.com/onnx/ir-py) and
[`onnxscript.rewriter`](https://github.com/microsoft/onnxscript).

## Why

The existing fusions manipulate `onnx.ModelProto` directly through the large
`OnnxModel` wrapper (`match_parent_path`, `is_safe_to_fuse_nodes`,
`nodes_to_add` / `nodes_to_remove`, ...). Expressing the same fusions as
declarative `onnxscript.rewriter` rules over `onnx-ir`:

- makes each fusion a small, self-contained `pattern()` / `check()` / `rewrite()`
  triple instead of imperative proto surgery,
- reuses the shared, well-tested pattern matcher (safety of the replaced
  subgraph, topological handling, subgraph traversal) instead of re-deriving it
  per fusion,
- operates on the in-memory IR with no repeated proto (de)serialization.

## Scope

This is an **incremental** migration. Each PR converts a small batch of
self-contained fusions and keeps the existing proto-based fusion in place until
the IR path is wired into the optimizer driver. Fusions already covered by other
onnx-ir rewriters are intentionally **not** duplicated here.

### First batch

| Fusion | Replaces | Target op |
|---|---|---|
| `quick_gelu_rules()` | `fusion_quickgelu.py` | `com.microsoft.QuickGelu` |
| `bias_add_rules()` | `fusion_bias_add.py` | `com.microsoft.BiasAdd` |
| `group_norm_rules()` | `fusion_group_norm.py` | `com.microsoft.GroupNorm` |

## Usage

```python
import onnx_ir as ir
from onnxscript.rewriter import rewrite
from onnx_ir_fusions import all_rules  # or quick_gelu_rules(), etc.

model = ir.load("model.onnx")
rewrite(model, pattern_rewrite_rules=all_rules())
ir.save(model, "model.fused.onnx")
```

## Adding a fusion

1. Create `_<name>.py` with a `RewriteRuleClassBase` subclass
   (`pattern` / `check` / `rewrite`) and a `..._rules()` factory returning a
   `RewriteRuleSet`.
2. Add co-located `_<name>_test.py` unit tests that build a small model, apply
   the rule, and assert the fused op appears (plus a negative test that a
   non-matching model is untouched).
3. Export the factory from `__init__.py` and add it to `all_rules()`.

See the `writing rewrite rules` pattern used across the onnx-ir ecosystem.
