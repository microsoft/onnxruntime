# Playbook 10: Python and C API Binding Extension

## Outcome

By the end of this playbook, you will be able to add or extend a runtime feature across the public C API and Python bindings, while preserving API compatibility and validating behavior with focused tests.

This playbook assumes you have already completed [Playbook 03](03-first-pr-and-environment-setup.md), [Playbook 04](04-session-lifecycle-from-load-to-run.md), and [Playbook 08](08-execution-provider-implementation.md).

## Start Here

- [docs/C_API_Guidelines.md](../C_API_Guidelines.md)
- [include/onnxruntime/core/session/onnxruntime_c_api.h](../../include/onnxruntime/core/session/onnxruntime_c_api.h)
- [onnxruntime/python/onnxruntime_pybind_state.cc](../../onnxruntime/python/onnxruntime_pybind_state.cc)
- [onnxruntime/test/python/onnxruntime_test_python.py](../../onnxruntime/test/python/onnxruntime_test_python.py)

## Mental Model

Binding work should follow a strict layer order:

1. C API contract definition and compatibility
2. runtime implementation behind that API
3. Python binding exposure
4. Python tests validating behavior and error handling

Skipping this order causes drift where Python appears to expose behavior that is not stable in the underlying public API.

## C API First

Use [docs/C_API_Guidelines.md](../C_API_Guidelines.md) as the authoritative checklist.

Key requirements when adding API surface:

- proper API documentation comments and UTF-8 string expectations
- correct API macros and calling convention
- `OrtStatus*` error contract (`nullptr` on success)
- no exception leakage across C/C++ boundaries
- no out-parameter mutation on failure
- allocator parameters for APIs that allocate memory

For table-based APIs, append new entries in version-safe order to preserve backward compatibility.

## Where to Add or Update C API Surface

Primary header:

- [include/onnxruntime/core/session/onnxruntime_c_api.h](../../include/onnxruntime/core/session/onnxruntime_c_api.h)

Typical examples to study include `SessionOptions`, `RunOptions`, and EP append/configuration APIs.

When extending existing behavior, prefer evolving option-based APIs where possible instead of introducing avoidable new top-level API calls.

## Python Binding Layer

Primary binding file:

- [onnxruntime/python/onnxruntime_pybind_state.cc](../../onnxruntime/python/onnxruntime_pybind_state.cc)

In this file you can trace:

- Python `SessionOptions` exposure and property mapping
- provider and EP option wiring
- helper methods that map Python calls to C/C++ runtime calls

For many extensions, the practical work is:

1. expose a new property or method in pybind
2. map to existing or newly added C/C++ runtime behavior
3. preserve existing defaults and failure semantics

Keep Python naming consistent with existing API style.

## Typical Change Paths

### Path A: Add a new configurable session option

1. add or extend C API/session option support
2. wire option into runtime behavior
3. expose in Python `SessionOptions`
4. add Python tests for default, custom value, and invalid value

### Path B: Add a run-time option

1. add C/C++ `RunOptions` support
2. plumb value into run path
3. expose through Python run options or run call parameters
4. add tests proving per-run override behavior

### Path C: Add binding-only helper for existing runtime capability

1. confirm stable existing runtime/C API behavior
2. add pybind helper wrapper
3. add focused Python tests without changing runtime semantics

## Test Strategy

Primary Python test anchor:

- [onnxruntime/test/python/onnxruntime_test_python.py](../../onnxruntime/test/python/onnxruntime_test_python.py)

Add focused tests that verify:

- default behavior unchanged
- new value changes behavior as intended
- invalid input returns clear failure
- provider interactions remain compatible

If a change affects specific EP behavior, add EP-gated tests that only run when that provider is available.

## Fast Validation Loop

Run targeted Python tests first:

```bash
python -m pytest onnxruntime/test/python/onnxruntime_test_python.py -k "session or providers or run_options"
```

Narrow to exact test names while iterating.

## Design Rules

- treat the C API as the compatibility boundary
- keep Python wrappers thin and explicit
- avoid silently changing default behavior
- preserve error message clarity for caller debugging
- add tests before broad refactors

## Common Failure Modes

- adding Python exposure without stable C API/runtime support
- breaking table ordering or compatibility for C API additions
- leaking C++ exceptions through C boundaries
- changing defaults in Python wrappers without migration guidance
- writing only positive tests and missing invalid-input coverage

## Exit Checklist

- [ ] C API contract is documented and compatibility-safe.
- [ ] Runtime implementation and C API semantics match.
- [ ] Python binding exposes the feature with consistent naming and defaults.
- [ ] Focused Python tests cover success and failure behavior.