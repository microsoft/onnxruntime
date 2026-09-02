---
name: python-kwargs-setattr-security
description: When reviewing or fixing Python code that uses setattr() with user-controlled kwargs to configure C++ extension objects (SessionOptions, RunOptions, etc.) in ONNX Runtime. Use this to apply the allowlist pattern that prevents arbitrary file writes and other attacks via reflected property access.
---

## Problem Pattern

Using `hasattr(obj, k) / setattr(obj, k, v)` with user-controlled kwargs is insecure. The `hasattr` check is NOT a security guard — it returns True for ALL exposed properties including dangerous ones.

```python
# INSECURE — do not use
for k, v in kwargs.items():
    if hasattr(options, k):
        setattr(options, k, v)
```

## Fix: Explicit Allowlist

Define a module-level frozenset of safe attribute names. Raise RuntimeError for known-but-blocked attrs; silently ignore unknown keys.

```python
# Define at module level, before the class
_ALLOWED_SESSION_OPTIONS = frozenset({
    "enable_cpu_mem_arena",
    "enable_mem_pattern",
    # ... only explicitly reviewed safe attrs
})

# In the method
for k, v in kwargs.items():
    if k in _ALLOWED_SESSION_OPTIONS:
        setattr(options, k, v)
    elif hasattr(options, k):  # reuse the existing instance, don't create new
        raise RuntimeError(
            f"SessionOptions attribute '{k}' is not permitted via the backend API. "
            f"Allowed attributes: {', '.join(sorted(_ALLOWED_SESSION_OPTIONS))}"
        )
    # else: silently ignore (may be kwargs for a different config object)
```

## Key Rules

1. **Use the existing object** in `hasattr(options, k)` — never `hasattr(ClassName(), k)` (creates throwaway C++ objects per iteration)
2. **RuntimeError** is the ORT convention for API misuse errors (not ValueError)
3. **Silent ignore for one path is OK when kwargs are forwarded to both paths**: `run_model()` passes the same kwargs dict to both `prepare()` (validates SessionOptions) and `rep.run()` (validates RunOptions). A RunOptions kwarg unknown to SessionOptions is silently ignored by `prepare()` — this is correct because `rep.run()` will validate it. Only raise RuntimeError when the attr exists on the target object but is blocked.
4. **Frozenset constant naming**: `_ALLOWED_<CLASSNAME>` — ALL_CAPS, Google Style
5. **No type annotations** on module-level constants (ORT Python convention)

## Dangerous SessionOptions Properties (never allowlist)

- `optimized_model_filepath` — triggers Model::Save(), overwrites arbitrary files
- `profile_file_prefix` + `enable_profiling` — writes profiling JSON to arbitrary path
- `register_custom_ops_library` — loads arbitrary shared libraries (method, not property)

## Files in ONNX Runtime

- `onnxruntime/python/backend/backend.py` — `_ALLOWED_SESSION_OPTIONS`
- `onnxruntime/python/backend/backend_rep.py` — `_ALLOWED_RUN_OPTIONS`
- Tests: `onnxruntime/test/python/onnxruntime_test_python_backend.py` — `TestBackendKwargsAllowlist`
