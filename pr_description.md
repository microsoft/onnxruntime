## Description

Fix the command-injection risk in the training helper by removing shell-based execution and validating the safe subprocess path with regression tests.

## Summary of Changes

### Security fix: replace shell execution with argument-list execution

| File | Change |
|------|--------|
| `orttraining/tools/scripts/train.py` | Replace `os.system(...)` with `subprocess.run([...], check=True)` and add a small `main(argv=None)` entry point for safe argument forwarding. |
| `orttraining/orttraining/test/python/test_train_script.py` | Add regression tests to confirm the helper does not execute shell commands on import and uses an argument list when invoking the training binary. |

### Motivation

The previous implementation built a single shell string from `sys.argv[1:]` and executed it with `os.system()`. That allowed shell metacharacters in user-supplied arguments to be interpreted by the shell. The updated path passes the command and arguments directly to `subprocess.run`, which avoids shell interpretation and preserves the original argument boundaries.

## Testing

- `pytest orttraining/orttraining/test/python/test_train_script.py -q`

## Checklist

- [x] Security fix implemented
- [x] Regression tests added
- [x] No shell-based command execution remains in the helper path
