# Profiling Environment Variables

ONNX Runtime supports configuring profiling behavior through environment variables, providing a convenient way to enable profiling without modifying application code or session configuration.

## Environment Variables

### ORT_ENABLE_PROFILING

Controls whether profiling is enabled for the inference session.

**Valid values:**
- `1` - Enable profiling (overrides `SessionOptions.enable_profiling` to `true`)
- `0` - Use the value from `SessionOptions.enable_profiling` (passthrough mode)
- Not set - Use the value from `SessionOptions.enable_profiling` (passthrough mode)

**Default behavior:** If not set, profiling is controlled solely by `SessionOptions.enable_profiling`.

### ORT_PROFILE_FILE_PREFIX

Specifies the prefix for the profiling output file. The profiler will append a timestamp to create the final filename.

**Valid values:**
- Any non-empty string - Used as the prefix for the profiling file
- Empty string `""` - Explicitly sets an empty prefix (overrides `SessionOptions.profile_file_prefix`)
- Not set - Uses `SessionOptions.profile_file_prefix` if available, otherwise uses the default prefix

**Default value:** `onnxruntime_profile`

## Priority Rules

When both environment variables and `SessionOptions` are configured, the following priority rules apply:

### Enable Profiling Priority
1. `ORT_ENABLE_PROFILING="1"` → **Always enables profiling** (highest priority)
2. `ORT_ENABLE_PROFILING="0"` or not set → Uses `SessionOptions.enable_profiling`

### Profile File Prefix Priority
1. `ORT_PROFILE_FILE_PREFIX` (non-empty or empty string) → **Uses environment variable value** (highest priority)
2. `SessionOptions.profile_file_prefix` → Uses session option value
3. Neither set → Uses default value: `onnxruntime_profile`

## Usage Examples

### Example 1: Enable profiling via environment variable

```bash
# Linux/macOS
export ORT_ENABLE_PROFILING=1
export ORT_PROFILE_FILE_PREFIX=my_model_profile

# Windows (PowerShell)
$env:ORT_ENABLE_PROFILING="1"
$env:ORT_PROFILE_FILE_PREFIX="my_model_profile"

# Windows (Command Prompt)
set ORT_ENABLE_PROFILING=1
set ORT_PROFILE_FILE_PREFIX=my_model_profile
```

Then run your application normally - profiling will be enabled automatically without code changes.

### Example 2: Use default settings with environment variable override

```cpp
// C++ code with profiling disabled
SessionOptions session_options;
session_options.enable_profiling = false;

// If ORT_ENABLE_PROFILING=1 is set, profiling will still be enabled
InferenceSession session(env, model_path, session_options);
```

### Example 3: Passthrough mode

```bash
# Set to "0" to use SessionOptions values
export ORT_ENABLE_PROFILING=0
```

In this case, the application's `SessionOptions.enable_profiling` value will be used.

## Complete Behavior Table

The following table shows all possible combinations of environment variables and `SessionOptions`, and the resulting behavior:

| # | ORT_ENABLE_PROFILING | SessionOptions.enable_profiling | ORT_PROFILE_FILE_PREFIX | SessionOptions.profile_file_prefix | Result: Profiling Enabled | Result: File Prefix |
|---|---------------------|--------------------------------|------------------------|-----------------------------------|--------------------------|-------------------|
| 1 | "1" | false | non-empty ("custom") | not set | ✅ true | "custom" |
| 2 | "1" | false | non-empty ("custom") | set ("session") | ✅ true | "custom" |
| 3 | "1" | false | empty ("") | not set | ✅ true | "" |
| 4 | "1" | false | empty ("") | set ("session") | ✅ true | "" |
| 5 | "1" | false | not set | not set | ✅ true | "onnxruntime_profile" |
| 6 | "1" | false | not set | set ("session") | ✅ true | "session" |
| 7 | "1" | true | non-empty ("custom") | not set | ✅ true | "custom" |
| 8 | "1" | true | non-empty ("custom") | set ("session") | ✅ true | "custom" |
| 9 | "1" | true | empty ("") | not set | ✅ true | "" |
| 10 | "1" | true | empty ("") | set ("session") | ✅ true | "" |
| 11 | "1" | true | not set | not set | ✅ true | "onnxruntime_profile" |
| 12 | "1" | true | not set | set ("session") | ✅ true | "session" |
| 13 | "0" | false | non-empty ("custom") | not set | ❌ false | "custom" |
| 14 | "0" | false | non-empty ("custom") | set ("session") | ❌ false | "custom" |
| 15 | "0" | false | empty ("") | not set | ❌ false | "" |
| 16 | "0" | false | empty ("") | set ("session") | ❌ false | "" |
| 17 | "0" | false | not set | not set | ❌ false | "onnxruntime_profile" |
| 18 | "0" | false | not set | set ("session") | ❌ false | "session" |
| 19 | "0" | true | non-empty ("custom") | not set | ✅ true | "custom" |
| 20 | "0" | true | non-empty ("custom") | set ("session") | ✅ true | "custom" |
| 21 | "0" | true | empty ("") | not set | ✅ true | "" |
| 22 | "0" | true | empty ("") | set ("session") | ✅ true | "" |
| 23 | "0" | true | not set | not set | ✅ true | "onnxruntime_profile" |
| 24 | "0" | true | not set | set ("session") | ✅ true | "session" |

**Note:** When `ORT_ENABLE_PROFILING` is not set (empty), it behaves the same as "0" (passthrough mode).

### Key Observations from the Table:

1. **Rows 1-12**: When `ORT_ENABLE_PROFILING="1"`, profiling is **always enabled** regardless of `SessionOptions.enable_profiling`
2. **Rows 13-24**: When `ORT_ENABLE_PROFILING="0"`, `SessionOptions.enable_profiling` determines the final state
3. **Prefix Priority**: `ORT_PROFILE_FILE_PREFIX` (when set, even to "") always takes priority over `SessionOptions.profile_file_prefix`
4. **Default Prefix**: The default prefix `onnxruntime_profile` is used only when both prefix settings are not set (rows 5, 11, 17, 23)
5. **Empty String Behavior**: Setting `ORT_PROFILE_FILE_PREFIX=""` is treated as an explicit override (different from "not set")

## Implementation Notes

- Environment variables are read during `InferenceSession` construction
- Invalid values for `ORT_ENABLE_PROFILING` (anything other than "0" or "1") are logged as warnings and ignored
- The prefix from `ORT_PROFILE_FILE_PREFIX` is validated to ensure it doesn't contain path separators or other invalid characters
- Environment variable values are logged at `INFO` level when profiling is enabled via environment variables
