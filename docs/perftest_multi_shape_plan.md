# Perftest Multi-Shape Profiling: Implementation Plan

## Overview

This document describes the phased implementation plan for adding multi-shape profiling
support to the ORT perftest tool (`onnxruntime_perf_test`). The feature enables benchmarking
multiple input shapes within a single session, reflecting how production applications use
models with dynamic dimensions.

See also: [Feature Request](../feature_request.md)

---

## Phase 1: CLI Parsing & Configuration (Foundation)

**Goal**: Add the `-data_shape` argument and parse it into `RunConfig`.

### Changes

- Add a new field to `RunConfig` in `test_configuration.h`:
  ```cpp
  std::map<std::string, std::vector<std::vector<int64_t>>> data_shape_groups;
  ```
- Add an `ABSL_FLAG` for `-data_shape` in `command_args_parser.cc`.
- Implement parsing for the bracket syntax:
  ```
  "input_name:[d0,d1,...][d0,d1,...]"
  ```
- Validate at parse time:
  - All inputs have the same number of shape groups.
  - All dimension values are positive integers.
  - Input names match model inputs (or defer validation to Phase 2).

### Files

| File | Change |
|------|--------|
| `test_configuration.h` | New `data_shape_groups` field in `RunConfig` |
| `command_args_parser.cc` | New flag, parsing logic, validation |

### Risk

None — purely additive. The new field is populated but not consumed until Phase 2.

---

## Phase 2: Multi-Shape Input Generation

**Goal**: Generate multiple `test_inputs_` entries (one per shape group) when `-data_shape` is provided.

### Changes

- Add a new method to `OnnxRuntimeTestSession`:
  ```cpp
  bool PopulateMultiShapeInputTestData(
      int32_t seed,
      const std::map<std::string, std::vector<std::vector<int64_t>>>& shape_groups);
  ```
- For each shape group index `i`, create tensors with the user-specified dimensions
  and store via `PreLoadTestData(i, input_idx, ...)`.
- In `PerformanceRunner::Initialize()`, call the new method when `data_shape_groups`
  is non-empty; otherwise fall back to existing `PopulateGeneratedInputTestData`.

### Files

| File | Change |
|------|--------|
| `ort_test_session.h` | Declare new method |
| `ort_test_session.cc` | Implement multi-shape tensor generation |
| `performance_runner.cc` | Gate initialization on `data_shape_groups` |

### Risk

Low — new code path gated behind the presence of shape groups. Existing `-I` behavior
unchanged when `-data_shape` is absent.

### Notes

- CUDA path: Create tensors on CPU, initialize, then `cudaMemcpy` to device (same
  pattern as existing `PopulateGeneratedInputTestData`).
- The `-I` flag is still required to trigger auto-generation; `-data_shape` without
  `-I` should produce a clear error.

---

## Phase 3: Deterministic Shape Cycling

**Goal**: Replace random input selection with round-robin when in multi-shape mode.

### Changes

- Add a shape index tracker to `OnnxRuntimeTestSession` (e.g., `size_t current_shape_idx_`).
- In `Run()`:
  - When `data_shape_groups` is active: use `current_shape_idx_ % num_groups` and increment.
  - Otherwise: preserve existing random selection behavior.
- Expose which shape group index was used for the current iteration (e.g., return it
  alongside `RunTiming`, or store it in a queryable member).

### Files

| File | Change |
|------|--------|
| `ort_test_session.h` | Add index tracker, expose last-used index |
| `ort_test_session.cc` | Conditional round-robin vs. random in `Run()` |
| `test_session.h` | Possibly extend interface to expose shape index |

### Risk

Low — behavioral change only when `-data_shape` is active. Random selection preserved
for all existing usage.

### Design Decision

Round-robin is chosen over random selection to ensure equal coverage across shape groups
and produce deterministic, reproducible results.

---

## Phase 4: Per-Shape Reporting

**Goal**: Track and print per-shape-group latency statistics.

### Changes

- Extend `PerformanceResult`:
  ```cpp
  // Per-shape-group timing (indexed by shape group)
  std::vector<std::vector<double>> per_shape_time_costs_total;
  std::vector<std::vector<double>> per_shape_time_costs_submission;
  ```
- In `RunOneIteration<false>()`, attribute the timing to the correct shape bucket
  based on the shape index from Phase 3.
- After the existing overall summary, print per-shape statistics:
  ```
  Latency per shape group:
    1. input : [1,16,1440,2560]
        Median:  17.04 ms
        Average: 17.08 ms
        Min:     16.32 ms
        Max:     18.99 ms
    ...
  ```
- Extend `DumpToFile` to include per-shape data (append shape group column to CSV).

### Files

| File | Change |
|------|--------|
| `performance_runner.h` | Extend `PerformanceResult` |
| `performance_runner.cc` | Attribution logic in `RunOneIteration`, new reporting |

### Risk

Low — additive output. The existing summary is printed first with identical format;
per-shape stats are appended only when the feature is active.

---

## Phase 5: Per-Shape Warmup

**Goal**: Warm up each shape group individually before measurement begins.

### Changes

- In `PerformanceRunner::Run()`, when `data_shape_groups` is non-empty:
  ```cpp
  for (size_t shape_idx = 0; shape_idx < num_shape_groups; ++shape_idx) {
      session_->SetCurrentShapeGroup(shape_idx);
      ORT_RETURN_IF_ERROR(RunOneIteration<true>());
  }
  ```
- This ensures each EP has processed every shape (triggering any lazy compilation or
  memory allocation) before timing begins.
- When `-data_shape` is not set, the existing single warmup iteration is unchanged.

### Files

| File | Change |
|------|--------|
| `ort_test_session.h/cc` | Add `SetCurrentShapeGroup()` method |
| `performance_runner.cc` | Per-shape warmup loop |

### Risk

Minimal — uses the existing `RunOneIteration<true>()` template which discards all
timing data at compile time.

---

## PR Strategy

| PR | Phases | Description |
|----|--------|-------------|
| PR 1 | 1 + 2 | CLI parsing and multi-shape input generation |
| PR 2 | 3 + 4 | Shape cycling and per-shape reporting (user-visible feature) |
| PR 3 | 5 | Per-shape warmup (correctness polish) |

Each PR is independently shippable. PR 1 is a no-op from the user's perspective (shapes
are generated but cycled randomly). PR 2 delivers the full user-facing feature. PR 3 is
an optional correctness improvement.

---

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Argument format | `-data_shape "name:[shape][shape]..."` | Explicit, general, handles arbitrary shapes (not just named free dims) |
| Shape cycling | Round-robin | Equal coverage, deterministic, reproducible |
| Multi-input models | Repeat `-data_shape` per input | All inputs must have same group count |
| Interaction with `-f` | `-data_shape` takes precedence | `-f` still applies to dimensions not covered by `-data_shape` |
| Warmup | One iteration per shape group | Ensures EP-specific lazy compilation occurs before measurement |

---

## Compatibility

- **No regression risk**: All new behavior is gated behind `-data_shape` being non-empty.
- **Existing output format**: Unchanged when the flag is absent.
- **Concurrent mode**: Per-shape attribution uses the existing `results_mutex_`.
- **CUDA/IO binding**: Multi-shape generation follows the same CPU→GPU copy pattern.
