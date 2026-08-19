# Shared Allocator Registration Design

## Problem

Registering a plugin execution provider (EP) can make an existing shared
`OrtAllocator` inaccessible when the EP advertises equivalent device memory
information.

`Environment` keeps two related collections:

- `shared_allocators_` contains the internal `IAllocator` wrappers used by
  inference sessions.
- `shared_ort_allocators_` contains the public `OrtAllocator*` values returned
  by `GetSharedAllocator()`.

`Environment::CreateSharedAllocatorImpl()` currently removes a matching entry
from `shared_ort_allocators_` before it checks `shared_allocators_`. During
automatic EP registration, `replace_existing` is `false`. If an equivalent
internal allocator already exists, the function returns without creating a new
allocator, leaving the internal entry intact but deleting its public lookup
entry. A later `GetSharedAllocator()` call then returns `nullptr`.

WebGPU can trigger this path when multiple enumerated devices advertise
equivalent memory information, but neither the faulty state transition nor its
fix is WebGPU-specific. The same failure can be produced with one example
plugin EP device and a previously registered equivalent custom allocator.

## Desired Behavior

When `CreateSharedAllocatorImpl()` is called with `replace_existing == false`
and an equivalent shared allocator already exists, registration must be a true
no-op:

- Keep the existing internal shared allocator.
- Keep the corresponding public `OrtAllocator*` lookup entry.
- Do not ask the EP factory to create a replacement allocator.
- Return success.

When `replace_existing == true`, existing replacement semantics remain
unchanged: remove the old public and internal entries, create the requested EP
allocator, and register it in both collections.

## Design

Reorder the duplicate check in
`Environment::CreateSharedAllocatorImpl()`:

1. Search `shared_allocators_` for equivalent device memory information using
   the existing name-insensitive comparison.
2. If a match exists and replacement is disabled, return success immediately.
   No collection has been mutated at this point.
3. Otherwise, remove the matching public `OrtAllocator*` entry as required for
   safe ownership teardown.
4. If an internal match exists, remove it.
5. Create and register the new allocator using the existing code path.

The order of public-before-internal removal remains intact on replacement paths
because an internal wrapper may own the public allocator pointer. Only the
non-replacement early return moves ahead of those mutations.

No API, allocator-equivalence rule, WebGPU device identity, or provider-specific
behavior changes.

## Regression Test

Add a provider-neutral test to
`onnxruntime/test/autoep/test_allocators.cc`:

1. Create memory information equivalent to the example plugin EP's default
   device allocator.
2. Register a `DummyAllocator` using that memory information.
3. Confirm `GetSharedAllocator()` returns the custom allocator.
4. Register the existing example plugin EP, which automatically calls
   `CreateSharedAllocatorImpl(..., replace_existing=false)` for its device.
5. Confirm `GetSharedAllocator()` still returns the same custom allocator.
6. Unregister the plugin and custom allocator through existing cleanup paths.

This test fails before the production change because step 4 removes the public
lookup entry and then returns early. It passes after the change and demonstrates
that multiple GPUs are not required to reproduce the bug.

The test will use the auto-EP test environment and scope guards for both the EP
and custom allocator registrations so state is removed even when an assertion
fails.

## Verification

- Run the new regression test alone and confirm it fails before the production
  change, then passes afterward.
- Run all `SharedAllocators.*` tests in `onnxruntime_autoep_test`.
- Run the broader allocator/EP registration tests supported by the existing
  Windows Release build.
- Rebuild the affected runtime and run the issue #32164 WebGPU reproducer with
  the currently enabled adapters to confirm `GetSharedAllocator()` remains
  non-null.

## Non-Goals

- Assigning unique `OrtMemoryInfo` identities to WebGPU adapters.
- Changing GPU enumeration or adapter selection.
- Changing public allocator APIs or memory-info equivalence semantics.
- Refactoring unrelated allocator registration or ownership logic.
