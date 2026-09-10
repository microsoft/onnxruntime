# Playbook 06: Adding a Contrib Operator

## Outcome

By the end of this playbook, you will be able to add a `com.microsoft` contrib operator by defining its schema, wiring at least one kernel, registering it with the target execution provider, and validating it with focused tests.

This playbook assumes you have already completed [Playbook 04](04-session-lifecycle-from-load-to-run.md) and [Playbook 05](05-adding-or-changing-a-kernel.md).

## Start Here

- [docs/ContribOperators.md](../ContribOperators.md)
- [onnxruntime/core/graph/contrib_ops/contrib_defs.cc](../../onnxruntime/core/graph/contrib_ops/contrib_defs.cc)
- [onnxruntime/contrib_ops/cpu/cpu_contrib_kernels.cc](../../onnxruntime/contrib_ops/cpu/cpu_contrib_kernels.cc)
- [onnxruntime/test/contrib_ops](../../onnxruntime/test/contrib_ops)

## Pick a Small Reference Path

For learning the contrib flow, two small operators are useful references:

- `SampleOp` for the smallest schema-to-kernel-to-test path
- `ExpandDims` for a slightly richer example with shape inference and negative tests

Those paths cover the key authoring layers without the complexity of transformer or quantization operators.

## Mental Model

A contrib operator usually adds four distinct things:

1. schema and shape inference in `core/graph/contrib_ops`
2. kernel class and compute logic in `onnxruntime/contrib_ops/<provider>`
3. provider kernel table registration
4. focused tests under `onnxruntime/test/contrib_ops`

If one of those layers is missing, the operator may compile but still fail to load, register, infer shapes, or execute.

## Step 1: Define the schema

Add the operator schema in [onnxruntime/core/graph/contrib_ops/contrib_defs.cc](../../onnxruntime/core/graph/contrib_ops/contrib_defs.cc) using `ONNX_MS_OPERATOR_SET_SCHEMA(...)`.

Reference examples:

- `SampleOp` for a minimal pass-through schema
- `ExpandDims` for type constraints plus custom shape inference

Your schema should define:

- operator name and version
- inputs and outputs
- attributes
- type constraints
- shape inference behavior when possible
- operator documentation

This is the source of truth that feeds generated docs such as [docs/ContribOperators.md](../ContribOperators.md).

## Step 2: Add the kernel implementation

Add the provider-specific kernel in the appropriate contrib provider folder.

CPU references:

- [onnxruntime/contrib_ops/cpu/sample.cc](../../onnxruntime/contrib_ops/cpu/sample.cc)
- [onnxruntime/contrib_ops/cpu/sample.h](../../onnxruntime/contrib_ops/cpu/sample.h)
- [onnxruntime/contrib_ops/cpu/expand_dims.cc](../../onnxruntime/contrib_ops/cpu/expand_dims.cc)
- [onnxruntime/contrib_ops/cpu/expand_dims.h](../../onnxruntime/contrib_ops/cpu/expand_dims.h)

For a simple CPU contrib op, this usually means:

- define an `OpKernel` subclass in a header
- implement `Compute()` behavior
- register the kernel with `ONNX_CPU_OPERATOR_*_MS_KERNEL(...)`

Keep the first provider implementation conservative. Get one provider working correctly before expanding to CUDA, WebGPU, JS, or others.

## Step 3: Register the kernel in the provider table

Provider-level contrib registration is not complete until the kernel is added to the provider kernel table.

For CPU, update [onnxruntime/contrib_ops/cpu/cpu_contrib_kernels.cc](../../onnxruntime/contrib_ops/cpu/cpu_contrib_kernels.cc).

Typical work includes:

- forward declaration of the kernel class macro instantiation
- adding `BuildKernelCreateInfo<...>` in the function table

Use `SampleOp` and `ExpandDims` as the minimal examples of this wiring.

If you skip this step, the kernel may exist in source but never become available to the runtime.

## Step 4: Add focused tests

Add tests in [onnxruntime/test/contrib_ops](../../onnxruntime/test/contrib_ops).

Reference examples:

- [onnxruntime/test/contrib_ops/sample_op_test.cc](../../onnxruntime/test/contrib_ops/sample_op_test.cc)
- [onnxruntime/test/contrib_ops/expand_dims_test.cc](../../onnxruntime/test/contrib_ops/expand_dims_test.cc)

Good first-test coverage includes:

- one basic success case
- shape-sensitive behavior if applicable
- invalid input or attribute failures
- provider-specific restrictions only if they are part of the contract

Use `OpTester` for focused operator validation.

## Step 5: Keep schema and kernel responsibilities separate

Use this split consistently:

- schema file owns contract, types, attributes, docs, and shape inference
- kernel file owns runtime computation and input validation during execution
- provider kernel table owns discoverability by that provider
- tests own executable specification of expected behavior

If shape inference can fail before runtime, prefer to encode it in the schema. If a runtime value must be checked during execution, handle it in the kernel.

## Step 6: Validate with the narrowest test slice

After the first substantive edit, run the smallest relevant contrib-op test slice.

Typical targeted loop from the build directory:

Linux:

```bash
./onnxruntime_test_all --gtest_filter="*ExpandDims*:*SampleOp*"
```

Windows:

```powershell
.\onnxruntime_test_all.exe --gtest_filter="*ExpandDims*:*SampleOp*"
```

If you add a new test file or test name, narrow the filter further to the exact operator.

## Design Rules for a New Contrib Op

- start with one provider unless multiple providers are essential
- keep version `1` unless you are intentionally versioning the contrib contract
- write shape inference when it is deterministic and cheap
- add negative tests for invalid axes, shapes, or attributes when those are part of the contract
- avoid overfitting the operator to one model if the behavior is meant to be reusable

## Common Failure Modes

- adding the schema but forgetting provider kernel table registration
- adding the kernel but not the schema, which makes model load or shape inference fail
- putting runtime-only validation into shape inference, or vice versa
- creating a contrib op when an ONNX standard op or existing contrib op already covers the need
- forgetting that [docs/ContribOperators.md](../ContribOperators.md) is generated and should not be edited directly

## Exit Checklist

- [ ] The schema exists in `core/graph/contrib_ops` with doc, type constraints, and shape inference as appropriate.
- [ ] At least one provider kernel exists and is registered in that provider’s contrib kernel table.
- [ ] Focused tests exist under `onnxruntime/test/contrib_ops`.
- [ ] You validated the smallest possible contrib-op test slice locally.