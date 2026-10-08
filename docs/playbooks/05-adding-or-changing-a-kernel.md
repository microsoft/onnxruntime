# Playbook 05: Adding or Changing a Kernel

## Outcome

By the end of this playbook, you will be able to trace an operator from kernel registration to compute logic to tests, and make a focused kernel change with the right validation loop.

This playbook assumes you have already completed [Playbook 02](02-build-test-and-debug-locally.md) and [Playbook 04](04-session-lifecycle-from-load-to-run.md).

## Start Here

- [onnxruntime/core/providers/cpu/math/element_wise_ops.cc](../../onnxruntime/core/providers/cpu/math/element_wise_ops.cc)
- [onnxruntime/core/providers/cpu/math/element_wise_ops.h](../../onnxruntime/core/providers/cpu/math/element_wise_ops.h)
- [onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc](../../onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc)
- [onnxruntime/test/providers/provider_test_utils.h](../../onnxruntime/test/providers/provider_test_utils.h)

## Pick the Smallest Real Example

For a first kernel-oriented change, use an existing CPU math kernel such as `Add` as your reference path.

That path is useful because it shows all three layers clearly:

1. registration macros in [onnxruntime/core/providers/cpu/math/element_wise_ops.cc](../../onnxruntime/core/providers/cpu/math/element_wise_ops.cc)
2. kernel class declarations in [onnxruntime/core/providers/cpu/math/element_wise_ops.h](../../onnxruntime/core/providers/cpu/math/element_wise_ops.h)
3. behavioral tests in [onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc](../../onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc)

## Mental Model

Most kernel work falls into one of these categories:

- add support for a new type or opset in an existing operator
- fix correctness for shapes, broadcasting, attributes, or edge cases
- add a provider-specific implementation for an existing operator
- add validation or error handling for unsupported input cases

Do not start with provider-wide registry plumbing. Start at one operator and follow its owning files.

## Step 1: Locate registration

In [onnxruntime/core/providers/cpu/math/element_wise_ops.cc](../../onnxruntime/core/providers/cpu/math/element_wise_ops.cc), look for the operator registration macros.

For `Add`, the file shows:

- versioned registrations for opsets 7 to 13
- non-versioned registrations for opset 14+
- one registration per supported type

That tells you three things immediately:

- which opsets are handled separately
- which types are supported
- which kernel class is instantiated for each registration

If your change is only about registration coverage, this file may be the only source change you need.

## Step 2: Find the kernel class

In [onnxruntime/core/providers/cpu/math/element_wise_ops.h](../../onnxruntime/core/providers/cpu/math/element_wise_ops.h), find the kernel class declaration.

For `Add`, the core shape is:

- `template <typename T> class Add final : public OpKernel`
- constructor for any one-time setup
- `Status Compute(OpKernelContext* context) const override`

This is the boundary between registration and runtime behavior.

When changing kernel behavior, identify first whether the logic belongs in:

- constructor-time validation
- `Compute()` input/output handling
- shared helper or functor code used by multiple kernels

## Step 3: Check neighboring kernels before editing

Before changing code, inspect one nearby operator that solves a similar problem.

Good examples in [onnxruntime/core/providers/cpu/math/element_wise_ops.h](../../onnxruntime/core/providers/cpu/math/element_wise_ops.h):

- `Div` for constructor-time validation of constant integer divisors
- unary elementwise kernels for functor-based implementations

This helps you match local style and avoid inventing a one-off pattern.

## Step 4: Add or adjust tests first when possible

Use [onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc](../../onnxruntime/test/providers/cpu/math/element_wise_ops_test.cc) as the main example.

The `Add` tests already cover:

- multiple scalar and tensor types
- broadcasting shapes
- zero-dimension behavior
- provider-specific exclusions when relevant

Use [onnxruntime/test/providers/provider_test_utils.h](../../onnxruntime/test/providers/provider_test_utils.h) and `OpTester` for targeted kernel tests.

Typical pattern:

```cpp
OpTester test("Add", 14);
test.AddInput<int32_t>("A", {3}, {1, 2, 3});
test.AddInput<int32_t>("B", {3}, {4, 5, 6});
test.AddOutput<int32_t>("C", {3}, {5, 7, 9});
test.Run();
```

If the bug is already reproduced by an existing test, use that as your focused validation loop.

## Step 5: Make the smallest kernel change

Prefer the narrowest change that matches the failure mode:

- missing type support: update registration and add tests for the new type
- wrong broadcast behavior: change compute logic and add one focused broadcast regression test
- invalid input handling: add or tighten validation and test the failure message
- provider-specific difference: limit the change to the relevant provider directory and tests

Do not widen into unrelated ops while you are still validating the first kernel change.

## Step 6: Run focused validation immediately

After the first substantive edit, run the smallest test slice that can falsify the change.

Typical targeted loop from the build directory:

Linux:

```bash
./onnxruntime_provider_test --gtest_filter="*MathOpTest*Add*"
```

Windows:

```powershell
.\onnxruntime_provider_test.exe --gtest_filter="*MathOpTest*Add*"
```

If your change is narrower than all `Add` tests, use an even tighter filter based on the specific test name you added.

## How to Decide Where a Kernel Change Belongs

- If the operator is already listed and routed, start in the kernel source file, not provider initialization code.
- If the operator is not claimed at all, start in registration.
- If execution reaches the kernel but results are wrong, start in `Compute()` or shared helpers.
- If only one provider is affected, stay inside that provider’s directory until proven otherwise.

## Common Failure Modes

- Editing a kernel implementation when the real issue is missing registration for an opset or type.
- Adding a broad test when one small `OpTester` case would isolate the problem faster.
- Changing multiple operators in the same file before validating the first one.
- Forgetting that provider-specific tests may need exclusions or provider-specific execution setup.

## Exit Checklist

- [ ] You can point to the registration macro for the operator you changed.
- [ ] You know where the kernel class and `Compute()` implementation live.
- [ ] You added or updated a focused test near existing operator tests.
- [ ] You ran the narrowest possible test filter after the change.