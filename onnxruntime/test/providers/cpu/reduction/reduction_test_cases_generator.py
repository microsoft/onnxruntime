# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import numpy as np


def TestReduction(op, data, axes, keepdims):  # noqa: N802
    if op == "ReduceL1":
        return np.sum(a=np.abs(data), axis=axes, keepdims=keepdims)
    elif op == "ReduceL2":
        return np.sqrt(np.sum(a=np.square(data), axis=axes, keepdims=keepdims))
    elif op == "ReduceLogSum":
        return np.log(np.sum(data, axis=axes, keepdims=keepdims))
    elif op == "ReduceLogSumExp":
        return np.log(np.sum(np.exp(data), axis=axes, keepdims=keepdims))
    elif op == "ReduceMax":
        return np.max(data, axis=axes, keepdims=keepdims)
    elif op == "ReduceMean":
        return np.mean(data, axis=axes, keepdims=keepdims)
    elif op == "ReduceMin":
        return np.min(data, axis=axes, keepdims=keepdims)
    elif op == "ReduceProd":
        return np.prod(data, axis=axes, keepdims=keepdims)
    elif op == "ReduceSum":
        return np.sum(data, axis=axes, keepdims=keepdims)
    elif op == "ReduceSumSquare":
        return np.sum(np.square(data), axis=axes, keepdims=keepdims)
    elif op == "ArgMax":
        axis = axes[0] if axes else 0
        res = np.argmax(data, axis)
        if keepdims:
            res = np.expand_dims(res, axis)
        return res
    elif op == "ArgMin":
        axis = axes[0] if axes else 0
        res = np.argmin(data, axis)
        if keepdims:
            res = np.expand_dims(res, axis)
        return res


def PrintResult(op, axes, keepdims, res):  # noqa: N802
    print('  {"%s",' % op)
    print("OpAttributesResult(")
    print("    // ReductionAttribute")
    print("      {")
    print(" // axes_")
    print("{", end="")
    print(*axes, sep=", ", end="") if axes else print("")
    print("},")
    print(" // keep_dims_")
    print(keepdims, ",")
    print("},")

    print(" // expected dims")
    print("{", end="")
    print(*res.shape, sep=", ", end="")
    print("},")

    print(" // expected values")
    print("{", end="")
    for i in range(res.size):
        print("%5.6ff," % res.item(i))

    print("})},")


def PrintDisableOptimizations():  # noqa: N802
    print("// Optimizations are disabled in this file to improve build throughput")
    print("#if defined(_MSC_VER) || defined(__INTEL_COMPILER)")
    print('#pragma optimize ("", off)')
    print("#elif defined(__GNUC__)")
    print("#if defined(__clang__)")
    print("\t#pragma clang optimize off")
    print("#else")
    print("\t#pragma GCC push_options")
    print('\t#pragma GCC optimize ("O0")')
    print("#endif")
    print("#endif")


def PrintReenableOptimizations():  # noqa: N802
    print("#if defined(_MSC_VER) || defined(__INTEL_COMPILER)")
    print('\t#pragma optimize ("", on)')
    print("#elif defined(__GNUC__)")
    print("#if defined(__clang__)")
    print("\t#pragma clang optimize on")
    print("#else")
    print("\t#pragma GCC pop_options")
    print("#endif")
    print("#endif")


if __name__ == "__main__":
    from itertools import product

    input_shape = [2, 3, 2, 2, 3]
    np.random.seed(0)
    input_data = np.random.uniform(size=input_shape)
    axes_options = [
        (-1, 3),
        (2, 3),
        (2, 1, 4),
        (0, -2, -3),
        (0, 2, 3),
        (0,),
        (2,),
        (4,),
        None,
    ]
    keepdims_options = [0, 1]
    ops = [
        "ReduceL1",
        "ReduceL2",
        "ReduceLogSum",
        "ReduceLogSumExp",
        "ReduceMax",
        "ReduceMean",
        "ReduceMin",
        "ReduceProd",
        "ReduceSum",
        "ReduceSumSquare",
        "ArgMax",
        "ArgMin",
    ]
    print("// Please don't manually edit this file. Generated from reduction_test_cases_generator.py")
    PrintDisableOptimizations()
    print("ReductionTestCases testcases = {")
    print("// input_data")
    print("{")
    for i in range(input_data.size):
        print(
            "%5.6ff," % input_data.item(i),
        )
    print("},")
    print("// input_dims")
    print("{", end="")
    print(*input_shape, sep=", ", end="")
    print("},")

    print("  // map_op_attribute_expected")
    print("{")

    for config in product(axes_options, keepdims_options, ops):
        axes, keepdims, op = config

        # ArgMax and ArgMin only take single axis (default 0)
        skip = False
        if op == "ArgMax" or op == "ArgMin":
            skip = axes is not None and len(axes) > 1

        if not skip:
            res = TestReduction(op, input_data, axes, keepdims)
            PrintResult(op, axes, keepdims, res)

    print("}")
    print("};")
    PrintReenableOptimizations()
