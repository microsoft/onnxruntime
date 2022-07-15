# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# pylint: disable=missing-docstring

import unittest

import numpy as np
import onnxruntime_pybind11_state as torch_ort
import torch
from parameterized import parameterized

# OPS - is a list of  list of [test_operator, the tested_tensor].
# The default value for tested_tensor is torch.rand (6)- size of 6 uniform distribution on the interval [0, 1).
# for floor and erf the ort- produce a roundoff error for floor(NaN), compare to cpu- stay NaN.  thus, nan_to_num change nan to zero.
ops = [
    ["abs", torch.tensor([-1, -2, 3, -6, -7])],
    ["acos"],
    ["acosh"],
    ["asinh"],
    ["atanh"],
    ["asin"],
    ["atan"],
    ["ceil"],
    ["cos"],
    ["cosh"],
    ["erf", torch.nan_to_num(torch.rand(5))],
    ["exp"],
    ["floor", torch.nan_to_num(torch.rand(5))],
    ["log"],
    ["reciprocal"],
    ["neg"],
    ["round"],
    ["relu"],
    ["sigmoid"],
    ["sin"],
    ["sinh"],
    ["sqrt"],
    ["tan"],
    ["tanh"],
]
# the following unary ops not been tested:
# ["isnan",       torch.tensor([1, float('nan'), 2])]]
# ["selu",        torch.randn(10)]]
# ["sign",        ]]
# ["hardsigmoid", ],
# ["isinf",       ],
# ["det"          ]]


def rename_func_to_op(testcase_func, param_num, param):
    return f"test_{parameterized.to_safe_name(str(param.args[0]))}"


def rename_func_to_inplace(testcase_func, param_num, param):
    return f"test_{parameterized.to_safe_name(str(param.args[0]))}_"


def rename_func_to_out(testcase_func, param_num, param):
    return f"test_{parameterized.to_safe_name(str(param.args[0]))}_out"


class OrtOpTests(unittest.TestCase):
    """test cases for supported eager ops"""

    def get_device(self):
        return torch_ort.device()

    def test_add(self):
        device = self.get_device()
        cpu_ones = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        ort_ones = cpu_ones.to(device)
        cpu_twos = cpu_ones + cpu_ones
        ort_twos = ort_ones + ort_ones
        assert torch.allclose(cpu_twos, ort_twos.cpu())

    def test_type_promotion_add(self):
        device = self.get_device()
        x = torch.ones(2, 5, dtype=torch.int64)
        y = torch.ones(2, 5, dtype=torch.float32)
        ort_x = x.to(device)
        ort_y = y.to(device)
        ort_z = ort_x + ort_y
        assert ort_z.dtype == torch.float32
        assert torch.allclose(ort_z.cpu(), (x + y))

    def test_add_alpha(self):
        device = self.get_device()
        cpu_ones = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        ort_ones = cpu_ones.to(device)
        assert torch.allclose(torch.add(cpu_ones, cpu_ones, alpha=2.5), torch.add(ort_ones, ort_ones, alpha=2.5).cpu())

    # the onnx operator Mul does not support type bool. The following test verifies cpu fall back works.
    def test_mul_bool(self):
        device = self.get_device()
        cpu_ones = torch.ones(3, 3, dtype=bool)
        ort_ones = cpu_ones.to(device)
        assert torch.allclose(torch.mul(cpu_ones, cpu_ones), torch.mul(ort_ones, ort_ones).cpu())

    # TODO: Add BFloat16 test coverage
    def test_add_(self):
        device = self.get_device()
        cpu_ones = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        ort_ones = cpu_ones.to(device)
        cpu_twos = cpu_ones
        cpu_twos += cpu_ones
        ort_twos = ort_ones
        ort_twos += ort_ones
        assert torch.allclose(cpu_twos, ort_twos.cpu())

    def test_sin_(self):
        device = self.get_device()
        cpu_sin_pi_ = torch.Tensor([np.pi])
        torch.sin_(cpu_sin_pi_)
        ort_sin_pi_ = torch.Tensor([np.pi]).to(device)
        torch.sin_(ort_sin_pi_)
        cpu_sin_pi = torch.sin(torch.Tensor([np.pi]))
        ort_sin_pi = torch.sin(torch.Tensor([np.pi]).to(device))
        assert torch.allclose(cpu_sin_pi, ort_sin_pi.cpu())
        assert torch.allclose(cpu_sin_pi_, ort_sin_pi_.cpu())
        assert torch.allclose(ort_sin_pi.cpu(), ort_sin_pi_.cpu())

    def test_sin(self):
        device = self.get_device()
        cpu_sin_pi = torch.sin(torch.Tensor([np.pi]))
        ort_sin_pi = torch.sin(torch.Tensor([np.pi]).to(device))
        assert torch.allclose(cpu_sin_pi, ort_sin_pi.cpu())

    def test_zero_like(self):
        device = self.get_device()
        ones = torch.ones((10, 10), dtype=torch.float32)
        cpu_zeros = torch.zeros_like(ones)
        ort_zeros = torch.zeros_like(ones.to(device))
        assert torch.allclose(cpu_zeros, ort_zeros.cpu())

    def test_gemm(self):
        device = self.get_device()
        cpu_ones = torch.Tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        ort_ones = cpu_ones.to(device)
        cpu_ans = cpu_ones * 4
        ort_ans = torch_ort.custom_ops.gemm(ort_ones, ort_ones, ort_ones, 1.0, 1.0, 0, 0)
        assert torch.allclose(cpu_ans, ort_ans.cpu())

    def test_batchnormalization_inplace(self):
        device = self.get_device()
        x = torch.Tensor([[[[-1, 0, 1]], [[2.0, 3.0, 4.0]]]]).to(device)
        s = torch.Tensor([1.0, 1.5]).to(device)
        bias = torch.Tensor([0.0, 1.0]).to(device)
        mean = torch.Tensor([0.0, 3.0]).to(device)
        var = torch.Tensor([1.0, 1.5]).to(device)
        y, mean_out, var_out = torch_ort.custom_ops.batchnorm_inplace(x, s, bias, mean, var, 1e-5, 0.9)
        assert torch.allclose(x.cpu(), y.cpu()), "x != y"
        assert torch.allclose(mean.cpu(), mean_out.cpu()), "mean != mean_out"
        assert torch.allclose(var.cpu(), var_out.cpu()), "var != var_out"

    def test_max(self):
        cpu_tensor = torch.rand(10, 10)
        ort_tensor = cpu_tensor.to("ort")
        ort_min = ort_tensor.max()
        cpu_min = cpu_tensor.max()
        assert torch.allclose(cpu_min, ort_min.cpu())
        assert cpu_min.dim() == ort_min.dim()

    def test_min(self):
        cpu_tensor = torch.rand(10, 10)
        ort_tensor = cpu_tensor.to("ort")
        ort_min = ort_tensor.min()
        cpu_min = cpu_tensor.min()
        assert torch.allclose(cpu_min, ort_min.cpu())
        assert cpu_min.dim() == ort_min.dim()

    def test_equal(self):
        device = self.get_device()
        cpu_a = torch.Tensor([1.0, 1.5])
        ort_a = cpu_a.to(device)
        cpu_b = torch.Tensor([1.0, 1.5])
        ort_b = cpu_b.to(device)
        cpu_c = torch.Tensor([1.0, 1.8])
        ort_c = cpu_c.to(device)
        cpu_d = torch.Tensor([1.0, 1.5, 2.1])
        ort_d = cpu_d.to(device)
        cpu_e = torch.Tensor([[1.0, 1.5]])
        ort_e = cpu_e.to(device)

        # a = b
        assert torch.equal(cpu_a, cpu_b)
        assert torch.equal(ort_a, ort_b)
        # a != c based on one value
        assert not torch.equal(cpu_a, cpu_c)
        assert not torch.equal(ort_a, ort_c)
        # a != d because size of dim 1 is not equal
        assert not torch.equal(cpu_a, cpu_d)
        assert not torch.equal(ort_a, ort_d)
        # a != e because dim does not match
        assert not torch.equal(cpu_a, cpu_e)
        assert not torch.equal(ort_a, ort_e)

    def test_ones(self):
        device = self.get_device()
        cpu_out_tensor = torch.tensor([])
        ort_out_tensor = cpu_out_tensor.to(device)
        cpu_ones = torch.ones((10, 10), out=cpu_out_tensor)
        ort_ones = cpu_ones.to(device)
        ort_ones_device = torch.ones((10, 10), out=ort_out_tensor, device=device)
        assert torch.allclose(cpu_ones, ort_ones.cpu())
        assert torch.allclose(cpu_ones, ort_ones_device.cpu())
        assert torch.allclose(cpu_out_tensor, ort_out_tensor.cpu())

    def test_narrow(self):
        cpu_tensor = torch.rand(10, 10)
        cpu_narrow = cpu_tensor.narrow(0, 5, 5)
        ort_narrow = cpu_narrow.to("ort")
        assert torch.allclose(cpu_narrow, ort_narrow.cpu())

    def test_zero_stride(self):
        device = self.get_device()
        cpu_tensor = torch.empty_strided(size=(6, 1024, 512), stride=(0, 0, 0))
        assert cpu_tensor.storage().size() == 1
        ort_tensor_copied = cpu_tensor.to(device)
        assert torch.allclose(cpu_tensor, ort_tensor_copied.cpu())
        ort_tensor = torch.empty_strided(size=(6, 1024, 512), stride=(0, 0, 0), device=device)
        assert ort_tensor.is_ort
        assert ort_tensor.stride() == (0, 0, 0)
        cpu_tensor_copied = ort_tensor.cpu()
        assert cpu_tensor_copied.stride() == (0, 0, 0)

    def test_softmax(self):
        device = self.get_device()
        cpu_tensor = torch.rand(3, 5)
        ort_tensor = cpu_tensor.to(device)
        cpu_result = torch.softmax(cpu_tensor, dim=1)
        ort_result = torch.softmax(ort_tensor, dim=1)
        assert torch.allclose(cpu_result, ort_result.cpu())

    def test_addmm(self):
        device = self.get_device()
        size = 4
        ort_tensor = torch.ones([size, size]).to(device)
        input_bias = torch.ones([size]).to(device)
        output = torch.addmm(input_bias, ort_tensor, ort_tensor)
        expected = torch.ones([size, size]) * 5
        assert torch.equal(output.to("cpu"), expected)

    def test_argmax(self):
        device = self.get_device()
        cpu_tensor = torch.rand(3, 5)
        ort_tensor = cpu_tensor.to(device)
        cpu_result = torch.argmax(cpu_tensor, dim=1)
        ort_result = torch.argmax(ort_tensor, dim=1)
        assert torch.allclose(cpu_result, ort_result.cpu())
        assert cpu_result.dim() == ort_result.dim()

    def test_masked_select(self):
        device = self.get_device()
        cpu_tensor = torch.randn(3, 4)
        cpu_mask = cpu_tensor.ge(0.5)
        ort_tensor = cpu_tensor.to(device)
        ort_mask = cpu_mask.to(device)
        cpu_result = cpu_tensor.masked_select(cpu_mask)
        ort_result = ort_tensor.masked_select(ort_mask)
        assert torch.allclose(cpu_result, ort_result.cpu())
        assert cpu_result.dim() == ort_result.dim()

    def test_masked_select_broadcast(self):
        device = self.get_device()
        cpu_tensor = torch.randn(3, 4)
        cpu_mask = torch.tensor([[0], [1], [1]], dtype=bool)
        ort_tensor = cpu_tensor.to(device)
        ort_mask = cpu_mask.to(device)
        cpu_result = cpu_tensor.masked_select(cpu_mask)
        ort_result = ort_tensor.masked_select(ort_mask)
        assert torch.allclose(cpu_result, ort_result.cpu())
        assert cpu_result.dim() == ort_result.dim()

    def test_bitwise_and(self):
        device = self.get_device()
        cpu_a = torch.tensor([[0], [1], [1]], dtype=bool)
        cpu_b = torch.tensor([[1], [0], [1]], dtype=bool)
        ort_a = cpu_a.to(device)
        ort_b = cpu_b.to(device)
        cpu_result = torch.bitwise_and(cpu_a, cpu_b)
        ort_result = torch.bitwise_and(ort_a, ort_b)
        assert torch.equal(cpu_result, ort_result.cpu())

    def test_bitwise_and_fallback(self):
        device = self.get_device()
        # use randint because bitwise_and is not supported on floats
        cpu_a = torch.randint(200, (3, 4))
        cpu_b = torch.randint(200, (3, 4))
        ort_a = cpu_a.to(device)
        ort_b = cpu_b.to(device)
        cpu_result = torch.bitwise_and(cpu_a, cpu_b)
        ort_result = torch.bitwise_and(ort_a, ort_b)
        assert torch.equal(cpu_result, ort_result.cpu())

    # @parameterized.expand generate test methods for ops and using name_func we renaming the test to be test_{ops}
    @parameterized.expand(ops, name_func=rename_func_to_op)
    def test_op(self, test_name, tensor_test=torch.rand(6)):
        # compile eval- creates a code object that evaluates the operator (for example torch.abs(tensor_test)) and returns its result.
        cpu_result = eval(compile("torch." + test_name + "(tensor_test)", "<string>", "eval"))
        ort_result = eval(compile("torch." + test_name + "(tensor_test.to(self.get_device()))", "<string>", "eval"))
        assert torch.allclose(cpu_result, ort_result.cpu(), equal_nan=True)

    @parameterized.expand(ops, name_func=rename_func_to_inplace)
    def test_op_inplace(self, test_name, tensor_test=torch.rand(6)):
        device = self.get_device()

        cpu_tensor = tensor_test
        ort_tensor = cpu_tensor.to(device)

        eval(compile("torch." + test_name + "_(cpu_tensor)", "<string>", "eval"))
        eval(compile("torch." + test_name + "_(ort_tensor)", "<string>", "eval"))

        assert torch.allclose(cpu_tensor, ort_tensor.cpu(), equal_nan=True)

    @parameterized.expand(ops, name_func=rename_func_to_out)
    def test_op_out(self, test_name, tensor_test=torch.rand(6)):
        ##relu -don't have output
        if test_name == "relu":
            self.skipTest(f"no {test_name}_output")
        ### troubleshoot later: the following tests are Failing.
        if test_name == "asin" or test_name == "log" or test_name == "atanh":
            self.skipTest(f" {test_name}_output Fails - skipping for now")
        device = self.get_device()
        cpu_tensor = tensor_test
        ort_tensor = cpu_tensor.to(device)

        cpu_out_tensor = torch.tensor([], dtype=tensor_test.dtype)
        ort_out_tensor = cpu_out_tensor.to(device)

        st_cpu = f"torch.{test_name}(cpu_tensor, out=cpu_out_tensor)"
        st_ort = f"torch.{test_name}(ort_tensor, out=ort_out_tensor)"
        cpu_result = eval(compile(st_cpu, "<string>", "eval"))
        ort_result = eval(compile(st_ort, "<string>", "eval"))

        assert torch.allclose(cpu_result, ort_result.cpu(), equal_nan=True)
        assert torch.allclose(cpu_out_tensor, ort_out_tensor.cpu(), equal_nan=True)
        assert torch.allclose(ort_result.cpu(), ort_out_tensor.cpu(), equal_nan=True)

    def test_resize(self):
        device = self.get_device()

        sizes = [[1], [1, 1], [2, 2], [1, 4]]

        # Basic resize from empty Tensor
        for size in sizes:
            torch_size = torch.Size(size)
            cpu_tensor = torch.tensor([])
            ort_tensor = torch.tensor([]).to(device)

            cpu_tensor.resize_(torch_size)
            ort_tensor.resize_(torch_size)

            self.assertEqual(cpu_tensor.size(), ort_tensor.size())

        # Validate cases where we resize from a non-empty tensor
        # to a larger tensor
        cpu_tensor = torch.tensor([1.0, 2.0])
        ort_tensor = cpu_tensor.to(device)

        cpu_tensor.resize_(torch.Size([3]))
        ort_tensor.resize_(torch.Size([3]))

        self.assertEqual(cpu_tensor.size(), ort_tensor.size())
        self.assertTrue(torch.allclose(cpu_tensor[:2], ort_tensor.cpu()[:2]))

        # Validate case when calling resize with current shape & size
        cpu_tensor = torch.tensor([1.0, 2.0])
        ort_tensor = cpu_tensor.to(device)

        cpu_tensor.resize_(torch.Size([2]))
        ort_tensor.resize_(torch.Size([2]))

        self.assertEqual(cpu_tensor.size(), ort_tensor.size())
        self.assertTrue(torch.allclose(cpu_tensor, ort_tensor.cpu()))

        # Validate case when calling resize with different shape but same size
        cpu_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        ort_tensor = cpu_tensor.to(device)

        cpu_tensor.resize_(torch.Size([1, 4]))
        ort_tensor.resize_(torch.Size([1, 4]))

        self.assertEqual(cpu_tensor.size(), ort_tensor.size())
        self.assertTrue(torch.allclose(cpu_tensor, ort_tensor.cpu()))

        # Validate cases where we resize from a non-empty tensor
        # to a smaller tensor
        cpu_tensor = torch.tensor([1.0, 2.0])
        ort_tensor = cpu_tensor.to(device)

        cpu_tensor.resize_(torch.Size([1]))
        ort_tensor.resize_(torch.Size([1]))

        self.assertEqual(cpu_tensor.size(), ort_tensor.size())
        self.assertTrue(torch.allclose(cpu_tensor, ort_tensor.cpu()))

    def test_abs_out(self):
        device = self.get_device()
        cpu_tensor = torch.tensor([-1, -2, 3, -6, -7])
        ort_tensor = cpu_tensor.to(device)

        cpu_out_tensor = torch.tensor([], dtype=torch.long)
        ort_out_tensor = cpu_out_tensor.to(device)

        cpu_result = torch.abs(cpu_tensor, out=cpu_out_tensor)
        ort_result = torch.abs(ort_tensor, out=ort_out_tensor)

        assert torch.equal(cpu_result, ort_result.cpu())
        assert torch.equal(cpu_out_tensor, ort_out_tensor.cpu())
        assert torch.equal(ort_result.cpu(), ort_out_tensor.cpu())

    def test_eq_tensor(self):
        device = self.get_device()
        cpu_a = torch.Tensor([1.0, 1.5, 2.0])
        ort_a = cpu_a.to(device)
        cpu_b = torch.Tensor([1.0, 1.5, 2.1])
        ort_b = cpu_b.to(device)

        for tensor_type in {torch.float, torch.bool}:
            for func in {"eq", "ne"}:
                print(f"Testing {func} with type {tensor_type}")
                cpu_out_tensor = torch.tensor([], dtype=tensor_type)
                ort_out_tensor = cpu_out_tensor.to(device)
                cpu_a_b_eq_result = eval(
                    compile("torch." + func + "(cpu_a, cpu_b, out=cpu_out_tensor)", "<string>", "eval")
                )
                ort_a_b_eq_result = eval(
                    compile("torch." + func + "(ort_a, ort_b, out=ort_out_tensor)", "<string>", "eval")
                )
                assert torch.equal(cpu_a_b_eq_result.to(device), ort_a_b_eq_result)
                assert torch.equal(cpu_out_tensor, ort_out_tensor.to("cpu"))
                assert ort_out_tensor.dtype == tensor_type

    def test_eq_scalar(self):
        device = self.get_device()
        cpu_tensor_int = torch.tensor([1, 1], dtype=torch.int32)
        cpu_scalar_int = torch.scalar_tensor(1, dtype=torch.int)
        cpu_scalar_int_not = torch.scalar_tensor(2, dtype=torch.int)
        cpu_tensor_float = torch.tensor([1.1, 1.1], dtype=torch.float32)
        cpu_scalar_float = torch.scalar_tensor(1.1, dtype=torch.float32)
        cpu_scalar_float_not = torch.scalar_tensor(1.0, dtype=torch.float32)

        ort_tensor_int = cpu_tensor_int.to(device)
        ort_scalar_int = cpu_scalar_int.to(device)
        ort_scalar_int_not = cpu_scalar_int_not.to(device)
        ort_tensor_float = cpu_tensor_float.to(device)
        ort_scalar_float = cpu_scalar_float.to(device)
        ort_scalar_float_not = cpu_scalar_float_not.to(device)

        # compare int to int, float to float - ort only supports same type at the moment
        cpu_out_tensor = torch.tensor([], dtype=torch.bool)
        ort_out_tensor = cpu_out_tensor.to(device)

        for func in {"eq", "ne"}:
            cpu_int_int_result = eval(
                compile("torch." + func + "(cpu_tensor_int, cpu_scalar_int, out=cpu_out_tensor)", "<string>", "eval")
            )
            cpu_int_int_not_result = eval(
                compile("torch." + func + "(cpu_tensor_int, cpu_scalar_int_not)", "<string>", "eval")
            )
            cpu_float_float_result = eval(
                compile("torch." + func + "(cpu_tensor_float, cpu_scalar_float)", "<string>", "eval")
            )
            cpu_float_float_not_result = eval(
                compile("torch." + func + "(cpu_tensor_float, cpu_scalar_float_not)", "<string>", "eval")
            )

            ort_int_int_result = eval(
                compile("torch." + func + "(ort_tensor_int, ort_scalar_int, out=ort_out_tensor)", "<string>", "eval")
            )
            ort_int_int_not_result = eval(
                compile("torch." + func + "(ort_tensor_int, ort_scalar_int_not)", "<string>", "eval")
            )
            ort_float_float_result = eval(
                compile("torch." + func + "(ort_tensor_float, ort_scalar_float)", "<string>", "eval")
            )
            ort_float_float_not_result = eval(
                compile("torch." + func + "(ort_tensor_float, ort_scalar_float_not)", "<string>", "eval")
            )

            assert torch.equal(cpu_out_tensor, ort_out_tensor.to("cpu"))
            assert torch.equal(cpu_int_int_result, ort_int_int_result.to("cpu"))
            assert torch.equal(cpu_int_int_not_result, ort_int_int_not_result.to("cpu"))
            assert torch.equal(cpu_float_float_result, ort_float_float_result.to("cpu"))
            assert torch.equal(cpu_float_float_not_result, ort_float_float_not_result.to("cpu"))

    def test_fill(self):
        device = self.get_device()
        for torch_type in [torch.int, torch.float]:
            cpu_tensor = torch.zeros(2, 2, dtype=torch_type)
            ort_tensor = cpu_tensor.to(device)
            for value in [True, 1.1, -1, 0]:
                cpu_tensor.fill_(value)
                ort_tensor.fill_(value)
                assert cpu_tensor.dtype == ort_tensor.dtype
                assert torch.equal(cpu_tensor, ort_tensor.to("cpu"))

    # tests both nonzero and nonzero.out
    def test_nonzero(self):
        device = self.get_device()

        for cpu_tensor in [
            torch.tensor([[[-1, 0, 1], [0, 1, 0]], [[0, 1, 0], [-1, 0, 1]]], dtype=torch.long),
            torch.tensor([[[-1, 0, 1], [0, 1, 0]], [[0, 1, 0], [-1, 0, 1]]], dtype=torch.float),
        ]:
            ort_tensor = cpu_tensor.to(device)

            cpu_out_tensor = torch.tensor([], dtype=torch.long)
            ort_out_tensor = cpu_out_tensor.to(device)

            # nonzero.out
            cpu_result = torch.nonzero(cpu_tensor, out=cpu_out_tensor)
            ort_result = torch.nonzero(ort_tensor, out=ort_out_tensor)
            assert torch.equal(cpu_out_tensor, ort_out_tensor.to("cpu"))
            assert torch.equal(cpu_result, ort_result.to("cpu"))

            # nonzero
            cpu_result = torch.nonzero(cpu_tensor)
            ort_result = torch.nonzero(ort_tensor)
            assert torch.equal(cpu_result, ort_result.to("cpu"))

            # check result between nonzero.out and nonzero
            assert torch.equal(ort_result.to("cpu"), ort_out_tensor.to("cpu"))


if __name__ == "__main__":
    unittest.main()
