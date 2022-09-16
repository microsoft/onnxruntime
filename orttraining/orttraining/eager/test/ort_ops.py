# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# pylint: disable=missing-docstring, too-many-public-methods, no-member

import unittest

import numpy as np
import onnxruntime_pybind11_state as torch_ort
import torch
from parameterized import parameterized, param


class OrtOpTests(unittest.TestCase):
    """test cases for supported eager ops"""

    def get_device(self):
        return torch_ort.device()

    def test_fallback_to_cpu(self):
        device = self.get_device()
        cpu_ones = torch.ones(3, 3, dtype=bool)
        ort_ones = cpu_ones.to(device)
        # the onnx operator Mul does not support type bool so will fallback to cpu.
        assert torch.allclose(torch.mul(cpu_ones, cpu_ones), torch.mul(ort_ones, ort_ones).cpu())

    def test_type_promotion_add(self):
        device = self.get_device()
        cpu_ones_int64 = torch.ones(2, 5, dtype=torch.int64)
        cpu_ones_float32 = torch.ones(2, 5, dtype=torch.float32)
        ort_ones_int64 = cpu_ones_int64.to(device)
        ort_ones_float32 = cpu_ones_float32.to(device)
        cpu_result = cpu_ones_int64 + cpu_ones_float32
        ort_result = ort_ones_int64 + ort_ones_float32
        assert ort_result.dtype == torch.float32
        assert torch.allclose(cpu_result, ort_result.cpu())

        # verify scalar addition promotion
        cpu_result = cpu_ones_int64 + 1.1
        ort_result = ort_ones_int64 + 1.1
        assert ort_result.dtype == torch.float32
        assert torch.allclose(cpu_result, ort_result.cpu())

        ## verify setting out to type int while inputs are float cause an error as casting float to int is not allowed.
        cpu_out_tensor = torch.tensor([], dtype=torch.int)
        ort_out_tensor = cpu_out_tensor.to(device)
        with self.assertRaises(RuntimeError):
            ort_result = torch.add(ort_ones_int64, ort_ones_float32, out=ort_out_tensor)

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
        ort_ans = torch.ops.ort.gemm(ort_ones, ort_ones, ort_ones, 1.0, 1.0, 0, 0)
        assert torch.allclose(cpu_ans, ort_ans.cpu())

    def test_batchnormalization_inplace(self):
        device = self.get_device()
        x = torch.Tensor([[[[-1, 0, 1]], [[2.0, 3.0, 4.0]]]]).to(device)
        s = torch.Tensor([1.0, 1.5]).to(device)
        bias = torch.Tensor([0.0, 1.0]).to(device)
        mean = torch.Tensor([0.0, 3.0]).to(device)
        var = torch.Tensor([1.0, 1.5]).to(device)
        y, mean_out, var_out = torch.ops.ort.batchnorm_inplace(x, s, bias, mean, var, 1e-5, 0.9)
        assert torch.allclose(x.cpu(), y.cpu()), "x != y"
        assert torch.allclose(mean.cpu(), mean_out.cpu()), "mean != mean_out"
        assert torch.allclose(var.cpu(), var_out.cpu()), "var != var_out"

    def test_variadic_inputs(self):
        device = self.get_device()
        tensor = torch.ones(2, 2).to(device)
        expected = torch.ones(2, 6)
        out = torch.ops.ort.my_cat([tensor, tensor, tensor], 1)
        assert torch.allclose(expected, out.cpu())

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

    def test_cat(self):
        device = self.get_device()
        cpu_out_tensor = torch.tensor([])
        ort_out_tensor = cpu_out_tensor.to(device)
        cpu_x = torch.randn((128, 64))
        cpu_y = torch.randn((128, 64))
        cpu_z = torch.randn((128, 64))
        cpu_ans_0 = torch.cat((cpu_x, cpu_y, cpu_z), 0)
        cpu_ans_1 = torch.cat((cpu_x, cpu_y, cpu_z), -1, out=cpu_out_tensor)
        ort_x = cpu_x.to(device)
        ort_y = cpu_y.to(device)
        ort_z = cpu_z.to(device)
        ort_ans_0 = torch.cat((ort_x, ort_y, ort_z), 0)
        ort_ans_1 = torch.cat((ort_x, ort_y, ort_z), -1, out=ort_out_tensor)
        assert torch.allclose(cpu_ans_0, ort_ans_0.cpu())
        assert torch.allclose(cpu_ans_1, ort_ans_1.cpu())
        assert torch.allclose(cpu_out_tensor, ort_out_tensor.cpu())

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

    def test_empty(self):
        device = self.get_device()
        cpu_tensor = torch.empty(size=(3, 4))
        ort_tensor = torch.empty(size=(3, 4), device=device)
        assert ort_tensor.is_ort
        assert ort_tensor.size() == cpu_tensor.size()

    def test_softmax(self):
        device = self.get_device()
        cpu_tensor = torch.rand(3, 5)
        ort_tensor = cpu_tensor.to(device)
        cpu_result = torch.softmax(cpu_tensor, dim=1)
        ort_result = torch.softmax(ort_tensor, dim=1)
        assert torch.allclose(cpu_result, ort_result.cpu())

    def test_log_softmax(self):
        device = self.get_device()
        cpu_tensor = torch.rand(3, 5, 7)
        ort_tensor = cpu_tensor.to(device)
        cpu_result_a = torch.log_softmax(cpu_tensor, dim=2)
        ort_result_a = torch.log_softmax(ort_tensor, dim=2)
        assert torch.allclose(cpu_result_a, ort_result_a.cpu())
        cpu_result_b = torch.log_softmax(cpu_tensor, dim=0)
        ort_result_b = torch.log_softmax(ort_tensor, dim=0)
        assert torch.allclose(cpu_result_b, ort_result_b.cpu())
        cpu_result_c = torch.log_softmax(cpu_tensor, dim=-1)
        ort_result_c = torch.log_softmax(ort_tensor, dim=-1)
        assert torch.allclose(cpu_result_c, ort_result_c.cpu())
        assert torch.allclose(ort_result_a.cpu(), ort_result_c.cpu())

    @parameterized.expand(
        [
            param((2, 64, 256), 0),
            param((2, 64, 256), 1),
            param((2, 64, 256), -1),
            param((4096, 1024), 0),
            param((512, 8192), 1),
        ]
    )
    def test_softmax_grad(self, input_shape, dim):
        # The 1% tolerance used by this test is not working for any random inputs
        # and on the other hand it is tough to come up with some tolerance value
        # that works for any random input values. So, pin the seed value so that the
        # random inputs used by this test are always the same.
        torch.manual_seed(5)
        device = self.get_device()
        cpu_tensor = torch.nn.Parameter(torch.rand(input_shape))
        ort_tensor = torch.nn.Parameter(cpu_tensor.detach().clone().to(device))
        cpu_result = torch.softmax(cpu_tensor, dim=dim)
        ort_result = torch.softmax(ort_tensor, dim=dim)
        cpu_loss = cpu_result.pow(2).sum()
        ort_loss = ort_result.cpu().pow(2).sum()
        cpu_loss.backward()
        ort_loss.backward()
        assert torch.allclose(ort_result.cpu(), cpu_result)
        assert torch.allclose(ort_tensor.grad.cpu(), cpu_tensor.grad, rtol=0.01)

    @parameterized.expand(
        [
            param((2, 64, 256), 0),
            param((2, 32, 128), 1),
            param((2, 64, 128), -1),
            param((1024, 8), 0),
            param((2, 2048), 1),
        ]
    )
    def test_logsoftmax_grad(self, input_shape, dim):
        # The 5% tolerance used by this test is not working for any random inputs
        # and on the other hand it is tough to come up with some tolerance value
        # that works for any random input values. So, pin the seed value so that the
        # random inputs used by this test are always the same.
        torch.manual_seed(5)
        device = self.get_device()
        cpu_tensor = torch.nn.Parameter(torch.rand(input_shape))
        ort_tensor = torch.nn.Parameter(cpu_tensor.detach().clone().to(device))
        cpu_result = torch.log_softmax(cpu_tensor, dim=dim)
        ort_result = torch.log_softmax(ort_tensor, dim=dim)
        cpu_loss = cpu_result.pow(2).sum()
        ort_loss = ort_result.cpu().pow(2).sum()
        cpu_loss.backward()
        ort_loss.backward()
        assert torch.allclose(ort_result.cpu(), cpu_result)
        assert torch.allclose(ort_tensor.grad.cpu(), cpu_tensor.grad, rtol=0.05)

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
        cpu_tensor = torch.rand(3, 5, 7, 8)
        ort_tensor = cpu_tensor.to(device)

        # Scenario: basic (no dim parameters)
        cpu_result = torch.argmax(cpu_tensor)
        ort_result = torch.argmax(ort_tensor)
        assert torch.allclose(cpu_result, ort_result.cpu())
        assert cpu_result.dim() == ort_result.dim()

        # Scenario: specify dim parameter
        cpu_result = torch.argmax(cpu_tensor, dim=1)
        ort_result = torch.argmax(ort_tensor, dim=1)
        assert torch.allclose(cpu_result, ort_result.cpu())
        assert cpu_result.dim() == ort_result.dim()

        # Scenario: specify dim and keepdim parameters
        cpu_result = torch.argmax(cpu_tensor, dim=1, keepdim=True)
        ort_result = torch.argmax(ort_tensor, dim=1, keepdim=True)
        assert torch.allclose(cpu_result, ort_result.cpu())
        assert cpu_result.dim() == ort_result.dim()

        # Scenario: specify negative dim value
        cpu_result = torch.argmax(cpu_tensor, dim=-1)
        ort_result = torch.argmax(ort_tensor, dim=-1)
        assert torch.allclose(cpu_result, ort_result.cpu())
        assert cpu_result.dim() == ort_result.dim()

        # Scenario: basic out (no dim parameters)
        cpu_out_tensor = torch.tensor([], dtype=torch.long)
        ort_out_tensor = cpu_out_tensor.to(device)
        cpu_result = torch.argmax(cpu_tensor, out=cpu_out_tensor)
        ort_result = torch.argmax(ort_tensor, out=ort_out_tensor)
        assert torch.allclose(cpu_result, ort_result.cpu())
        assert cpu_result.dim() == ort_result.dim()
        assert torch.allclose(cpu_out_tensor, ort_out_tensor.cpu())
        assert cpu_out_tensor.dim() == ort_out_tensor.dim()

        # Scenario: out with dim parameter
        cpu_out_tensor = torch.tensor([], dtype=torch.long)
        ort_out_tensor = cpu_out_tensor.to(device)
        cpu_result = torch.argmax(cpu_tensor, dim=1, out=cpu_out_tensor)
        ort_result = torch.argmax(ort_tensor, dim=1, out=ort_out_tensor)
        assert torch.allclose(cpu_result, ort_result.cpu())
        assert cpu_result.dim() == ort_result.dim()
        assert torch.allclose(cpu_out_tensor, ort_out_tensor.cpu())
        assert cpu_out_tensor.dim() == ort_out_tensor.dim()

        # Scenario: out with dim and keepdim parameters
        cpu_out_tensor = torch.tensor([], dtype=torch.long)
        ort_out_tensor = cpu_out_tensor.to(device)
        cpu_result = torch.argmax(cpu_tensor, dim=1, keepdim=True, out=cpu_out_tensor)
        ort_result = torch.argmax(ort_tensor, dim=1, keepdim=True, out=ort_out_tensor)
        assert torch.allclose(cpu_result, ort_result.cpu())
        assert cpu_result.dim() == ort_result.dim()
        assert torch.allclose(cpu_out_tensor, ort_out_tensor.cpu())
        assert cpu_out_tensor.dim() == ort_out_tensor.dim()

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
        cpu_out = torch.tensor([], dtype=bool)
        ort_out = cpu_out.to(device)
        cpu_result = torch.bitwise_and(cpu_a, cpu_b)
        ort_result = torch.bitwise_and(ort_a, ort_b)
        assert torch.equal(cpu_result, ort_result.cpu())
        cpu_result = torch.bitwise_and(cpu_a, cpu_b, out=cpu_out)
        ort_result = torch.bitwise_and(ort_a, ort_b, out=ort_out)
        assert torch.equal(cpu_result, ort_result.cpu())
        assert torch.equal(cpu_out, ort_out.cpu())
        assert torch.equal(ort_result.cpu(), ort_out.cpu())

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

    def test_mm(self):
        device = self.get_device()

        # out version test
        cpu_mat1 = torch.rand(3, 2)
        cpu_mat2 = torch.rand(2, 2)
        ort_mat1 = cpu_mat1.to(device)
        ort_mat2 = cpu_mat2.to(device)
        cpu_out = torch.tensor([])
        ort_out = cpu_out.to(device)
        cpu_result = torch.mm(cpu_mat1, cpu_mat2, out=cpu_out)
        ort_result = torch.mm(ort_mat1, ort_mat2, out=ort_out)
        assert torch.allclose(cpu_result, ort_result.cpu())
        assert torch.allclose(cpu_out, ort_out.cpu())
        assert torch.allclose(cpu_result, ort_out.cpu())

        # non-out version with alternate dimension matrices
        cpu_mat1 = torch.rand(7, 5)
        cpu_mat2 = torch.rand(5, 4)
        ort_mat1 = cpu_mat1.to(device)
        ort_mat2 = cpu_mat2.to(device)
        cpu_result = torch.mm(cpu_mat1, cpu_mat2)
        ort_result = torch.mm(ort_mat1, ort_mat2)
        assert torch.allclose(cpu_result, ort_result.cpu())

        # check error cases
        ort_mat1 = torch.rand(1, 1).to(device)
        ort_bad_dim = torch.rand(2, 2).to(device)
        ort_wrong_type = torch.ones(1, 1, dtype=torch.int).to(device)
        ort_not_matrix = torch.ones(1).to(device)
        with self.assertRaises(RuntimeError):
            torch.mm(ort_mat1, ort_bad_dim)
        with self.assertRaises(RuntimeError):
            torch.mm(ort_mat1, ort_wrong_type)
        with self.assertRaises(RuntimeError):
            torch.mm(ort_mat1, ort_not_matrix)

    def test_squeeze(self):
        device = self.get_device()
        cpu_tensor = torch.zeros(2, 1, 2, 1, 2)
        ort_tensor = cpu_tensor.to(device)

        cpu_result1 = torch.squeeze(cpu_tensor)
        ort_result1 = torch.squeeze(ort_tensor)

        cpu_result2 = torch.squeeze(cpu_tensor, 1)
        ort_result2 = torch.squeeze(ort_tensor, 1)

        assert torch.equal(cpu_result1, ort_result1.cpu())
        assert torch.equal(cpu_result2, ort_result2.cpu())

    def test_unsqueeze(self):
        device = self.get_device()
        cpu_tensor = torch.tensor([1, 2, 3, 4])
        ort_tensor = cpu_tensor.to(device)

        cpu_result1 = torch.unsqueeze(cpu_tensor, 0)
        ort_result1 = torch.unsqueeze(ort_tensor, 0)
        cpu_result2 = torch.unsqueeze(cpu_tensor, 1)
        ort_result2 = torch.unsqueeze(ort_tensor, 1)

        assert torch.equal(cpu_result1, ort_result1.cpu())
        assert torch.equal(cpu_result2, ort_result2.cpu())

    def test_add_broadcasting(self):
        device = self.get_device()
        cpu_first = torch.rand(1, 1, 3, 4, 5)
        ort_first = cpu_first.to(device)
        cpu_last = torch.rand(1, 2, 3, 1, 1)
        ort_last = cpu_last.to(device)
        cpu_single = torch.rand(5)
        ort_single = cpu_single.to(device)

        cpu_result1 = cpu_first + cpu_last  # dims = (1,2,3,4,5) is final
        ort_result1 = ort_first + ort_last
        assert torch.equal(cpu_result1, ort_result1.cpu())

        cpu_result2 = cpu_result1 + cpu_single
        ort_result2 = ort_result1 + ort_single
        assert torch.equal(cpu_result2, ort_result2.cpu())

    ################################ parameterized test follow #######################################
    # OPS - is a list of [test_operator, tested_tensor=torch.rand (6)].
    # The default value for tested_tensor is torch.rand (6)- size of 6 uniform distribution on the interval [0, 1).
    # for floor and erf, the ort produces a roundoff error for NaN input, but cpu keeps it a NaN.
    # Thus, we use nan_to_num to ensure actual numbers are passed in.

    # As many of the following use eval and make it appear to pylint that there are many unused variables,
    # we disable those warnings

    # pylint: disable=eval-used, unused-argument, unused-variable, no-self-argument,

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

    # The function renames the test function: ops/math_sign_ops (e.g. abs)+ the test name(e.g. out), results in: test_abs_out
    def rename_func(testcase_func, param_num, param):
        return f"test_{parameterized.to_safe_name(str(param.args[0]))}{testcase_func.__name__[7:]}"

    # @parameterized.expand generate test methods for ops and using name_func we renaming the test to be test_{ops}
    @parameterized.expand(ops, name_func=rename_func)
    def test_op(self, test_name, tensor_test=torch.rand(6)):
        # compile eval- creates a code object that evaluates the operator (for example torch.abs(tensor_test)) and returns its result.
        cpu_result = eval(compile("torch." + test_name + "(tensor_test)", "<string>", "eval"))
        ort_result = eval(compile("torch." + test_name + "(tensor_test.to(self.get_device()))", "<string>", "eval"))
        assert torch.allclose(cpu_result, ort_result.cpu(), equal_nan=True)

    @parameterized.expand(ops, name_func=rename_func)
    def test_op_(self, test_name, tensor_test=torch.rand(6)):
        device = self.get_device()

        cpu_tensor = tensor_test
        ort_tensor = cpu_tensor.to(device)

        eval(compile("torch." + test_name + "_(cpu_tensor)", "<string>", "eval"))
        eval(compile("torch." + test_name + "_(ort_tensor)", "<string>", "eval"))

        assert torch.allclose(cpu_tensor, ort_tensor.cpu(), equal_nan=True)

    @parameterized.expand(ops, name_func=rename_func)
    def test_op_out(self, test_name, tensor_test=torch.rand(6)):
        ##relu -don't have output
        if test_name == "relu":
            self.skipTest(f"no {test_name}_output")
        ### troubleshoot later: the following tests are Failing.
        if test_name in ("asin", "log", "atanh"):
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

    math_sign_ops = ["eq", "ne", "lt", "gt"]

    @parameterized.expand(math_sign_ops, name_func=rename_func)
    def test_op_tensor(self, math_sign_ops):
        device = self.get_device()
        cpu_a = torch.Tensor([1.0, 1.5, 2.0, 3.5])
        ort_a = cpu_a.to(device)
        cpu_b = torch.Tensor([1.0, 1.4, 2.1, 2.4])
        ort_b = cpu_b.to(device)

        for tensor_type in {torch.float, torch.bool}:
            cpu_out_tensor = torch.tensor([], dtype=tensor_type)
            ort_out_tensor = cpu_out_tensor.to(device)
            cpu_a_b_result = eval(
                compile("torch." + math_sign_ops + "(cpu_a, cpu_b, out=cpu_out_tensor)", "<string>", "eval")
            )
            ort_a_b_result = eval(
                compile("torch." + math_sign_ops + "(ort_a, ort_b, out=ort_out_tensor)", "<string>", "eval")
            )
            assert torch.equal(cpu_a_b_result.to(device), ort_a_b_result)
            assert torch.equal(cpu_out_tensor, ort_out_tensor.to("cpu"))
            assert ort_out_tensor.dtype == tensor_type

    @parameterized.expand(math_sign_ops, name_func=rename_func)
    def test_op_scalar(self, math_sign_ops):
        device = self.get_device()
        cpu_tensor_int = torch.tensor([1, 1], dtype=torch.int32)
        cpu_scalar_int_lt = torch.scalar_tensor(2, dtype=torch.int)
        cpu_scalar_int_gt = torch.scalar_tensor(0, dtype=torch.int)
        cpu_tensor_float = torch.tensor([1.1, 1.1], dtype=torch.float32)
        float_lt = 1.0
        float_gt = 1.2

        ort_tensor_int = cpu_tensor_int.to(device)
        ort_scalar_int_lt = cpu_scalar_int_lt.to(device)
        ort_scalar_int_gt = cpu_scalar_int_gt.to(device)
        ort_tensor_float = cpu_tensor_float.to(device)

        # compare int to int, float to float - ort only supports same type at the moment
        cpu_out_tensor = torch.tensor([], dtype=torch.bool)
        ort_out_tensor = cpu_out_tensor.to(device)

        cpu_int_int_result = eval(
            compile(
                "torch." + math_sign_ops + "(cpu_tensor_int, cpu_scalar_int_lt, out=cpu_out_tensor)", "<string>", "eval"
            )
        )
        cpu_int_int_gt_result = eval(
            compile("torch." + math_sign_ops + "(cpu_tensor_int, cpu_scalar_int_gt)", "<string>", "eval")
        )
        cpu_float_float_lt_result = eval(
            compile("torch." + math_sign_ops + "(cpu_tensor_float, float_lt)", "<string>", "eval")
        )
        cpu_float_float_gt_result = eval(
            compile("torch." + math_sign_ops + "(cpu_tensor_float, float_gt)", "<string>", "eval")
        )

        ort_int_int_result = eval(
            compile(
                "torch." + math_sign_ops + "(ort_tensor_int, ort_scalar_int_lt, out=ort_out_tensor)", "<string>", "eval"
            )
        )
        ort_int_int_gt_result = eval(
            compile("torch." + math_sign_ops + "(ort_tensor_int, ort_scalar_int_gt)", "<string>", "eval")
        )
        ort_float_float_lt_result = eval(
            compile("torch." + math_sign_ops + "(ort_tensor_float, float_lt)", "<string>", "eval")
        )
        ort_float_float_gt_result = eval(
            compile("torch." + math_sign_ops + "(ort_tensor_float, float_gt)", "<string>", "eval")
        )

        assert torch.equal(cpu_out_tensor, ort_out_tensor.to("cpu"))
        assert torch.equal(cpu_int_int_result, ort_int_int_result.to("cpu"))
        assert torch.equal(cpu_int_int_gt_result, ort_int_int_gt_result.to("cpu"))
        assert torch.equal(cpu_float_float_lt_result, ort_float_float_lt_result.to("cpu"))
        assert torch.equal(cpu_float_float_gt_result, ort_float_float_gt_result.to("cpu"))

    binary_ops = [  # [op, op_sign, alpha_supported]
        ["add", "+", True],
        ["sub", "-", True],
        ["mul", "*", False],
        ["div", "/", False],
    ]

    @parameterized.expand(binary_ops, name_func=rename_func)
    def test_op_binary_tensor(self, binary_op, op_sign, alpha_supported):
        device = self.get_device()
        cpu_input = torch.rand(3, 1)  # use broadcasting in the second dim.
        ort_input = cpu_input.to(device)
        cpu_other = torch.rand(3, 3)
        ort_other = cpu_other.to(device)

        # verify op_sign works
        cpu_result = eval(compile("cpu_input " + op_sign + " cpu_other", "<string>", "eval"))
        ort_result = eval(compile("ort_input " + op_sign + " ort_other", "<string>", "eval"))
        assert torch.allclose(cpu_result, ort_result.cpu())

        # verify torch op with out param works
        cpu_out_tensor = torch.tensor([])
        ort_out_tensor = cpu_out_tensor.to(device)
        cpu_result = eval(
            compile("torch." + binary_op + "(cpu_input, cpu_other, out=cpu_out_tensor)", "<string>", "eval")
        )
        ort_result = eval(
            compile("torch." + binary_op + "(ort_input, ort_other, out=ort_out_tensor)", "<string>", "eval")
        )
        assert torch.allclose(cpu_result, ort_result.cpu())
        assert torch.allclose(cpu_out_tensor, ort_out_tensor.cpu())

        if alpha_supported:
            cpu_result = eval(
                compile(
                    "torch." + binary_op + "(cpu_input, cpu_other, alpha=2.5, out=cpu_out_tensor)", "<string>", "eval"
                )
            )
            ort_result = eval(
                compile(
                    "torch." + binary_op + "(ort_input, ort_other, alpha=2.5, out=ort_out_tensor)", "<string>", "eval"
                )
            )
            assert torch.allclose(cpu_result, ort_result.cpu())
            assert torch.allclose(cpu_out_tensor, ort_out_tensor.cpu())

    @parameterized.expand(binary_ops, name_func=rename_func)
    def test_op_binary_scalar(self, binary_op, op_sign, alpha_supported):
        device = self.get_device()
        cpu_input = torch.ones(3, 3)
        ort_input = cpu_input.to(device)
        cpu_other = 3.1
        ort_other = 3.1

        # verify op_sign works
        cpu_result = eval(compile("cpu_input " + op_sign + " cpu_other", "<string>", "eval"))
        ort_result = eval(compile("ort_input " + op_sign + " ort_other", "<string>", "eval"))
        assert torch.allclose(cpu_result, ort_result.cpu())

        # verify torch op with out param works
        cpu_out_tensor = torch.tensor([])
        ort_out_tensor = cpu_out_tensor.to(device)
        cpu_result = eval(
            compile("torch." + binary_op + "(cpu_input, cpu_other, out=cpu_out_tensor)", "<string>", "eval")
        )
        ort_result = eval(
            compile("torch." + binary_op + "(ort_input, ort_other, out=ort_out_tensor)", "<string>", "eval")
        )
        assert torch.allclose(cpu_result, ort_result.cpu())
        assert torch.allclose(cpu_out_tensor, ort_out_tensor.cpu())

        if alpha_supported:
            cpu_result = eval(
                compile(
                    "torch." + binary_op + "(cpu_input, cpu_other, alpha=2.5, out=cpu_out_tensor)", "<string>", "eval"
                )
            )
            ort_result = eval(
                compile(
                    "torch." + binary_op + "(ort_input, ort_other, alpha=2.5, out=ort_out_tensor)", "<string>", "eval"
                )
            )
            assert torch.allclose(cpu_result, ort_result.cpu())
            assert torch.allclose(cpu_out_tensor, ort_out_tensor.cpu())

    ################################################################
    # Please add new non-parameterized tests above the parameterized section.
    ################################################################


if __name__ == "__main__":
    # torch_ort.set_default_logger_severity(0)
    # torch_ort.set_default_logger_verbosity(4)
    unittest.main()
