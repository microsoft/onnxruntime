# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Inference-only DLPack interchange tests, required by the transformers-enabled CI legs.

PyTorch is mandatory here: missing dependencies or DLPack support must fail, not skip.
The dtype contract below targets the PyTorch 2.10 version installed by build.py.
"""

import ctypes
import gc
import math
import unittest
import weakref

import torch
from onnx import TensorProto

import onnxruntime as ort
from onnxruntime.capi import _pybind_state as C

FLOAT8_DTYPES = (
    (torch.float8_e4m3fn, TensorProto.FLOAT8E4M3FN),
    (torch.float8_e4m3fnuz, TensorProto.FLOAT8E4M3FNUZ),
    (torch.float8_e5m2, TensorProto.FLOAT8E5M2),
    (torch.float8_e5m2fnuz, TensorProto.FLOAT8E5M2FNUZ),
    (torch.float8_e8m0fnu, TensorProto.FLOAT8E8M0),
)
INTERCHANGE_DTYPES = (
    (torch.float32, TensorProto.FLOAT),
    (torch.float64, TensorProto.DOUBLE),
    (torch.float16, TensorProto.FLOAT16),
    (torch.bfloat16, TensorProto.BFLOAT16),
    (torch.bool, TensorProto.BOOL),
    (torch.int8, TensorProto.INT8),
    (torch.int16, TensorProto.INT16),
    (torch.int32, TensorProto.INT32),
    (torch.int64, TensorProto.INT64),
    (torch.uint8, TensorProto.UINT8),
    (torch.uint16, TensorProto.UINT16),
    (torch.uint32, TensorProto.UINT32),
    (torch.uint64, TensorProto.UINT64),
    *FLOAT8_DTYPES,
)
PACKED_DTYPES = (
    (torch.int4, TensorProto.INT4, TensorProto.INT8, 4, "Unsupported kInt bits 4"),
    (torch.uint4, TensorProto.UINT4, TensorProto.UINT8, 4, "Unsupported kUInt bits 4"),
    (torch.int2, TensorProto.INT2, TensorProto.INT8, 2, "Unsupported kInt bits 2"),
    (torch.uint2, TensorProto.UINT2, TensorProto.UINT8, 2, "Unsupported kUInt bits 2"),
    (torch.float4_e2m1fn_x2, TensorProto.FLOAT4E2M1, None, 4, "Unsupported kDLFloat4_e2m1fn lanes 1"),
)
SHAPES = ((), (2, 3), (2, 1, 3), (0,), (2, 0, 3))


def tensor_bytes(tensor):
    # Reshape first because PyTorch cannot reinterpret a scalar as a smaller dtype.
    return tensor.reshape(-1).view(torch.uint8)


def make_tensor(dtype, shape):
    count = math.prod(shape)
    if dtype in (entry[0] for entry in FLOAT8_DTYPES):
        return torch.arange(count, dtype=torch.int64).to(torch.uint8).view(dtype).reshape(shape)
    if dtype == torch.bool:
        values = [False, True]
    elif dtype.is_floating_point:
        values = [-2.5, -0.0, 0.0, 1.5, float("inf"), float("nan")]
    else:
        limits = torch.iinfo(dtype)
        values = [limits.min, limits.max, 0, 1, 2, 3]
    return torch.tensor((values * ((count + len(values) - 1) // len(values)))[:count], dtype=dtype).reshape(shape)


class TestDlpackPyTorch(unittest.TestCase):
    def assert_tensor(self, actual, expected):
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        self.assertEqual(actual.device, expected.device)
        if actual.numel():
            # Exact bits cover NaNs, signed zero and shell dtypes without arithmetic.
            # Empty tensors may have zero strides, which PyTorch cannot reinterpret.
            torch.testing.assert_close(tensor_bytes(actual), tensor_bytes(expected), rtol=0, atol=0)

    def test_pytorch_to_ort(self):
        for dtype, onnx_type in INTERCHANGE_DTYPES:
            for shape in SHAPES:
                for capsule in (False, True):
                    with self.subTest(dtype=dtype, shape=shape, capsule=capsule):
                        source = make_tensor(dtype, shape)
                        expected = source.clone()
                        imported = ort.OrtValue.from_dlpack(source.__dlpack__() if capsule else source)
                        self.assertEqual(imported.element_type(), onnx_type)
                        self.assertEqual(imported.shape(), list(shape))
                        self.assertEqual(imported.__dlpack_device__(), (1, 0))
                        if source.numel():
                            self.assertEqual(imported.data_ptr(), source.data_ptr())
                            self.assertEqual(
                                ctypes.string_at(imported.data_ptr(), source.numel() * source.element_size()),
                                tensor_bytes(expected).numpy().tobytes(),
                            )
                        del source
                        gc.collect()
                        result = torch.from_dlpack(imported)
                        del imported
                        gc.collect()
                        self.assert_tensor(result, expected)

    def test_ort_to_pytorch(self):
        for dtype, onnx_type in INTERCHANGE_DTYPES:
            for shape in SHAPES:
                for capsule in (False, True):
                    with self.subTest(dtype=dtype, shape=shape, capsule=capsule):
                        expected = make_tensor(dtype, shape)
                        # ORT owns this allocation independently of the expected PyTorch tensor.
                        source = ort.OrtValue.ortvalue_from_shape_and_type(list(shape), onnx_type)
                        if expected.numel():
                            ctypes.memmove(
                                source.data_ptr(), expected.data_ptr(), expected.numel() * expected.element_size()
                            )
                        result = torch.from_dlpack(source.__dlpack__() if capsule else source)
                        self.assert_tensor(result, expected)
                        if expected.numel():
                            self.assertEqual(result.data_ptr(), source.data_ptr())
                            tensor_bytes(result)[0] ^= 1
                            tensor_bytes(expected)[0] ^= 1
                            self.assertEqual(
                                ctypes.string_at(source.data_ptr(), 1), tensor_bytes(expected)[:1].numpy().tobytes()
                            )
                        source_ref = weakref.ref(source)
                        del source
                        gc.collect()
                        self.assertIsNone(source_ref())
                        self.assert_tensor(result, expected)

    def test_float8_all_bit_patterns(self):
        for dtype, onnx_type in FLOAT8_DTYPES:
            with self.subTest(dtype=dtype):
                source = torch.arange(256, dtype=torch.int64).to(torch.uint8).view(dtype).reshape(16, 16)
                imported = ort.OrtValue.from_dlpack(source)
                self.assertEqual(imported.element_type(), onnx_type)
                self.assert_tensor(torch.from_dlpack(imported), source)
                owned = ort.OrtValue.ortvalue_from_shape_and_type([16, 16], onnx_type)
                ctypes.memmove(owned.data_ptr(), source.data_ptr(), 256)
                self.assert_tensor(torch.from_dlpack(owned), source)

    def test_capsule_keeps_ort_allocation_alive(self):
        for dtype, onnx_type in INTERCHANGE_DTYPES:
            with self.subTest(dtype=dtype):
                expected = make_tensor(dtype, (2, 3))
                source = ort.OrtValue.ortvalue_from_shape_and_type([2, 3], onnx_type)
                pointer = source.data_ptr()
                ctypes.memmove(pointer, expected.data_ptr(), expected.numel() * expected.element_size())
                capsule = source.__dlpack__()
                source_ref = weakref.ref(source)
                del source
                gc.collect()
                self.assertIsNone(source_ref())
                result = torch.from_dlpack(capsule)
                self.assertEqual(result.data_ptr(), pointer)
                self.assert_tensor(result, expected)

    def test_ortvalue_dlpack_surface(self):
        for value_type in (ort.OrtValue, C.OrtValue):
            with self.subTest(value_type=value_type):
                self.assertFalse(hasattr(value_type, "to_dlpack"))
                for method in ("from_dlpack", "__dlpack__", "__dlpack_device__"):
                    self.assertTrue(callable(getattr(value_type, method)))

    def test_native_ortvalue_protocol(self):
        for dtype, onnx_type in INTERCHANGE_DTYPES:
            with self.subTest(dtype=dtype):
                source = make_tensor(dtype, (2, 3))
                native = C.OrtValue.from_dlpack(source)
                self.assertEqual(native.element_type(), onnx_type)
                self.assertEqual(native.data_ptr(), source.data_ptr())
                self.assertEqual(native.__dlpack_device__(), (1, 0))
                self.assert_tensor(torch.from_dlpack(native), source)
                public = ort.OrtValue.from_dlpack(native)
                self.assertEqual(public.data_ptr(), source.data_ptr())
                imported = C.OrtValue.from_dlpack(public)
                self.assertEqual(imported.data_ptr(), source.data_ptr())
                self.assert_tensor(torch.from_dlpack(imported), source)
                values = C.OrtValueVector()
                values.push_back(source)
                self.assertEqual(values[0].data_ptr(), source.data_ptr())
                self.assert_tensor(torch.from_dlpack(values[0]), source)
                values.push_back(public)
                for consumer in (torch.from_dlpack, C.OrtValue.from_dlpack):
                    for result in values.to_dlpacks(consumer):
                        self.assertEqual(result.data_ptr(), source.data_ptr())
                        self.assert_tensor(torch.from_dlpack(result), source)

    def test_from_dlpack_propagates_producer_errors(self):
        class InvalidProducer:
            def __dlpack__(self):
                raise ValueError("DLPack producer failed")

        class InvalidProtocolAttribute:
            @property
            def __dlpack__(self):
                raise ValueError("DLPack lookup failed")

        for value_type in (ort.OrtValue, C.OrtValue):
            for producer, error in (
                (InvalidProducer(), "DLPack producer failed"),
                (InvalidProtocolAttribute(), "DLPack lookup failed"),
            ):
                with self.subTest(value_type=value_type, producer=producer), self.assertRaisesRegex(ValueError, error):
                    value_type.from_dlpack(producer)

    def test_packed_dtype_boundaries(self):
        for dtype, onnx_type, storage_type, bits, error in PACKED_DTYPES:
            with self.subTest(dtype=dtype):
                # ORT's shape counts logical sub-byte elements, including an odd tail.
                source = ort.OrtValue.ortvalue_from_shape_and_type([5], onnx_type)
                num_bytes = (5 * bits + 7) // 8
                ctypes.memset(source.data_ptr(), 0x35, num_bytes)
                imported = ort.OrtValue.from_dlpack(source)
                self.assertEqual(imported.element_type(), onnx_type)
                self.assertEqual(imported.shape(), [5])
                self.assertEqual(imported.data_ptr(), source.data_ptr())
                with self.assertRaisesRegex(BufferError, error):
                    torch.from_dlpack(source)
                del source
                gc.collect()
                self.assertEqual(ctypes.string_at(imported.data_ptr(), num_bytes), b"\x35" * num_bytes)

                pytorch_source = torch.arange(5, dtype=torch.uint8).view(dtype)
                if storage_type is None:
                    # PyTorch float4 has two lanes per element; ORT requires one.
                    with self.assertRaisesRegex(RuntimeError, "ORT does not support lanes != 1"):
                        ort.OrtValue.from_dlpack(pytorch_source)
                else:
                    # PyTorch int2/int4 shell dtypes export as full bytes, not packed elements.
                    imported = ort.OrtValue.from_dlpack(pytorch_source)
                    self.assertEqual(imported.element_type(), storage_type)
                    self.assertEqual(imported.shape(), [5])
                    self.assertEqual(imported.data_ptr(), pytorch_source.data_ptr())
                    self.assertEqual(ctypes.string_at(imported.data_ptr(), 5), bytes(range(5)))

    def test_contiguous_slice_and_singleton_strides(self):
        sources = (
            torch.arange(20, dtype=torch.float32)[5:11].reshape(2, 3),
            torch.empty_strided((2, 1, 3), (3, 99, 1), dtype=torch.float32).fill_(2),
        )
        for source in sources:
            with self.subTest(shape=source.shape, strides=source.stride()):
                self.assertTrue(source.is_contiguous())
                imported = ort.OrtValue.from_dlpack(source)
                self.assertEqual(imported.data_ptr(), source.data_ptr())
                self.assert_tensor(torch.from_dlpack(imported), source)

    def test_noncontiguous_input_is_rejected(self):
        source = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
        with self.assertRaisesRegex(RuntimeError, "ORT only supports contiguous tensor"):
            ort.OrtValue.from_dlpack(source)


if __name__ == "__main__":
    unittest.main()
