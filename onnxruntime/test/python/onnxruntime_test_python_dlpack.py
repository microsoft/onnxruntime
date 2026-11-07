# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""Inference-only DLPack interchange tests, required by the transformers-enabled CI legs.

PyTorch is mandatory here: missing dependencies or DLPack support must fail, not skip.
The dtype contract below targets the PyTorch 2.10 version installed by build.py.
"""

import ctypes
import gc
import math
import os
import sys
import threading
import time
import unittest
import weakref
from pathlib import Path

import numpy as np
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


class DLDevice(ctypes.Structure):
    _fields_ = (("device_type", ctypes.c_int), ("device_id", ctypes.c_int))


class DLDataType(ctypes.Structure):
    _fields_ = (("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16))


class DLTensor(ctypes.Structure):
    _fields_ = (
        ("data", ctypes.c_void_p),
        ("device", DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    )


class DLManagedTensor(ctypes.Structure):
    _fields_ = (("dl_tensor", DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", ctypes.c_void_p))


def managed_tensor_from_capsule(capsule):
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    return ctypes.cast(get_pointer(capsule, b"dltensor"), ctypes.POINTER(DLManagedTensor))


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
        self.assertFalse(hasattr(C.OrtValueVector, "dlpack_at"))

    def test_numpy_owner_survives_dlpack_export(self):
        for native in (False, True):
            for reinterpret in (False, True):
                for capsule in (False, True):
                    with self.subTest(native=native, reinterpret=reinterpret, capsule=capsule):
                        array = np.arange(6, dtype=np.float32)
                        owner_ref = weakref.ref(array)
                        pointer = array.ctypes.data
                        if reinterpret:
                            value_type = C.OrtValue if native else ort.OrtValue
                            value = value_type.ortvalue_from_numpy_with_onnx_type(array, TensorProto.FLOAT)
                        elif native:
                            value = C.OrtValue.ortvalue_from_numpy(array, C.OrtDevice(C.OrtDevice.cpu(), 0, 0))
                        else:
                            value = ort.OrtValue.ortvalue_from_numpy(array)
                        exported = value.__dlpack__() if capsule else torch.from_dlpack(value)
                        del array, value
                        gc.collect()
                        self.assertIsNotNone(owner_ref())
                        result = torch.from_dlpack(exported) if capsule else exported
                        self.assertEqual(result.data_ptr(), pointer)
                        torch.testing.assert_close(result, torch.arange(6, dtype=torch.float32))
                        result[0] = 7
                        self.assertEqual(owner_ref()[0], 7)
                        del exported, result
                        gc.collect()
                        self.assertIsNone(owner_ref())

    def test_numpy_owner_survives_native_copy_and_vector_export(self):
        array = np.arange(6, dtype=np.float32)
        owner_ref = weakref.ref(array)
        value = ort.OrtValue.ortvalue_from_numpy(array)
        imported = C.OrtValue.from_dlpack(value)
        values = C.OrtValueVector()
        values.push_back(imported)
        del array, value, imported
        gc.collect()
        self.assertIsNotNone(owner_ref())
        result = values.to_dlpacks(torch.from_dlpack)[0]
        del values
        gc.collect()
        torch.testing.assert_close(result, torch.arange(6, dtype=torch.float32))
        del result
        gc.collect()
        self.assertIsNone(owner_ref())

    def test_numpy_noncontiguous_copy_survives_export(self):
        array = np.arange(6, dtype=np.float32)[::-1]
        owner_ref = weakref.ref(array)
        value = ort.OrtValue.ortvalue_from_numpy(array)
        result = torch.from_dlpack(value)
        self.assertNotEqual(result.data_ptr(), array.ctypes.data)
        del array, value
        gc.collect()
        self.assertIsNone(owner_ref())
        torch.testing.assert_close(result, torch.arange(6, dtype=torch.float32).flip(0))

    def test_numpy_owner_release_without_gil_while_main_thread_is_blocked(self):
        for contiguous in (False, True):
            with self.subTest(contiguous=contiguous):
                array = np.arange(6, dtype=np.float32)
                if not contiguous:
                    array = array[::-1]
                released = threading.Event()
                owner_ref = weakref.ref(array, lambda _, released=released: released.set())
                value = ort.OrtValue.ortvalue_from_numpy(array)
                capsule = value.__dlpack__()
                managed = managed_tensor_from_capsule(capsule)
                deleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(DLManagedTensor))(managed.contents.deleter)
                set_name = ctypes.pythonapi.PyCapsule_SetName
                set_name.argtypes = [ctypes.py_object, ctypes.c_char_p]
                set_name.restype = ctypes.c_int
                self.assertEqual(set_name(capsule, b"used_dltensor"), 0)
                del value, array
                gc.collect()
                self.assertEqual(owner_ref() is not None, contiguous)
                observed = []

                def release(deleter=deleter, managed=managed, observed=observed, released=released):
                    # Let the main thread block in join before releasing native storage.
                    time.sleep(0.05)
                    # CFUNCTYPE releases the GIL while calling the actual C++ deleter.
                    deleter(managed)
                    observed.append(released.wait(timeout=5))

                worker = threading.Thread(target=release)
                worker.start()
                worker.join(timeout=10)
                self.assertFalse(worker.is_alive(), "Native destruction must not wait for the main thread.")
                self.assertEqual(observed, [True], "NumPy storage must be released while the main thread is blocked.")
                self.assertIsNone(owner_ref())

    def test_gpu_vendor_metadata_without_gpu_allocation(self):
        new_capsule = ctypes.pythonapi.PyCapsule_New
        new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        new_capsule.restype = ctypes.py_object
        # Empty tensors exercise device metadata without allocating or reading GPU memory.
        shape = (ctypes.c_int64 * 1)(0)
        for device_type in (2, 10):  # kDLCUDA, kDLROCM
            with self.subTest(device_type=device_type):
                provider = "CUDAExecutionProvider" if device_type == 2 else "MIGraphXExecutionProvider"
                device_id = 0 if provider in ort.get_available_providers() else 3
                managed = DLManagedTensor(
                    DLTensor(None, DLDevice(device_type, device_id), 1, DLDataType(2, 32, 1), shape, None, 0),
                    None,
                    None,
                )
                capsule = new_capsule(ctypes.addressof(managed), b"dltensor", None)
                imported = ort.OrtValue.from_dlpack(capsule, synchronize=False)
                exported = None
                result = None
                values = C.OrtValueVector()
                try:
                    self.assertEqual(imported.__dlpack_device__(), (device_type, device_id))
                    for value_type in (ort.OrtValue, C.OrtValue):
                        for producer in (imported, imported._get_c_value()):
                            copied = value_type.from_dlpack(producer, synchronize=False)
                            self.assertEqual(copied.__dlpack_device__(), (device_type, device_id))
                            del copied
                    del producer
                    values.push_back(imported._get_c_value())
                    exported = values.to_dlpacks(None, synchronize=False)
                    for result in (imported.__dlpack__(stream=-1), *exported):
                        device = managed_tensor_from_capsule(result).contents.dl_tensor.device
                        self.assertEqual((device.device_type, device.device_id), (device_type, device_id))
                finally:
                    # The synthetic producer's descriptor must outlive every native owner.
                    del result, imported, exported, values

    def test_import_synchronization_opt_out(self):
        source = torch.arange(6, dtype=torch.float32)
        for value_type in (ort.OrtValue, C.OrtValue):
            with self.subTest(value_type=value_type):
                imported = value_type.from_dlpack(source, synchronize=False)
                self.assert_tensor(torch.from_dlpack(imported), source)
        values = C.OrtValueVector()
        values.push_back(source, synchronize=False)
        self.assert_tensor(values.to_dlpacks(torch.from_dlpack, synchronize=False)[0], source)

    def test_import_opt_out_without_gpu_metadata_omits_stream(self):
        source = torch.arange(6, dtype=torch.float32)

        class Producer:
            def __dlpack__(self):
                return source.__dlpack__()

        class CpuProducer(Producer):
            def __dlpack_device__(self):
                return (1, 0)

        for value_type in (ort.OrtValue, C.OrtValue):
            for producer in (Producer(), CpuProducer()):
                with self.subTest(value_type=value_type, producer=producer):
                    imported = value_type.from_dlpack(producer, synchronize=False)
                    self.assert_tensor(torch.from_dlpack(imported), source)

    def test_unsupported_gpu_vendor_is_not_exported_as_cuda(self):
        array = np.empty(1, dtype=np.float32)
        # A zero-element DirectML-labelled tensor requires no actual GPU memory.
        device = C.OrtDevice(C.OrtDevice.gpu(), C.OrtDevice.default_memory(), 0x1414, 0)
        values = C.OrtValueVector()
        values.push_back_batch([array], [array.ctypes.data], [array.dtype], [[0]], [device])
        value = values[0]
        for export in (value.__dlpack_device__, value.__dlpack__, lambda: values.to_dlpacks(None)):
            with (
                self.subTest(export=export),
                self.assertRaisesRegex(RuntimeError, "supports only NVIDIA CUDA and AMD ROCm"),
            ):
                export()

    def test_numpy_reinterpret_rejects_noncontiguous_storage(self):
        for value_type in (ort.OrtValue, C.OrtValue):
            for array in (np.arange(6, dtype=np.float32)[::-1], np.zeros((2, 3), dtype=np.float32).T):
                with (
                    self.subTest(value_type=value_type, strides=array.strides),
                    self.assertRaisesRegex(ValueError, "C-contiguous NumPy array.*ascontiguousarray"),
                ):
                    value_type.ortvalue_from_numpy_with_onnx_type(array, TensorProto.FLOAT)

    def test_cpu_stream_argument(self):
        for value in (
            ort.OrtValue.ortvalue_from_shape_and_type([1], TensorProto.FLOAT),
            C.OrtValue.from_dlpack(torch.ones(1)),
        ):
            with self.subTest(value_type=type(value)):
                torch.from_dlpack(value.__dlpack__(stream=None))
                for stream in (-1, 0, 1, 2, False, "invalid"):
                    with self.subTest(stream=stream), self.assertRaisesRegex(BufferError, "require stream=None"):
                        value.__dlpack__(stream=stream)

    def test_vector_capsules_are_independent(self):
        values = C.OrtValueVector()
        for offset in (0, 10):
            values.push_back(torch.arange(6, dtype=torch.float32) + offset)
        for consumer in (None, lambda capsule: capsule):
            with self.subTest(consumer=consumer):
                capsules = values.to_dlpacks(consumer)
                self.assertIsNot(capsules[0], capsules[1])
                results = [torch.from_dlpack(capsule) for capsule in capsules]
                for offset, result in zip((0, 10), results, strict=True):
                    torch.testing.assert_close(result, torch.arange(6, dtype=torch.float32) + offset)
                for capsule in capsules:
                    with self.assertRaisesRegex(TypeError, "valid DLPack capsule"):
                        C.OrtValue.from_dlpack(capsule)

    def test_vector_unconsumed_capsules_release_owners(self):
        for consumer in (None, lambda capsule: None):
            with self.subTest(consumer=consumer):
                array = np.arange(6, dtype=np.float32)
                owner_ref = weakref.ref(array)
                values = C.OrtValueVector()
                values.push_back(ort.OrtValue.ortvalue_from_numpy(array)._get_c_value())
                del array
                results = values.to_dlpacks(consumer)
                del values
                gc.collect()
                if consumer is None:
                    self.assertIsNotNone(owner_ref())
                del results
                gc.collect()
                self.assertIsNone(owner_ref())

    def test_vector_consumer_errors_preserve_exception_and_release_owners(self):
        for fail_at in (0, 1):
            for consume_before_error in (False, True):
                with self.subTest(fail_at=fail_at, consume_before_error=consume_before_error):
                    arrays = [np.arange(6, dtype=np.float32) + offset for offset in (0, 10)]
                    owner_refs = [weakref.ref(array) for array in arrays]
                    values = C.OrtValueVector()
                    for array in arrays:
                        values.push_back(ort.OrtValue.ortvalue_from_numpy(array)._get_c_value())
                    del array, arrays
                    error = ValueError("consumer failed")
                    calls = 0

                    def consumer(capsule, fail_at=fail_at, consume_before_error=consume_before_error, error=error):
                        nonlocal calls
                        index = calls
                        calls += 1
                        if index == fail_at:
                            if consume_before_error:
                                tensor = torch.from_dlpack(capsule)
                                self.assertEqual(tensor.numel(), 6)
                            raise error
                        return torch.from_dlpack(capsule)

                    with self.assertRaises(ValueError) as caught:
                        values.to_dlpacks(consumer)
                    self.assertIs(caught.exception, error)
                    self.assertEqual(calls, fail_at + 1)
                    # A traceback may legitimately retain the callback's capsule or tensor.
                    error.__traceback__ = None
                    del values
                    gc.collect()
                    self.assertTrue(all(owner_ref() is None for owner_ref in owner_refs))

    def test_vector_consumer_can_retain_capsule_after_error(self):
        array = np.arange(6, dtype=np.float32)
        owner_ref = weakref.ref(array)
        values = C.OrtValueVector()
        values.push_back(ort.OrtValue.ortvalue_from_numpy(array)._get_c_value())
        retained = []

        def consumer(capsule):
            retained.append(capsule)
            raise ValueError("deferred consumption")

        with self.assertRaisesRegex(ValueError, "deferred consumption"):
            values.to_dlpacks(consumer)
        del array, values
        gc.collect()
        self.assertIsNotNone(owner_ref())
        result = torch.from_dlpack(retained.pop())
        torch.testing.assert_close(result, torch.arange(6, dtype=torch.float32))
        del result
        gc.collect()
        self.assertIsNone(owner_ref())

    def test_vector_consumer_can_mutate_original_vector(self):
        values = C.OrtValueVector()
        values.push_back(torch.ones(1))
        values.push_back(torch.full((1,), 2.0))

        def consumer(capsule):
            for _ in range(20):
                values.push_back(torch.zeros(1))
            return torch.from_dlpack(capsule)

        results = values.to_dlpacks(consumer)
        self.assertEqual(len(results), 2)
        self.assertEqual(len(values), 42)
        self.assertEqual([result.item() for result in results], [1, 2])

    def test_vector_preparation_failure_does_not_invoke_consumer(self):
        array = np.arange(6, dtype=np.float32)
        owner_ref = weakref.ref(array)
        values = C.OrtValueVector()
        values.push_back(ort.OrtValue.ortvalue_from_numpy(array)._get_c_value())
        device = C.OrtDevice(C.OrtDevice.gpu(), C.OrtDevice.default_memory(), 0x1414, 0)
        values.push_back_batch([array], [array.ctypes.data], [array.dtype], [[0]], [device])
        calls = []

        def consumer(capsule):
            calls.append(capsule)
            return torch.from_dlpack(capsule)

        with self.assertRaisesRegex(RuntimeError, "supports only NVIDIA CUDA and AMD ROCm"):
            values.to_dlpacks(consumer)
        self.assertEqual(calls, [])
        del array, values
        gc.collect()
        self.assertIsNone(owner_ref())

    def test_vector_invalid_consumer(self):
        for size in (0, 1):
            values = C.OrtValueVector()
            if size:
                values.push_back(torch.ones(1))
            with self.subTest(size=size), self.assertRaisesRegex(TypeError, "callable consumer or None"):
                values.to_dlpacks(42)
        self.assertEqual(C.OrtValueVector().to_dlpacks(None), [])

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

        class InvalidAttributeDescriptor:
            @property
            def __dlpack__(self):
                raise AttributeError("DLPack descriptor failed")

        for value_type in (ort.OrtValue, C.OrtValue):
            for producer, error in (
                (InvalidProducer(), "DLPack producer failed"),
                (InvalidProtocolAttribute(), "DLPack lookup failed"),
            ):
                with self.subTest(value_type=value_type, producer=producer), self.assertRaisesRegex(ValueError, error):
                    value_type.from_dlpack(producer)
            for synchronize in (False, True):
                with (
                    self.subTest(value_type=value_type, synchronize=synchronize),
                    self.assertRaisesRegex(AttributeError, "DLPack descriptor failed"),
                ):
                    value_type.from_dlpack(InvalidAttributeDescriptor(), synchronize=synchronize)

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


class TestDlpackPluginCallbacks(unittest.TestCase):
    device_id = 32760  # Avoid colliding with a real GPU provider registered by the build.

    def setUp(self):
        library_name = (
            "example_plugin_ep.dll"
            if os.name == "nt"
            else "libexample_plugin_ep.dylib"
            if sys.platform == "darwin"
            else "libexample_plugin_ep.so"
        )
        self.library_path = Path.cwd() / library_name
        if not self.library_path.is_file():
            self.skipTest("requires the example_plugin_ep test library in the build output directory")
        self.library = ctypes.CDLL(str(self.library_path))
        self.current_device = self.library.ExampleEpTestHooks_GetDlpackCurrentDevice
        self.set_current_device = self.library.ExampleEpTestHooks_SetDlpackCurrentDevice
        self.set_current_device.argtypes = [ctypes.c_int]
        self.set_current_device.restype = None
        self.device_entry_count = self.library.ExampleEpTestHooks_GetDlpackDeviceEntryCount
        self.device_entry_count.argtypes = []
        self.device_entry_count.restype = ctypes.c_size_t
        self.sync_count = self.library.ExampleEpTestHooks_GetDlpackSyncCount
        self.release_count = self.library.ExampleEpTestHooks_GetDlpackReleaseCount
        for method in (self.current_device, self.sync_count, self.release_count):
            method.argtypes = []
            method.restype = ctypes.c_int
        self.library.ExampleEpTestHooks_ResetDlpack()
        self.block_sync = self.library.ExampleEpTestHooks_SetDlpackSyncBlocked
        self.block_sync.argtypes = [ctypes.c_int]
        self.block_sync.restype = None
        self.registrations = []
        self.descriptors = []

    def tearDown(self):
        for registration in reversed(getattr(self, "registrations", [])):
            ort.unregister_execution_provider_library(registration)

    def register(self, suffix):
        name = f"example_ep_dlpack_{suffix}"
        ort.register_execution_provider_library(name, str(self.library_path))
        self.registrations.append(name)
        return name

    def unregister(self, name):
        ort.unregister_execution_provider_library(name)
        self.registrations.remove(name)

    def capsule(self, deleter=None):
        shape = (ctypes.c_int64 * 1)(0)
        managed = DLManagedTensor(
            DLTensor(None, DLDevice(2, self.device_id), 1, DLDataType(2, 32, 1), shape, None, 0),
            None,
            ctypes.cast(deleter, ctypes.c_void_p) if deleter is not None else None,
        )
        self.descriptors.append((shape, managed, deleter))
        new_capsule = ctypes.pythonapi.PyCapsule_New
        new_capsule.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
        new_capsule.restype = ctypes.py_object
        return new_capsule(ctypes.addressof(managed), b"dltensor", None)

    def test_complementary_factory_capabilities(self):
        self.register("sync")
        self.register("context")
        test = self

        class Producer:
            def __dlpack_device__(self):
                return (2, test.device_id)

            def __dlpack__(self, *, stream=None):
                test.assertEqual(test.current_device(), test.device_id)
                return test.capsule()

        for value_type in (ort.OrtValue, C.OrtValue):
            value = value_type.from_dlpack(Producer())
            self.assertEqual(self.current_device(), 7)
            before = self.sync_count()
            capsule = value.__dlpack__()
            self.assertEqual(self.sync_count(), before + 1)
            self.assertEqual(value.__dlpack_device__(), (2, self.device_id))
            del capsule, value
        self.assertEqual(self.sync_count(), 4)

    def test_producer_device_changes_are_restored(self):
        self.register("both")
        test = self

        class Producer:
            def __dlpack_device__(self):
                return (2, test.device_id)

            def __dlpack__(self, *, stream=None):
                test.assertEqual(test.current_device(), test.device_id)
                test.set_current_device(9)
                if fail:
                    raise ValueError("Producer changed device before failing")
                return test.capsule()

        for value_type in (ort.OrtValue, C.OrtValue):
            for original in (7, self.device_id):
                for synchronize in (False, True):
                    for fail in (False, True):
                        with self.subTest(value_type=value_type, original=original, synchronize=synchronize, fail=fail):
                            self.set_current_device(original)
                            if fail:
                                with self.assertRaisesRegex(ValueError, "Producer changed device before failing"):
                                    value_type.from_dlpack(Producer(), synchronize=synchronize)
                            else:
                                value_type.from_dlpack(Producer(), synchronize=synchronize)
                            self.assertEqual(self.current_device(), original)

    def test_failed_gpu_synchronization_consumes_capsule_once(self):
        self.register("fail_sync")
        for value_type in (ort.OrtValue, C.OrtValue):
            with self.subTest(value_type=value_type):
                releases = []
                deleter = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(
                    lambda pointer, releases=releases: releases.append(pointer)
                )
                capsule = self.capsule(deleter)
                with self.assertRaisesRegex(RuntimeError, "DLPack test synchronization failed"):
                    value_type.from_dlpack(capsule)
                self.assertEqual(len(releases), 1)
                with self.assertRaisesRegex(TypeError, "valid DLPack capsule"):
                    value_type.from_dlpack(capsule, synchronize=False)
                del capsule
                self.assertEqual(len(releases), 1)
        self.assertEqual(self.sync_count(), 2)

    def test_thread_device_context_is_isolated_and_cleaned(self):
        self.set_current_device(self.device_id)
        self.assertEqual(self.device_entry_count(), 1)
        observed = []

        def change_device():
            try:
                observed.append(self.current_device())
                self.set_current_device(9)
                observed.append(self.current_device())
            finally:
                self.set_current_device(7)

        worker = threading.Thread(target=change_device)
        worker.start()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(observed, [7, 9])
        self.assertEqual(self.current_device(), self.device_id)
        self.assertEqual(self.device_entry_count(), 1)
        worker = threading.Thread(target=lambda: observed.append(self.current_device()))
        worker.start()
        worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(observed, [7, 9, 7])
        self.assertEqual(self.device_entry_count(), 1)

    def test_unregister_inside_producer_pins_factory_until_restoration(self):
        name = self.register("both")
        test = self

        class Producer:
            def __dlpack_device__(self):
                return (2, test.device_id)

            def __dlpack__(self, *, stream=None):
                test.assertEqual(test.current_device(), test.device_id)
                test.unregister(name)
                test.assertEqual(test.release_count(), 0)
                raise ValueError("Producer export failed after unregister")

        with self.assertRaisesRegex(ValueError, "Producer export failed after unregister"):
            ort.OrtValue.from_dlpack(Producer(), synchronize=False)
        self.assertEqual(self.current_device(), 7)
        self.assertEqual(self.release_count(), 1)

    def test_unregister_during_native_wait_pins_factory(self):
        name = self.register("sync")
        capsule = self.capsule()
        results = []
        errors = []

        def import_value():
            try:
                results.append(ort.OrtValue.from_dlpack(capsule))
            except Exception as error:
                errors.append(error)
            finally:
                self.set_current_device(7)

        worker = threading.Thread(target=import_value)
        self.block_sync(1)
        worker.start()
        try:
            deadline = time.monotonic() + 5
            while self.sync_count() == 0 and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertEqual(self.sync_count(), 1)
            self.unregister(name)
            self.assertEqual(self.release_count(), 0)
        finally:
            self.block_sync(0)
            worker.join(timeout=5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertEqual(len(results), 1)
        self.assertEqual(self.release_count(), 1)

    def test_old_factory_callbacks_are_not_read(self):
        self.register("old")
        value = ort.OrtValue.from_dlpack(self.capsule(), synchronize=False)
        with self.assertRaises((BufferError, RuntimeError)):
            value.__dlpack__()
        self.assertEqual(self.sync_count(), 0)
        self.assertEqual(self.current_device(), 7)


@unittest.skipUnless(
    torch.cuda.is_available()
    and ("MIGraphXExecutionProvider" if torch.version.hip else "CUDAExecutionProvider")
    in ort.get_available_providers(),
    "requires PyTorch CUDA/ROCm and the corresponding ORT CUDA/MIGraphX execution provider",
)
class TestDlpackGPU(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two GPU devices")
    def test_protocol_import_restores_current_device(self):
        tensor_device = torch.cuda.device_count() - 1
        device_type = 10 if torch.version.hip else 2
        with torch.cuda.device(tensor_device):
            source = torch.arange(6, dtype=torch.float32, device=f"cuda:{tensor_device}")
            torch.cuda.current_stream().synchronize()

        class InvalidProducer:
            def __dlpack_device__(self):
                return (device_type, tensor_device)

            def __dlpack__(self, *, stream=None):
                if torch.cuda.current_device() != tensor_device:
                    raise AssertionError("Producer's device must be current")
                raise ValueError("Producer export failed")

        for value_type in (ort.OrtValue, C.OrtValue):
            for synchronize in (False, True):
                with self.subTest(value_type=value_type, synchronize=synchronize), torch.cuda.device(0):
                    imported = value_type.from_dlpack(source, synchronize=synchronize)
                    self.assertEqual(torch.cuda.current_device(), 0)
                    self.assertEqual(imported.__dlpack_device__(), (device_type, tensor_device))
                    torch.testing.assert_close(torch.from_dlpack(imported), source)
                    with self.assertRaisesRegex(ValueError, "Producer export failed"):
                        value_type.from_dlpack(InvalidProducer(), synchronize=synchronize)
                    self.assertEqual(torch.cuda.current_device(), 0)

    def test_import_claims_capsule_before_releasing_gil(self):
        source = torch.zeros(6, device="cuda")
        torch.cuda.synchronize()
        is_valid = ctypes.pythonapi.PyCapsule_IsValid
        is_valid.argtypes = [ctypes.py_object, ctypes.c_char_p]
        is_valid.restype = ctypes.c_int
        for value_type in (ort.OrtValue, C.OrtValue):
            with self.subTest(value_type=value_type):
                capsule = source.__dlpack__()
                producer = torch.cuda.Stream()
                with torch.cuda.stream(producer):
                    torch.cuda._sleep(1_000_000_000)
                    source.fill_(7)
                    ready = torch.cuda.Event()
                    ready.record()
                observed = []
                stop = threading.Event()

                def observer(stop=stop, capsule=capsule, observed=observed, ready=ready):
                    while not stop.is_set():
                        if is_valid(capsule, b"used_dltensor"):
                            try:
                                C.OrtValue.from_dlpack(capsule, synchronize=False)
                            except TypeError:
                                observed.append(not ready.query())
                            return
                        time.sleep(0.001)

                thread = threading.Thread(target=observer)
                thread.start()
                try:
                    imported = value_type.from_dlpack(capsule)
                    thread.join(timeout=5)
                finally:
                    stop.set()
                    thread.join()
                self.assertEqual(observed, [True], "Another Python thread must run while the import waits for the GPU.")
                torch.testing.assert_close(torch.from_dlpack(imported), source)

    def test_export_releases_gil_during_device_wait(self):
        source = torch.ones(6, device="cuda")
        value = ort.OrtValue.from_dlpack(source)
        producer = torch.cuda.Stream()
        with torch.cuda.stream(producer):
            torch.cuda._sleep(1_000_000_000)
            source.add_(1)
            ready = torch.cuda.Event()
            ready.record()
        observed = []

        def observer():
            time.sleep(0.05)
            observed.append(not ready.query())

        thread = threading.Thread(target=observer)
        thread.start()
        try:
            capsule = value.__dlpack__()
        finally:
            thread.join()
        self.assertEqual(observed, [True], "Another Python thread must run while the export waits for the GPU.")
        torch.testing.assert_close(torch.from_dlpack(capsule), source)

    def test_import_waits_for_producer_stream(self):
        tensor_device = torch.cuda.device_count() - 1
        producer = torch.cuda.Stream(device=tensor_device)
        for value_type in (ort.OrtValue, C.OrtValue):
            for capsule in (False, True):
                with self.subTest(value_type=value_type, capsule=capsule):
                    with torch.cuda.stream(producer):
                        source = torch.zeros(6, device=f"cuda:{tensor_device}")
                        torch.cuda._sleep(50_000_000)
                        source.fill_(7)
                        ready = torch.cuda.Event()
                        ready.record()
                        exported = source.__dlpack__() if capsule else source
                    with torch.cuda.device(0):
                        imported = value_type.from_dlpack(exported)
                        self.assertEqual(torch.cuda.current_device(), 0)
                    self.assertTrue(ready.query())
                    torch.testing.assert_close(
                        torch.from_dlpack(imported), torch.full((6,), 7.0, device=f"cuda:{tensor_device}")
                    )

    def test_export_waits_for_producer_and_restores_current_device(self):
        tensor_device = torch.cuda.device_count() - 1
        source = torch.zeros(6, device=f"cuda:{tensor_device}")
        public = ort.OrtValue.from_dlpack(source)
        values = C.OrtValueVector()
        values.push_back(public._get_c_value())
        producer = torch.cuda.Stream(device=tensor_device)
        consumer = torch.cuda.Stream(device=tensor_device)
        for export in (
            lambda: public.__dlpack__(stream=0 if torch.version.hip else 1),
            lambda: public._get_c_value().__dlpack__(stream=1 if torch.version.hip else 2),
            lambda: public.__dlpack__(stream=consumer.cuda_stream),
            lambda: public.__dlpack__(),
            lambda: values.to_dlpacks(None)[0],
        ):
            with torch.cuda.stream(producer):
                torch.cuda._sleep(50_000_000)
                source.add_(1)
                ready = torch.cuda.Event()
                ready.record()
            with torch.cuda.device(0):
                capsule = export()
                self.assertEqual(torch.cuda.current_device(), 0)
            self.assertTrue(ready.query())
            with torch.cuda.stream(consumer):
                result = torch.from_dlpack(capsule).clone()
            consumer.synchronize()
            torch.testing.assert_close(result, source)

    def test_no_sync_export_is_capture_safe(self):
        source = torch.ones(6, device="cuda")
        value = ort.OrtValue.from_dlpack(source)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            source.add_(1)
            capsule = value.__dlpack__(stream=-1)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(torch.from_dlpack(capsule), source)

    def test_no_sync_import_and_batch_export_are_capture_safe(self):
        source = torch.ones(6, device="cuda")
        for value_type in (ort.OrtValue, C.OrtValue):
            with self.subTest(value_type=value_type):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    source.add_(1)
                    imported = value_type.from_dlpack(source, synchronize=False)
                    copied = value_type.from_dlpack(imported, synchronize=False)
                    values = C.OrtValueVector()
                    values.push_back(source, synchronize=False)
                    capsules = values.to_dlpacks(None, synchronize=False)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(torch.from_dlpack(imported), source)
                torch.testing.assert_close(torch.from_dlpack(copied), source)
                torch.testing.assert_close(torch.from_dlpack(capsules[0]), source)

    def test_batch_prepares_all_capsules_before_callbacks(self):
        source = torch.ones(6, device="cuda")
        values = C.OrtValueVector()
        values.push_back(source, synchronize=False)
        values.push_back(source, synchronize=False)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        calls = 0

        def consumer(capsule):
            nonlocal calls
            if calls == 0:
                # Any later device-wide fence would invalidate this capture.
                graph.capture_begin()
                source.add_(1)
            calls += 1
            return torch.from_dlpack(capsule)

        with torch.cuda.stream(stream):
            try:
                results = values.to_dlpacks(consumer)
            finally:
                if torch.cuda.is_current_stream_capturing():
                    graph.capture_end()
            graph.replay()
        stream.synchronize()
        self.assertEqual(calls, 2)
        for result in results:
            torch.testing.assert_close(result, source)

    def test_invalid_gpu_stream(self):
        value = ort.OrtValue.from_dlpack(torch.ones(1, device="cuda"))
        for stream in (-2,) if torch.version.hip else (0, -2):
            with self.subTest(stream=stream), self.assertRaisesRegex(ValueError, "Invalid GPU DLPack stream"):
                value.__dlpack__(stream=stream)
        for stream in (False, 1.5, "invalid"):
            with self.subTest(stream=stream), self.assertRaisesRegex(TypeError, "must be an integer"):
                value.__dlpack__(stream=stream)


if __name__ == "__main__":
    unittest.main()
