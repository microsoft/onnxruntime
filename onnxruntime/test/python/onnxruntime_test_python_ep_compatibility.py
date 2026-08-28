# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import platform
import sys
import tempfile
import unittest

import onnx

from onnxruntime.capi.onnxruntime_pybind11_state import (
    OrtCompiledModelCompatibility,
    OrtDeviceEpIncompatibilityReason,
    get_compatibility_info_from_model,
    get_compatibility_info_from_model_bytes,
    get_ep_devices,
    get_hardware_device_ep_incompatibility_details,
    get_hardware_devices,
    get_model_compatibility_for_ep_devices,
)

# handle change from python 3.8 and on where loading a dll from the current directory needs to be explicitly allowed.
if platform.system() == "Windows" and sys.version_info.major >= 3 and sys.version_info.minor >= 8:  # noqa: YTT204
    os.add_dll_directory(os.getcwd())


def _create_model_with_compatibility_metadata(ep_compatibility_info=None):
    """Create a minimal valid ONNX model with optional compatibility metadata."""
    graph = onnx.helper.make_graph([], "test_graph", [], [])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])

    if ep_compatibility_info:
        for ep_type, compat_info in ep_compatibility_info.items():
            entry = onnx.StringStringEntryProto()
            entry.key = f"ep_compatibility_info.{ep_type}"
            entry.value = compat_info
            model.metadata_props.append(entry)

    return model.SerializeToString()


class TestEpCompatibility(unittest.TestCase):
    def test_invalid_args(self):
        # empty devices
        with self.assertRaises(RuntimeError):
            get_model_compatibility_for_ep_devices([], "info")
        # None compatibility info should raise TypeError before native call
        with self.assertRaises(TypeError):
            get_model_compatibility_for_ep_devices(get_ep_devices(), None)  # type: ignore[arg-type]

    def test_basic_smoke(self):
        devices = list(get_ep_devices())
        if not devices:
            self.skipTest("No EP devices available in this build")

        # Always select CPUExecutionProvider; skip if not present.
        cpu_devices = [d for d in devices if getattr(d, "ep_name", None) == "CPUExecutionProvider"]
        if not cpu_devices:
            self.skipTest("CPUExecutionProvider not available in this build")
        selected = [cpu_devices[0]]

        # API requires all devices belong to the same EP; we pass only one.
        status = get_model_compatibility_for_ep_devices(selected, "arbitrary-compat-string")
        self.assertEqual(status, OrtCompiledModelCompatibility.EP_NOT_APPLICABLE)

    def test_get_compatibility_info_from_model_bytes_with_metadata(self):
        ep_type = "TestCompatEP"
        expected_compat_info = "test_compat_v1.0_driver_123"
        model_data = _create_model_with_compatibility_metadata({ep_type: expected_compat_info})

        result = get_compatibility_info_from_model_bytes(model_data, ep_type)
        self.assertIsNotNone(result)
        self.assertEqual(result, expected_compat_info)

    def test_get_compatibility_info_from_model_bytes_not_found(self):
        model_data = _create_model_with_compatibility_metadata({"DifferentEP": "some_value"})

        result = get_compatibility_info_from_model_bytes(model_data, "NonExistentEP")
        self.assertIsNone(result)

    def test_get_compatibility_info_from_model_bytes_no_metadata(self):
        model_data = _create_model_with_compatibility_metadata()

        result = get_compatibility_info_from_model_bytes(model_data, "AnyEP")
        self.assertIsNone(result)

    def test_get_compatibility_info_from_model_bytes_invalid_data(self):
        with self.assertRaises(RuntimeError):
            get_compatibility_info_from_model_bytes(b"this is not a valid ONNX model", "TestEP")

    def test_get_compatibility_info_from_model_bytes_invalid_args(self):
        with self.assertRaises(RuntimeError):
            get_compatibility_info_from_model_bytes(b"", "TestEP")
        with self.assertRaises(RuntimeError):
            get_compatibility_info_from_model_bytes(b"data", "")

    def test_get_compatibility_info_from_model_file_with_metadata(self):
        ep_type = "TestCompatEP"
        expected_compat_info = "file_compat_v2.0"
        model_data = _create_model_with_compatibility_metadata({ep_type: expected_compat_info})

        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
            f.write(model_data)
            model_path = f.name

        try:
            result = get_compatibility_info_from_model(model_path, ep_type)
            self.assertIsNotNone(result)
            self.assertEqual(result, expected_compat_info)
        finally:
            os.unlink(model_path)

    def test_get_compatibility_info_from_model_file_not_found(self):
        with self.assertRaises(RuntimeError):
            get_compatibility_info_from_model("nonexistent_model_path.onnx", "TestEP")

    def test_get_compatibility_info_from_model_invalid_args(self):
        with self.assertRaises(RuntimeError):
            get_compatibility_info_from_model("", "TestEP")
        with self.assertRaises(RuntimeError):
            get_compatibility_info_from_model("model.onnx", "")


class TestHardwareDeviceCompatibility(unittest.TestCase):
    def test_get_hardware_devices_returns_devices(self):
        """Test that get_hardware_devices returns at least one device (CPU)."""
        devices = get_hardware_devices()
        self.assertIsNotNone(devices)
        self.assertTrue(len(devices) > 0, "Expected at least one hardware device")

        # Each device should be a valid OrtHardwareDevice
        for device in devices:
            self.assertIsNotNone(device)
            # Device should have type property which is an OrtHardwareDeviceType enum
            # CPU=0, GPU=1, NPU=2
            device_type = device.type
            self.assertIn(device_type.value, [0, 1, 2], f"Unexpected device type: {device_type}")
            # Device should have vendor property
            vendor = device.vendor
            self.assertIsNotNone(vendor)

    def test_get_hardware_device_ep_incompatibility_details_cpu_ep(self):
        """Test getting incompatibility details for CPU EP with CPU device."""
        devices = get_hardware_devices()
        self.assertTrue(len(devices) > 0, "Expected at least one hardware device")

        # Find CPU device (type.value == 0)
        cpu_devices = [d for d in devices if d.type.value == 0]
        if not cpu_devices:
            self.skipTest("No CPU device available")

        cpu_device = cpu_devices[0]
        details = get_hardware_device_ep_incompatibility_details("CPUExecutionProvider", cpu_device)

        # Should return a dict with expected keys
        self.assertIsInstance(details, dict)
        self.assertIn("reasons_bitmask", details)
        self.assertIn("notes", details)
        self.assertIn("error_code", details)

        # CPU EP should be compatible with CPU device (no incompatibility reasons)
        self.assertEqual(details["reasons_bitmask"], 0)  # 0 = no incompatibility
        self.assertEqual(details["error_code"], 0)

    def test_get_hardware_device_ep_incompatibility_details_invalid_ep(self):
        """Test that empty EP name raises error."""
        devices = get_hardware_devices()
        self.assertTrue(len(devices) > 0, "Expected at least one hardware device")

        first_device = devices[0]
        # Empty EP name should raise error
        with self.assertRaises(RuntimeError):
            get_hardware_device_ep_incompatibility_details("", first_device)

    def test_ortdevice_ep_incompatibility_reason_enum(self):
        """Test that the OrtDeviceEpIncompatibilityReason enum has expected values."""
        self.assertEqual(OrtDeviceEpIncompatibilityReason.NONE.value, 0)
        self.assertEqual(OrtDeviceEpIncompatibilityReason.DRIVER_INCOMPATIBLE.value, 1)
        self.assertEqual(OrtDeviceEpIncompatibilityReason.DEVICE_INCOMPATIBLE.value, 2)
        self.assertEqual(OrtDeviceEpIncompatibilityReason.MISSING_DEPENDENCY.value, 4)
        # UNKNOWN is the high-bit flag (0x80000000) and may be exposed as signed or unsigned.
        self.assertEqual(OrtDeviceEpIncompatibilityReason.UNKNOWN.value & 0xFFFFFFFF, 0x80000000)


if __name__ == "__main__":
    unittest.main()
