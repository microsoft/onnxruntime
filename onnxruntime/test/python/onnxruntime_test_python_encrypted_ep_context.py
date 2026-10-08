# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
from __future__ import annotations

import gc
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from autoep_helper import AutoEpTestCase
from helper import get_name, get_shared_library_filename_for_platform

import onnxruntime as ort
from onnxruntime.capi.onnxruntime_pybind11_state import Fail, InvalidArgument


class TestEncryptedEpContext(AutoEpTestCase):
    @unittest.skipUnless(
        importlib.util.find_spec("cryptography"),
        "Install onnxruntime/test/python/requirements.txt to run authenticated encryption tests",
    )
    def test_persisted_encrypted_compiled_model_and_context_run_inference(self):
        from cryptography.exceptions import InvalidTag  # noqa: PLC0415
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # noqa: PLC0415

        try:
            library = get_name(get_shared_library_filename_for_platform("example_plugin_ep"))
        except FileNotFoundError:
            self.skipTest("The example plugin EP integration artifact is not available")
        ep_name = "python_encrypted_ep_context"
        self.register_execution_provider_library(ep_name, os.path.abspath(library))
        try:
            device = next(device for device in ort.get_ep_devices() if device.ep_name == ep_name)
            with tempfile.TemporaryDirectory(prefix="ort.encrypted_") as directory:
                model_file = Path(directory, "model.aesgcm")
                context_file = Path(directory, "context.aesgcm")
                key = AESGCM.generate_key(bit_length=256)
                context_name = ""
                writes = 0
                reads = 0

                def encrypt(plaintext: bytes, name: str) -> bytes:
                    nonce = os.urandom(12)
                    return nonce + AESGCM(key).encrypt(nonce, plaintext, name.encode("utf-8"))

                def decrypt(record: bytes, read_key: bytes, name: str) -> bytes:
                    return AESGCM(read_key).decrypt(record[:12], record[12:], name.encode("utf-8"))

                def options() -> ort.SessionOptions:
                    result = ort.SessionOptions()
                    result.add_session_config_entry("ep.example.test_execute_ep_context", "1")
                    result.add_provider_for_devices([device], {})
                    return result

                x = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [3, 2])
                y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [3, 2])
                z = onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [3, 2])
                graph = onnx.helper.make_graph([onnx.helper.make_node("Mul", ["X", "Y"], ["Z"])], "mul", [x, y], [z])
                original = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 13)])
                original.ir_version = 8
                input_model = original.SerializeToString()
                compile_options = options()
                compiler = ort.ModelCompiler(compile_options, input_model, embed_compiled_data_into_model=False)

                def write_context(name: str, data: ort.OrtEpContextData):
                    nonlocal context_name, writes
                    self.assertEqual(writes, 0)
                    writes += 1
                    context_name = name
                    context_file.write_bytes(encrypt(data.read(), name))

                compiler.set_ep_context_data_write_func(write_context)
                compiled = compiler.compile_to_bytes()
                compiled_graph = onnx.load_model_from_string(compiled).graph
                self.assertEqual(len(compiled_graph.node), 1)
                self.assertEqual(compiled_graph.node[0].op_type, "EPContext")
                self.assertEqual(len(compiled_graph.initializer), 0)
                model_file.write_bytes(encrypt(compiled, "compiled.onnx"))
                del compiler, compile_options, compiled, compiled_graph, original, input_model
                gc.collect()
                self.assertEqual(writes, 1)
                self.assertTrue(context_name)
                self.assertFalse(Path(context_name).exists())
                self.assertEqual({path.name for path in Path(directory).iterdir()}, {"model.aesgcm", "context.aesgcm"})

                encrypted_model = model_file.read_bytes()
                encrypted_context = context_file.read_bytes()
                restored = decrypt(encrypted_model, key, "compiled.onnx")
                load_options = options()

                def read_context(name: str, output: ort.OrtEpContextDataBuffer):
                    nonlocal reads
                    self.assertEqual(name, context_name)
                    reads += 1
                    plaintext = decrypt(context_file.read_bytes(), key, name)
                    output.allocate(len(plaintext))
                    output.write(plaintext)

                load_options.set_ep_context_data_read_func(read_context, len(encrypted_context) - 28)
                session = ort.InferenceSession(restored, sess_options=load_options)
                for iteration in range(2):
                    left = np.arange(6, dtype=np.float32).reshape(3, 2) - iteration
                    right = 2 * np.arange(6, dtype=np.float32).reshape(3, 2) + iteration
                    np.testing.assert_array_equal(session.run(None, {"X": left, "Y": right})[0], left * right)
                self.assertEqual(reads, 1)
                del session, load_options
                gc.collect()

                def assert_context_rejected(record: bytes, read_key: bytes):
                    failure_options = options()

                    def read_corrupt(name: str, _output: ort.OrtEpContextDataBuffer):
                        try:
                            decrypt(record, read_key, name)
                        except InvalidTag as failure:
                            raise ValueError("EPContext authentication failed") from failure
                        raise AssertionError("Corrupt ciphertext was accepted")

                    failure_options.set_ep_context_data_read_func(read_corrupt, len(record) - 28)
                    with self.assertRaisesRegex(Fail, "EPContext authentication failed"):
                        ort.InferenceSession(restored, sess_options=failure_options)

                wrong_key = bytes([key[0] ^ 1]) + key[1:]
                with self.assertRaises(InvalidTag):
                    decrypt(encrypted_model, wrong_key, "compiled.onnx")
                assert_context_rejected(encrypted_context, wrong_key)
                tampered_model = encrypted_model[:-1] + bytes([encrypted_model[-1] ^ 1])
                with self.assertRaises(InvalidTag):
                    decrypt(tampered_model, key, "compiled.onnx")
                tampered_context = encrypted_context[:-1] + bytes([encrypted_context[-1] ^ 1])
                assert_context_rejected(tampered_context, key)
                with self.assertRaises(InvalidTag):
                    decrypt(encrypted_context, key, "different-context-name")
                with self.assertRaisesRegex(InvalidArgument, "requires a model path"):
                    ort.InferenceSession(restored, sess_options=options())
                self.assertFalse(Path(context_name).exists())
                self.assertEqual({path.name for path in Path(directory).iterdir()}, {"model.aesgcm", "context.aesgcm"})
        finally:
            self.unregister_execution_provider_library(ep_name)


if __name__ == "__main__":
    unittest.main()
