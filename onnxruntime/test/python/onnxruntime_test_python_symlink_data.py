# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import struct
import tempfile
import unittest

import numpy as np
from onnx import TensorProto, helper, save

import onnxruntime as ort


class TestWhitelistedData(unittest.TestCase):
    def test_huggingface_hub_symlink(self):
        # working directory structure (simulate huggingface hub local cache):
        # temp_dir/
        #   blobs/
        #     guid1
        #     guid2
        #   snapshots/version/onnx/
        #       model.onnx -> ../../blobs/guid1
        #       data.bin   -> ../../blobs/guid2

        self.temp_dir = tempfile.mkdtemp()
        try:
            blobs_dir = os.path.join(self.temp_dir, "blobs")
            os.makedirs(blobs_dir)

            snapshots_dir = os.path.join(self.temp_dir, "snapshots", "version", "onnx")
            os.makedirs(snapshots_dir)

            # Create real files in blobs
            # We'll use the helper to create the model, but we need to control where files end up.
            # Let's manually create the data file in blobs
            data_blob_path = os.path.join(blobs_dir, "guid2")
            vals = [float(i) for i in range(10)]
            with open(data_blob_path, "wb") as f:
                f.writelines(struct.pack("f", v) for v in vals)

            # Create model in blobs (referencing "data.bin" as external data)
            # Note: The model proto will just say "data.bin".
            # When loaded from snapshots/onnx/model.onnx, ORT looks for snapshots/onnx/data.bin

            input_ = helper.make_tensor_value_info("input", TensorProto.FLOAT, [10])
            output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [10])
            tensor = helper.make_tensor("external_data", TensorProto.FLOAT, [10], vals)
            tensor.data_location = TensorProto.EXTERNAL
            tensor.ClearField("float_data")
            tensor.ClearField("raw_data")

            k = tensor.external_data.add()
            k.key = "location"
            k.value = "data.bin"  # Relative path

            offset = tensor.external_data.add()
            offset.key = "offset"
            offset.value = "0"

            length = tensor.external_data.add()
            length.key = "length"
            length.value = str(len(vals) * 4)

            const_node = helper.make_node("Constant", [], ["const_out"], value=tensor)
            add_node = helper.make_node("Add", ["input", "const_out"], ["output"])
            graph = helper.make_graph([const_node, add_node], "test_graph", [input_], [output])
            model = helper.make_model(graph)

            model_blob_path = os.path.join(blobs_dir, "guid1")
            save(model, model_blob_path)

            # Now create symlinks in snapshots
            model_symlink_path = os.path.join(snapshots_dir, "model.onnx")
            data_symlink_path = os.path.join(snapshots_dir, "data.bin")

            try:
                os.symlink(model_blob_path, model_symlink_path)
                os.symlink(data_blob_path, data_symlink_path)
            except (OSError, NotImplementedError) as e:
                self.skipTest(f"Skipping symlink test: symlink creation is not supported in this environment: {e}")

            sess = ort.InferenceSession(model_symlink_path, providers=["CPUExecutionProvider"])

            input_data = np.zeros(10, dtype=np.float32)
            res = sess.run(["output"], {"input": input_data})
            expected = np.array([float(i) for i in range(10)], dtype=np.float32)
            np.testing.assert_allclose(res[0], expected)

        finally:
            shutil.rmtree(self.temp_dir)

    def test_symlink_with_data_in_model_sub_dir(self):
        # working directory structure (data is in model sub directory):
        # temp_dir/
        #   blobs/
        #     guid1
        #     data/guid2
        #   snapshots/version/onnx/
        #       model.onnx -> ../../blobs/guid1
        #       data.bin   -> ../../blobs/data/guid2

        self.temp_dir = tempfile.mkdtemp()
        try:
            blobs_dir = os.path.join(self.temp_dir, "blobs")
            os.makedirs(blobs_dir)
            data_dir = os.path.join(blobs_dir, "data")
            os.makedirs(data_dir)

            snapshots_dir = os.path.join(self.temp_dir, "snapshots", "version", "onnx")
            os.makedirs(snapshots_dir)

            # Create real files in blobs
            # We'll use the helper to create the model, but we need to control where files end up.
            # Let's manually create the data file in blobs
            data_blob_path = os.path.join(data_dir, "guid2")
            vals = [float(i) for i in range(10)]
            with open(data_blob_path, "wb") as f:
                f.writelines(struct.pack("f", v) for v in vals)

            # Create model in blobs (referencing "data.bin" as external data)
            # Note: The model proto will just say "data.bin".
            # When loaded from snapshots/onnx/model.onnx, ORT looks for snapshots/onnx/data.bin

            input_ = helper.make_tensor_value_info("input", TensorProto.FLOAT, [10])
            output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [10])
            tensor = helper.make_tensor("external_data", TensorProto.FLOAT, [10], vals)
            tensor.data_location = TensorProto.EXTERNAL
            tensor.ClearField("float_data")
            tensor.ClearField("raw_data")

            k = tensor.external_data.add()
            k.key = "location"
            k.value = "data.bin"  # Relative path

            offset = tensor.external_data.add()
            offset.key = "offset"
            offset.value = "0"

            length = tensor.external_data.add()
            length.key = "length"
            length.value = str(len(vals) * 4)

            const_node = helper.make_node("Constant", [], ["const_out"], value=tensor)
            add_node = helper.make_node("Add", ["input", "const_out"], ["output"])
            graph = helper.make_graph([const_node, add_node], "test_graph", [input_], [output])
            model = helper.make_model(graph)

            model_blob_path = os.path.join(blobs_dir, "guid1")
            save(model, model_blob_path)

            # Now create symlinks in snapshots
            model_symlink_path = os.path.join(snapshots_dir, "model.onnx")
            data_symlink_path = os.path.join(snapshots_dir, "data.bin")

            try:
                os.symlink(model_blob_path, model_symlink_path)
                os.symlink(data_blob_path, data_symlink_path)
            except (OSError, NotImplementedError) as e:
                self.skipTest(f"Skipping symlink test: symlink creation is not supported in this environment: {e}")

            sess = ort.InferenceSession(model_symlink_path, providers=["CPUExecutionProvider"])

            input_data = np.zeros(10, dtype=np.float32)
            res = sess.run(["output"], {"input": input_data})
            expected = np.array([float(i) for i in range(10)], dtype=np.float32)
            np.testing.assert_allclose(res[0], expected)

        finally:
            shutil.rmtree(self.temp_dir)

    def test_symlink_with_data_not_in_model_sub_dir(self):
        # working directory structure (data is not in model directory or its sub directories):
        # temp_dir/
        #   model/
        #     guid1
        #   data/
        #     guid2
        #   snapshots/version/onnx/
        #       model.onnx -> ../../model/guid1
        #       data.bin   -> ../../data/guid2

        self.temp_dir = tempfile.mkdtemp()
        try:
            model_dir = os.path.join(self.temp_dir, "model")
            os.makedirs(model_dir)
            data_dir = os.path.join(self.temp_dir, "data")
            os.makedirs(data_dir)

            snapshots_dir = os.path.join(self.temp_dir, "snapshots", "version", "onnx")
            os.makedirs(snapshots_dir)

            # Create real files in blobs
            # We'll use the helper to create the model, but we need to control where files end up.
            # Let's manually create the data file in blobs
            data_blob_path = os.path.join(data_dir, "guid2")
            vals = [float(i) for i in range(10)]
            with open(data_blob_path, "wb") as f:
                f.writelines(struct.pack("f", v) for v in vals)

            # Create model in blobs (referencing "data.bin" as external data)
            # Note: The model proto will just say "data.bin".
            # When loaded from snapshots/onnx/model.onnx, ORT looks for snapshots/onnx/data.bin

            input_ = helper.make_tensor_value_info("input", TensorProto.FLOAT, [10])
            output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [10])
            tensor = helper.make_tensor("external_data", TensorProto.FLOAT, [10], vals)
            tensor.data_location = TensorProto.EXTERNAL
            tensor.ClearField("float_data")
            tensor.ClearField("raw_data")

            k = tensor.external_data.add()
            k.key = "location"
            k.value = "data.bin"  # Relative path

            offset = tensor.external_data.add()
            offset.key = "offset"
            offset.value = "0"

            length = tensor.external_data.add()
            length.key = "length"
            length.value = str(len(vals) * 4)

            const_node = helper.make_node("Constant", [], ["const_out"], value=tensor)
            add_node = helper.make_node("Add", ["input", "const_out"], ["output"])
            graph = helper.make_graph([const_node, add_node], "test_graph", [input_], [output])
            model = helper.make_model(graph)

            model_blob_path = os.path.join(model_dir, "guid1")
            save(model, model_blob_path)

            # Now create symlinks in snapshots
            model_symlink_path = os.path.join(snapshots_dir, "model.onnx")
            data_symlink_path = os.path.join(snapshots_dir, "data.bin")

            try:
                os.symlink(model_blob_path, model_symlink_path)
                os.symlink(data_blob_path, data_symlink_path)
            except (OSError, NotImplementedError) as e:
                self.skipTest(f"Skipping symlink test: symlink creation is not supported in this environment: {e}")

            with self.assertRaises(Exception) as cm:
                ort.InferenceSession(model_symlink_path, providers=["CPUExecutionProvider"])

            # We expect an error about external data not being in whitelisted directories
            self.assertIn("External data path validation failed", str(cm.exception))
        finally:
            shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main()
