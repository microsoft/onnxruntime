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


class TestSymLinkOnnxModelExternalData(unittest.TestCase):
    def test_symlink_model_and_data_under_same_directory(self):
        # The following directory structure simulates huggingface hub local cache:
        # temp_dir/         (This corresponds to .cache/huggingface/hub/model_id/)
        #   blobs/
        #     guid1
        #     guid2
        #   snapshots/version/
        #       model.onnx -> ../../blobs/guid1
        #       data.bin   -> ../../blobs/guid2

        self.temp_dir = tempfile.mkdtemp()
        try:
            blobs_dir = os.path.join(self.temp_dir, "blobs")
            os.makedirs(blobs_dir)

            snapshots_dir = os.path.join(self.temp_dir, "snapshots", "version")
            os.makedirs(snapshots_dir)

            # Create real files in blobs
            # We'll use the helper to create the model, but we need to control where files end up.
            # Let's manually create the data file in blobs
            data_blob_path = os.path.join(blobs_dir, "guid2")
            vals = [float(i) for i in range(10)]
            with open(data_blob_path, "wb") as f:
                f.writelines(struct.pack("f", v) for v in vals)

            # Create model in blobs (referencing "data.bin" as external data)
            # When loaded from snapshots/version/model.onnx, ORT looks for snapshots/version/data.bin

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
            # Model is opset-agnostic (Constant + Add); pin a released opset so ONNX 1.22
            # does not default to the under-development opset 27, which fails strict
            # ALLOW_RELEASED_ONNX_OPSET_ONLY legs (ValidateOpsetForDomain throws).
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])

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
        #   snapshots/version/
        #       model.onnx -> ../../blobs/guid1
        #       data.bin   -> ../../blobs/data/guid2

        self.temp_dir = tempfile.mkdtemp()
        try:
            blobs_dir = os.path.join(self.temp_dir, "blobs")
            os.makedirs(blobs_dir)
            data_dir = os.path.join(blobs_dir, "data")
            os.makedirs(data_dir)

            snapshots_dir = os.path.join(self.temp_dir, "snapshots", "version")
            os.makedirs(snapshots_dir)

            # Create real files in blobs
            # We'll use the helper to create the model, but we need to control where files end up.
            # Let's manually create the data file in blobs
            data_blob_path = os.path.join(data_dir, "guid2")
            vals = [float(i) for i in range(10)]
            with open(data_blob_path, "wb") as f:
                f.writelines(struct.pack("f", v) for v in vals)

            # Create model in blobs (referencing "data.bin" as external data)
            # When loaded from snapshots/version/model.onnx, ORT looks for snapshots/version/data.bin

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
            # Model is opset-agnostic (Constant + Add); pin a released opset so ONNX 1.22
            # does not default to the under-development opset 27, which fails strict
            # ALLOW_RELEASED_ONNX_OPSET_ONLY legs (ValidateOpsetForDomain throws).
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])

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
        #   snapshots/version/
        #       model.onnx -> ../../model/guid1
        #       data.bin   -> ../../data/guid2

        self.temp_dir = tempfile.mkdtemp()
        try:
            model_dir = os.path.join(self.temp_dir, "model")
            os.makedirs(model_dir)
            data_dir = os.path.join(self.temp_dir, "data")
            os.makedirs(data_dir)

            snapshots_dir = os.path.join(self.temp_dir, "snapshots", "version")
            os.makedirs(snapshots_dir)

            # Create real files in data_dir
            # We'll use the helper to create the model, but we need to control where files end up.
            # Let's manually create the data file in data_dir
            data_blob_path = os.path.join(data_dir, "guid2")
            vals = [float(i) for i in range(10)]
            with open(data_blob_path, "wb") as f:
                f.writelines(struct.pack("f", v) for v in vals)

            # Create model in model_dir (referencing "data.bin" as external data)
            # When loaded from snapshots/version/model.onnx, ORT looks for snapshots/version/data.bin

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
            # Model is opset-agnostic (Constant + Add); pin a released opset so ONNX 1.22
            # does not default to the under-development opset 27, which fails strict
            # ALLOW_RELEASED_ONNX_OPSET_ONLY legs (ValidateOpsetForDomain throws).
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])

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

            # We expect an error about external data not under model directory or the real model directory.
            self.assertIn("External data path validation failed", str(cm.exception))
        finally:
            shutil.rmtree(self.temp_dir)

    def test_symlink_hf_shared_blob_store(self):
        # Since version 1.32, huggingface_hub stores Xet files in a blob store shared across the repos of a cache:
        # hub_dir/                            (This corresponds to .cache/huggingface/hub/)
        #   blobs/
        #     28/28aa...                      (model)
        #     d4/d4bb...                      (external data)
        #   models--org--name/
        #     blobs/
        #       model_etag -> ../../blobs/28/28aa...
        #       data_etag  -> ../../blobs/d4/d4bb...
        #     snapshots/version/onnx/
        #       model.onnx -> ../../../blobs/model_etag
        #       data.bin   -> ../../../blobs/data_etag
        # The model and its external data resolve into different folders. A file can also be a regular file in the
        # repo's blobs/ folder, e.g. if it was downloaded by an older version of huggingface_hub.
        # Exceed kSmallTensorExternalDataThreshold (127 bytes) on a graph initializer so InferenceSession
        # exercises both ValidateExternalDataPath (Graph::Resolve) and ValidateOpenedExternalDataPath
        # (GetExtDataFromTensorProto).
        vals = [float(i) for i in range(64)]
        data_bytes = b"".join(struct.pack("<f", v) for v in vals)
        model_proto = _create_model_with_external_data("data.bin", vals)
        for model_in_shared_store, data_in_shared_store in ((True, True), (False, True), (True, False)):
            with (
                self.subTest(model_in_shared_store=model_in_shared_store, data_in_shared_store=data_in_shared_store),
                tempfile.TemporaryDirectory() as hub_dir,
            ):
                repo_dir = os.path.join(hub_dir, "models--org--name")
                snapshot_dir = os.path.join(repo_dir, "snapshots", "version", "onnx")
                os.makedirs(snapshot_dir)
                os.makedirs(os.path.join(repo_dir, "blobs"))

                try:
                    model_xet_hash = "28" + "a" * 62 if model_in_shared_store else None
                    data_xet_hash = "d4" + "b" * 62 if data_in_shared_store else None
                    _add_hf_cache_file(
                        repo_dir,
                        snapshot_dir,
                        "model.onnx",
                        "model_etag",
                        model_xet_hash,
                        lambda path: save(model_proto, path),
                    )
                    _add_hf_cache_file(
                        repo_dir,
                        snapshot_dir,
                        "data.bin",
                        "data_etag",
                        data_xet_hash,
                        lambda path: _write_bytes(path, data_bytes),
                    )
                except (OSError, NotImplementedError) as e:
                    self.skipTest(f"Skipping symlink test: symlink creation is not supported in this environment: {e}")

                sess = ort.InferenceSession(
                    os.path.join(snapshot_dir, "model.onnx"), providers=["CPUExecutionProvider"]
                )

                res = sess.run(["output"], {"input": np.zeros(len(vals), dtype=np.float32)})
                np.testing.assert_allclose(res[0], np.array(vals, dtype=np.float32))


def _write_bytes(path, content):
    with open(path, "wb") as f:
        f.write(content)


def _add_hf_cache_file(repo_dir, snapshot_dir, name, etag, xet_hash, write_fn):
    """Adds a file to a snapshot of a Hugging Face Hub cache repo like huggingface_hub does.

    Writes the target blob first, then creates the snapshot symlink to the repo blob named `etag` (which is itself
    a symlink into the cache's shared blob store when `xet_hash` is given).
    """
    repo_blob = os.path.join(repo_dir, "blobs", etag)
    blob = repo_blob
    if xet_hash:
        blob = os.path.join(os.path.dirname(repo_dir), "blobs", xet_hash[:2], xet_hash)
        os.makedirs(os.path.dirname(blob), exist_ok=True)
        write_fn(blob)
        os.symlink(os.path.relpath(blob, os.path.dirname(repo_blob)), repo_blob)
    else:
        write_fn(blob)
    os.symlink(os.path.relpath(repo_blob, snapshot_dir), os.path.join(snapshot_dir, name))
    return blob


def _create_model_with_external_data(location, vals):
    input_ = helper.make_tensor_value_info("input", TensorProto.FLOAT, [len(vals)])
    output = helper.make_tensor_value_info("output", TensorProto.FLOAT, [len(vals)])
    tensor = helper.make_tensor("external_data", TensorProto.FLOAT, [len(vals)], vals)
    tensor.data_location = TensorProto.EXTERNAL
    tensor.ClearField("float_data")
    tensor.ClearField("raw_data")
    for key, value in (("location", location), ("offset", "0"), ("length", str(len(vals) * 4))):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value

    add_node = helper.make_node("Add", ["input", "external_data"], ["output"])
    graph = helper.make_graph([add_node], "test_graph", [input_], [output], initializer=[tensor])
    # Pin a released opset so that strict ALLOW_RELEASED_ONNX_OPSET_ONLY legs do not fail.
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 22)])


if __name__ == "__main__":
    unittest.main()
