# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pathlib
import tempfile
import unittest

import numpy as np
import onnx_test_data_utils

# example usage from <ort root>/tools/python
# python -m unittest test_onnx_test_data_utils.py

try:
    import PIL.Image
except ImportError:
    PIL = None

COLOR = (200, 100, 50)


@unittest.skipIf(PIL is None, "Pillow is required to test image conversion.")
class TestImageToNumpy(unittest.TestCase):
    def setUp(self):
        self._temp_dir = tempfile.TemporaryDirectory(prefix="ort.onnx_test_data_utils_")
        self.addCleanup(self._temp_dir.cleanup)

    def _save_solid_image(self, width, height):
        path = str(pathlib.Path(self._temp_dir.name) / f"image_{width}x{height}.png")
        PIL.Image.fromarray(np.full((height, width, 3), COLOR, dtype=np.uint8)).save(path)
        return path

    def _check_solid_resize(self, width, height, target_hw):
        path = self._save_solid_image(width, height)
        data = onnx_test_data_utils.image_to_numpy(path, target_hw, channels_last=True, add_batch_dim=False)
        self.assertEqual(data.shape, (*target_hw, 3))
        self.assertEqual(data.dtype, np.float32)
        # a solid image stays solid after resize and center crop. Any crop that reaches outside the resized image
        # would add black (0) rows or columns.
        np.testing.assert_array_equal(data, np.broadcast_to(np.array(COLOR, dtype=np.float32), data.shape))

    def test_no_resize(self):
        path = self._save_solid_image(40, 30)
        data = onnx_test_data_utils.image_to_numpy(path, None, channels_last=False, add_batch_dim=True)
        self.assertEqual(data.shape, (1, 3, 30, 40))
        for channel, value in enumerate(COLOR):
            np.testing.assert_array_equal(data[0, channel], np.full((30, 40), value, dtype=np.float32))

    def test_resize_to_height_width(self):
        self._check_solid_resize(40, 30, (20, 25))

    def test_resize_with_odd_margin(self):
        # the scaled image is 27 wide and is cropped to 24 wide, so the margin is odd
        self._check_solid_resize(40, 30, (20, 24))
        self._check_solid_resize(41, 30, (21, 20))

    def test_resize_where_scaled_size_is_not_integral(self):
        # w * ratio truncates to 223 for the 277 pixel side if int() were used, which left a black row after cropping
        self._check_solid_resize(554, 277, (224, 224))
        self._check_solid_resize(1920, 1080, (224, 224))

    def test_resize_channels_last_without_batch_dim(self):
        path = self._save_solid_image(40, 30)
        data = onnx_test_data_utils.image_to_numpy(path, [20, 25], channels_last=True, add_batch_dim=False)
        self.assertEqual(data.shape, (20, 25, 3))
        np.testing.assert_array_equal(data[0, 0], np.array(COLOR, dtype=np.float32))

    def test_center_crop_selects_the_middle(self):
        # left third black, middle third white, right third black. A 30x30 center crop of the 90x30 image keeps
        # only the middle third.
        pixels = np.zeros((30, 90, 3), dtype=np.uint8)
        pixels[:, 30:60] = 255
        path = str(pathlib.Path(self._temp_dir.name) / "stripes.png")
        PIL.Image.fromarray(pixels).save(path)
        data = onnx_test_data_utils.image_to_numpy(path, [30, 30], channels_last=True, add_batch_dim=False)
        self.assertEqual(data.shape, (30, 30, 3))
        np.testing.assert_array_equal(data, np.full((30, 30, 3), 255, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
