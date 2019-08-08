import unittest
import onnx

from quantize import quantize, QuantizationMode
import calibrate

class ProcessLogFileTest(unittest.TestCase):
    # def test_baddir(self):
    #     self.assertRaises(ValueError, calibrate.process_logfiles('/non/existent/path'))

    def test_gooddata(self):
        expected = "gpu_0_conv1_1"
        sfacs, zpts = calibrate.process_logfiles('test_data')
        self.assertTrue(expected in sfacs)
        self.assertTrue(expected in zpts)

        self.assertEqual(8.999529411764707, sfacs[expected])
        self.assertEqual(23086, zpts[expected])

if __name__ == '__main__':
    unittest.main()

## Load the onnx model
# model = onnx.load('path/to/the/model.onnx')
