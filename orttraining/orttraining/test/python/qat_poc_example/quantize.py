import logging

import numpy as np
import torch
from torchvision import datasets, transforms

import onnxruntime.quantization as quantization


def preprocess(input_model_dir, output_model_dir):
    """Preprocesses the given onnx model for quantization. Unused for QAT process."""
    quantization.shape_inference.quant_pre_process(input_model_dir, output_model_dir)


class MnistDataReader:
    """Generates calibration data from the MNIST dataset."""

    def __init__(self, data_dir, num_samples):
        self.data_dir = data_dir
        self.num_samples = num_samples

        mnist_data = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
        )
        probs = torch.tensor([1 / mnist_data.data.shape[0]] * num_samples, dtype=torch.float32)
        index = probs.multinomial(num_samples=num_samples, replacement=False)
        self.data = torch.reshape(mnist_data.data[index], [num_samples, -1])
        self.input_name = "input-0"
        self.datasize = num_samples
        self.data_loader = iter(self.data)

    def get_next(self):
        next_array = next(self.data_loader, None)
        if next_array is None:
            return None
        return {self.input_name: np.expand_dims(next_array, axis=0).tolist()}


def quantize_static(input_model_dir, output_model_dir):
    """Statically quantizes the given onnx model using onnxruntime's static quantization tools.

    Calibration data is generated from the MNIST dataset.
    """

    extra_options = {"AddQDQPairToWeight": True, "QuantizeBias": False}
    logging.info("Using calibration data from MNIST dataset picking 1000 samples at random.")
    calibration_data_reader = MnistDataReader("data", 1000)

    # Quantize the model
    logging.info(
        "Invoking onnxruntime.quantization.quantize_static with AddQDQPairToWeight=True and QuantizeBias=False.."
    )
    logging.info("Quantized model will be saved to %s." % output_model_dir)
    quantization.quantize_static(
        input_model_dir,
        output_model_dir,
        calibration_data_reader,
        extra_options=extra_options,
    )
