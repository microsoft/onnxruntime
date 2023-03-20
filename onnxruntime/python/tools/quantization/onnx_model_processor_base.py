import argparse

import onnx
from onnx import ModelProto

from onnxruntime.quantization.onnx_model import ONNXModel


class ONNXModelProcessorBase(object):
    model: ModelProto = None

    def __init__(self, model=None):
        self.set_model(model)

    def set_model(self, model=None):
        self.model = model

    def get_model(self):
        return self.model

    def import_model_from_path(self, model_path: str):
        self.set_model(ONNXModel(onnx.load(model_path)))

    def export_model_to_path(self, model_path, use_external_data_format=False):
        if self.model is not None:
            ONNXModel(self.model).save_model_to_file(model_path, use_external_data_format)

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", required=True, type=str, help="input onnx model path")
        parser.add_argument("--output", required=True, type=str, help="optimized onnx model path")
        parser.add_argument(
            "--use_external_data_format",
            required=False,
            action="store_true",
            default=False,
            help="use external data format to store large model (>2GB)",
        )

        return parser
