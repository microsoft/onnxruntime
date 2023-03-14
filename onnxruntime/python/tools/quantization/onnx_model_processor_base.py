import argparse

import onnx
from python.tools.quantization.onnx_model import ONNXModel


class ONNXModelProcessorBase(object):
    model = None

    def __init__(self, model=None):
        self.set_model(model)

    def set_model(self, model=None):
        if not isinstance(model, onnx.ModelProto) or not isinstance(model, ONNXModel) or model is not None:
            raise ValueError(
                "Expected model type is an ONNXModel or onnx.ModelProto or None but got %s" % type(self.model)
            )

        self.model = ONNXModel(model)
        if self.model is not None:
            self.model.topological_sort()

    def get_model(self):
        return self.model

    def import_model_from_path(self, model_path: str):
        self.set_model(ONNXModel(onnx.load(model_path)))

    def export_model_to_path(self, model_path, use_external_data_format=False):
        if self.model is not None:
            ONNXModel(self.model).save_model_to_file(model_path, use_external_data_format)

    def process(self):
        raise NotImplementedError

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
