import argparse
from pathlib import Path

import onnx
from onnx import ModelProto

import onnxruntime.quantization.quant_utils as quant_utils
from onnxruntime.quantization.onnx_model import ONNXModel


class ONNXModelProcessorBase(object):
    model: ModelProto = None
    model_path: str = None
    model_optimized: bool = False
    model_sorted: bool = False

    def __init__(self, model=None, sort_model=False):
        self.set_model(model, sort_model)

    def set_model(self, model: ModelProto = None, sort_model=False):
        if sort_model and model is not None:
            onnx_model = ONNXModel(model)
            onnx_model.topological_sort()
            self.model = onnx_model.model
        elif model is not None:
            self.model = model

    def get_model(self):
        return self.model

    def import_model_from_path(self, model_path: str, optimize_model=False, sort_model=False):
        self.model_path = model_path
        if sort_model:
            onnx_model = ONNXModel(onnx.load(model_path))
            onnx_model.topological_sort()
            self.model = onnx_model.model
        else:
            self.model = onnx.load(model_path)
        if optimize_model:
            print("Optimizing model...")
            self.optimize_model()

    def export_model_to_path(self, model_path, use_external_data_format=False):
        if self.model is not None:
            ONNXModel(self.model).save_model_to_file(model_path, use_external_data_format)

    def optimize_model(self):
        """
            Generate model that applies graph optimization (constant folding, etc.)
            parameter model_path: path to the original onnx model
            parameter opt_model_path: path to the optimized onnx model
        :return: optimized onnx model
        """
        if self.model_path is None:
            self.model_path = "Custom.onnx"
            self.export_model_to_path(self.model_path)
        opt_model_path = self.model_path[:-5] + "_optimized.onnx"
        quant_utils.optimize_model(Path(self.model_path), Path(opt_model_path))
        self.import_model_from_path(opt_model_path, optimize_model=False)

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
