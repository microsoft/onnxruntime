import numpy as np
import onnx

from .quantize import quantize, QuantizationMode


def calibrate_custom(model):
    quant_dict = {}
    for node in model.graph.node:
        if node.op_type == "MatMul":
            quant_dict[node.input[0]] = [np.uint8(128), np.float32(0.01)]
            quant_dict[node.output[0]] = [np.uint8(128), np.float32(0.01)]
        if node.op_type == "Add":
            #quant_dict[node.input[0]] = [np.uint8(128), np.float32(0.01)]
            #quant_dict[node.input[1]] = [np.uint8(128), np.float32(0.01)]
            quant_dict[node.output[0]] = [np.uint8(128), np.float32(0.01)]
        if node.op_type == "Mul":
            #quant_dict[node.input[0]] = [np.uint8(128), np.float32(0.01)]
            #quant_dict[node.input[1]] = [np.uint8(128), np.float32(0.01)]
            quant_dict[node.output[0]] = [np.uint8(128), np.float32(0.01)]
    return quant_dict


def main():
    filename = r"D:\projects\quantization\from_intel\bertsquad\bertsquad10.onnx"
    filename_q = r"D:\projects\quantization\from_intel\bertsquad\bertsquad10.quant.mul.onnx"

    model = onnx.load(filename)

    param_q = calibrate_custom(model)

    quantized_model = quantize(model,
                               quantization_mode=QuantizationMode.QLinearOps,
                               static=False,
                               quantization_params=param_q,
                               nodes_to_exclude=[
                                   'bert/encoder/layer_9/attention/self/mul_2',
                                   'bert/encoder/layer_9/attention/self/mul_1',
                                   'bert/encoder/layer_8/attention/self/mul_2',
                                   'bert/encoder/layer_7/attention/self/mul_2'
                               ])

    onnx.save(quantized_model, filename_q)


if __name__ == "__main__":
    main()
