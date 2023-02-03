def get_quant_params(model):
    """Returns quantization parameters for the given model.

    Quantization parameters in this function refers to all scale and zero point
    inputs to any QuantizeLinear, DequantizeLinear or FakeQuant node in the model."""

    return {
        quant_param_name
        for node in model.graph.node
        for quant_param_name in node.input[1:]
        if node.op_type == "QuantizeLinear" or node.op_type == "DequantizeLinear" or node.op_type == "FakeQuant"
    }
