import onnx

from ..quant_utils import (
    TENSOR_NAME_QUANT_SUFFIX,
    QuantizedValue,
    QuantizedValueType,
    attribute_to_kwarg,
    compute_scale_zp,
    get_qmin_qmax_for_qType,
    ms_domain,
)
from .base_operator import QuantOperatorBase
from .qdq_base_operator import QDQOperatorBase


class QLinearSoftmax(QuantOperatorBase):
    def quantize(self):
        node = self.node
        # set limitations for softmax output scale and zp, because the output of softmax is always 0-1
        if self.quantizer.activation_qType == onnx.onnx_pb.TensorProto.UINT8:
            out_scale = 1 / 256.0
            out_zero_point = 0
        else:
            out_scale = 1 / 256.0
            out_zero_point = -128
        # only try to quantize when given quantization parameters for it
        (
            data_found,
            output_scale_name,
            output_zp_name,
            _,
            _,
        ) = self.quantizer._get_quantization_params(node.output[0], out_scale, out_zero_point)

        # get quantized input tensor names, quantize input if needed
        (
            quantized_input_names,
            input_zero_point_names,
            input_scale_names,
            nodes,
        ) = self.quantizer.quantize_activation(node, [0])

        if not data_found or quantized_input_names is None:
            return super().quantize()

        # Create an entry for output quantized value.
        qlinear_output_name = node.output[0] + TENSOR_NAME_QUANT_SUFFIX
        quantized_output_value = QuantizedValue(
            node.output[0],
            qlinear_output_name,
            output_scale_name,
            output_zp_name,
            QuantizedValueType.Input,
        )
        self.quantizer.quantized_value_map[node.output[0]] = quantized_output_value

        # Create qlinear softmax node for given type
        kwargs = {}
        for attribute in node.attribute:
            kwargs.update(attribute_to_kwarg(attribute))
        kwargs["domain"] = ms_domain
        # make qlinearsoft has the real opset_version, its default SinceVersion would be 1
        kwargs["opset"] = self.quantizer.opset_version
        qlinear_node_name = node.name + "_quant" if node.name else ""
        qnode = onnx.helper.make_node(
            "QLinear" + node.op_type,
            [
                quantized_input_names[0],
                input_scale_names[0],
                input_zero_point_names[0],
                output_scale_name,
                output_zp_name,
            ],
            [qlinear_output_name],
            qlinear_node_name,
            **kwargs,
        )

        # add all newly created nodes
        nodes.append(qnode)
        self.quantizer.new_nodes += nodes
        return None


class QDQSoftmax(QDQOperatorBase):
    def quantize(self):
        super().quantize()
        output_name = self.node.output[0]
        quant_overrides = self.quantizer.tensor_quant_overrides.get(output_name, {})

        quant_type = self.quantizer.activation_qType
        if "quant_type" in quant_overrides:
            quant_type = quant_overrides["quant_type"].tensor_type

        if "scale" in quant_overrides and "zero_point" in quant_overrides:
            out_zero_point, out_scale = quant_overrides["zero_point"], quant_overrides["scale"]
        else:
            # Unless overridden by the user, force Softmax to range from 0.0 to 1.0
            rmin = quant_overrides.get("rmin", 0.0)
            rmax = quant_overrides.get("rmax", 1.0)
            symmetric = quant_overrides.get("symmetric", self.quantizer.is_activation_symmetric)
            reduce_range = quant_overrides.get("reduce_range", False)
            qmin, qmax = get_qmin_qmax_for_qType(quant_type, reduce_range=reduce_range, symmetric=symmetric)
            out_zero_point, out_scale = compute_scale_zp(rmin, rmax, qmin, qmax, symmetric=symmetric)

        self.quantizer.set_quant_scale_zp(output_name, (out_scale, out_zero_point))
