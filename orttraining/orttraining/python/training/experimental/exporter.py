import torch
import torch.onnx.symbolic_helper
import torch.onnx.utils

# Only this version is tested.
onnx_version = 13


def _export_jit_graph_to_onnx_model_proto(graph, operator_export_type):
    from torch.onnx.symbolic_helper import _set_onnx_shape_inference, _set_operator_export_type, _set_opset_version

    _set_onnx_shape_inference(True)
    _set_operator_export_type(operator_export_type)
    _set_opset_version(onnx_version)
    torch._C._jit_pass_run_decompositions(graph)
    graph = torch.onnx.utils._optimize_graph(graph, operator_export_type, params_dict={})
    proto, _1, _2, _3 = graph._export_onnx(
        {}, onnx_version, {}, False, operator_export_type, False, False, {}, True, "", {}
    )
    return proto
