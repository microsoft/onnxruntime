import onnx
from onnx import helper
from onnx import TensorProto
from onnx import shape_inference
import numpy as np

graph_def_0 = helper.make_graph(
    nodes=[
        helper.make_node(op_type="Shape", inputs=['A'], outputs=['A_shape'], name='shape0'),
        helper.make_node(op_type="Reshape", inputs=['A_shape', 'shape'], outputs=['A_reshaped'], name='reshape'),
        helper.make_node(op_type="Shape", inputs=['A_reshaped'], outputs=['A_shape1'], name='shape1'),
        helper.make_node(op_type="ConstantOfShape", inputs=['A_shape1'], outputs=['const1'], name='const1', value=helper.make_tensor('val', TensorProto.INT64,
            [1], [1])),
        helper.make_node(op_type="Mul", inputs=['const1', 'neg_one'], outputs=['mul'], name='mul'),
        helper.make_node(op_type="Equal", inputs=['A_reshaped', 'mul'], outputs=['equal'], name='equal'),
        helper.make_node(op_type="Where", inputs=['equal', 'const1', 'A_reshaped'], outputs=['where'], name='where'),
        helper.make_node(op_type="Expand", inputs=['B','where'], outputs=['C'], name='expand'),

    ],
    name='test-model',
    inputs=[
        # create inputs with symbolic dims
        helper.make_tensor_value_info("A", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, None),
    ],
    outputs=[
        helper.make_tensor_value_info('C', TensorProto.FLOAT, None)
    ],
    initializer=[
        helper.make_tensor('shape', TensorProto.INT64, [1], [-1]),
        helper.make_tensor('neg_one', TensorProto.INT64, [1], [-1]),
    ])

model = helper.make_model(graph_def_0, opset_imports=[helper.make_operatorsetid("", 12)])
onnx.save_model(model, "cpu_fallback_pattern_0.onnx")

graph_def_1 = helper.make_graph(
    nodes=[
        helper.make_node(op_type="Shape", inputs=['A'], outputs=['A_shape'], name='shape0'),
        helper.make_node(op_type="ConstantOfShape", inputs=['A_shape'], outputs=['const1'], name='const1', value=helper.make_tensor('val', TensorProto.INT64,
            [1], [1])),
        helper.make_node(op_type="Expand", inputs=['B','const1'], outputs=['C'], name='expand'),

    ],
    name='test-model',
    inputs=[
        # create inputs with symbolic dims
        helper.make_tensor_value_info("A", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, None),
    ],
    outputs=[
        helper.make_tensor_value_info('C', TensorProto.FLOAT, None)
    ],
    initializer=[])

model = helper.make_model(graph_def_1, opset_imports=[helper.make_operatorsetid("", 12)])
onnx.save_model(model, "cpu_fallback_pattern_1.onnx")


graph_def_2 = helper.make_graph(
    nodes=[
        helper.make_node(op_type="Size", inputs=['A'], outputs=['A_size'], name='size0'),
        helper.make_node(op_type="Range", inputs=['zero', 'A_size', 'two'], outputs=['range'], name='range'),
        helper.make_node(op_type="ReduceSum", inputs=['B', 'range'], outputs=['C'], name='reduce'),
    ],
    name='test-model',
    inputs=[
        # create inputs with symbolic dims
        helper.make_tensor_value_info("A", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, None),
    ],
    outputs=[
        helper.make_tensor_value_info('C', TensorProto.FLOAT, None)
    ],
    initializer=[
        helper.make_tensor('zero', TensorProto.INT64, [], [0]),
        helper.make_tensor('two', TensorProto.INT64, [], [2]),
    ])

model = helper.make_model(graph_def_2, opset_imports=[helper.make_operatorsetid("", 13)])
onnx.save_model(model, "cpu_fallback_pattern_2.onnx")


graph_def_3 = helper.make_graph(
    nodes=[
        helper.make_node(op_type="Size", inputs=['A'], outputs=['size0'], name='size0'),
        helper.make_node(op_type="Range", inputs=['zero', 'size0', 'two'], outputs=['range0'], name='range0'),
        helper.make_node(op_type="ReduceSum", inputs=['B', 'range0'], outputs=['reduce0'], name='reduce0'),

        helper.make_node(op_type="Identity", inputs=['reduce0'], outputs=['reduce0_cpy'], name='identity'),

        helper.make_node(op_type="Size", inputs=['reduce0_cpy'], outputs=['size1'], name='size1'),
        helper.make_node(op_type="Range", inputs=['zero', 'size1', 'two'], outputs=['range1'], name='range1'),
        helper.make_node(op_type="ReduceSum", inputs=['B', 'range1'], outputs=['reduce1'], name='reduce1'),

        helper.make_node(op_type="Sum", inputs=['reduce0', 'reduce1'], outputs=['C'], name='sum'),

    ],
    name='test-model',
    inputs=[
        # create inputs with symbolic dims
        helper.make_tensor_value_info("A", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("B", TensorProto.FLOAT, None),
    ],
    outputs=[
        helper.make_tensor_value_info('C', TensorProto.FLOAT, None)
    ],
    initializer=[
        helper.make_tensor('zero', TensorProto.INT64, [], [0]),
        helper.make_tensor('two', TensorProto.INT64, [], [2]),
    ])

model = helper.make_model(graph_def_3, opset_imports=[helper.make_operatorsetid("", 13)])
onnx.save_model(model, "cpu_fallback_pattern_3.onnx")

graph_def_4 = helper.make_graph(
    nodes=[
        helper.make_node(op_type="Size", inputs=['A'], outputs=['A_size'], name='size0'),
        helper.make_node(op_type="Range", inputs=['zero', 'A_size', 'two'], outputs=['range'], name='range'),
        helper.make_node(op_type="ReduceSum", inputs=['B', 'range'], outputs=['reduce'], name='reduce'),
        helper.make_node(op_type="ConstantOfShape", inputs=['reduce'], outputs=['const1'], name='const1', value=helper.make_tensor('val', TensorProto.INT64,
            [1], [1])),
        helper.make_node(op_type="Expand", inputs=['C','const1'], outputs=['D'], name='expand'),
        
    ],
    name='test-model',
    inputs=[
        # create inputs with symbolic dims
        helper.make_tensor_value_info("A", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("B", TensorProto.INT64, None),
        helper.make_tensor_value_info("C", TensorProto.FLOAT, None),
    ],
    outputs=[
        helper.make_tensor_value_info('D', TensorProto.FLOAT, None)
    ],
    initializer=[
        helper.make_tensor('zero', TensorProto.INT64, [], [0]),
        helper.make_tensor('two', TensorProto.INT64, [], [2]),
    ])

model = helper.make_model(graph_def_4, opset_imports=[helper.make_operatorsetid("", 13)])
onnx.save_model(model, "cpu_fallback_pattern_4.onnx")

graph_def_5 = helper.make_graph(
    nodes=[
        helper.make_node(op_type="Shape", inputs=['A'], outputs=['A_shape'], name='shape0'),
        helper.make_node(op_type="Gather", inputs=['A_shape', 'zero'], outputs=['batch'], name='gather0'),
        helper.make_node(op_type="Concat", inputs=['batch', 'seq_len'], outputs=['shape'], name='concat', axis=0),
        helper.make_node(op_type="Shape", inputs=['B'], outputs=['B_shape'], name='shape1'),
        helper.make_node(op_type="Gather", inputs=['B_shape', 'one'], outputs=['seq_len'], name='gather1'),
        helper.make_node(op_type="Reshape", inputs=['C','shape'], outputs=['D'], name='reshape'),
        
    ],
    name='test-model',
    inputs=[
        # create inputs with symbolic dims
        helper.make_tensor_value_info("A", TensorProto.FLOAT, None),
        helper.make_tensor_value_info("B", TensorProto.INT64, None),
        helper.make_tensor_value_info("C", TensorProto.FLOAT, None),
    ],
    outputs=[
        helper.make_tensor_value_info('D', TensorProto.FLOAT, None)
    ],
    initializer=[
        helper.make_tensor('zero', TensorProto.INT64, [1], [0]),
        helper.make_tensor('one', TensorProto.INT64, [1], [1]),
    ])

model = helper.make_model(graph_def_5, opset_imports=[helper.make_operatorsetid("", 13)])
onnx.save_model(model, "cpu_fallback_pattern_5.onnx")
