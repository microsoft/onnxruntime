import onnx
from onnx import helper
from onnx import TensorProto
import numpy as np

def GenerateModel(model_name):
  nodes = [
    helper.make_node("Gather", ["embed_weights","input_1"], ["gather_out"], "gather"),

    helper.make_node("Add", ["gather_out", "add_q_weight"], ["add_q_out"], "add_q"),
    helper.make_node("Add", ["gather_out", "add_k_weight"], ["add_k_out"], "add_k"),
    helper.make_node("Add", ["gather_out", "add_v_weight"], ["add_v_out"], "add_v"),

    helper.make_node("Concat", ["add_q_out", "add_k_out", "add_v_out"], 
      ["concat_out"], "concat", axis=0),

    helper.make_node("Add", ["add_qkv_weight", "concat_out"], ["add_out"], "add"),
    helper.make_node("ReduceSum",["add_out"],["predictions"],"reduce_sum_1", axes=[0], keepdims=1),
  ]

  embed_weights = np.random.uniform(-1,1,8000).tolist()

  add_q_weight = [-0.23681640625, -0.16552734375, 0.2191162109375, -0.1756591796875,
              -0.03460693359375, -0.05316162109375, -0.336181640625, -0.253662109375]

  add_k_weight = [0.0246734619140625, 0.011993408203125, 0.0178375244140625, 0.00998687744140625,
                  0.0255126953125, 0.076416015625, -0.040771484375, 0.0107879638671875]

  add_v_weight = [-0.005893707275390625, -0.00916290283203125, 0.04541015625, 0.0159454345703125,
                  -0.0029163360595703125, -0.03472900390625, 0.0535888671875, 0.0091094970703125]
  
  initializers = [  # initializers
    helper.make_tensor('embed_weights', TensorProto.FLOAT, [1000, 8], embed_weights), 
    helper.make_tensor('add_q_weight', TensorProto.FLOAT, [8], add_q_weight),
    helper.make_tensor('add_k_weight', TensorProto.FLOAT, [8], add_k_weight),
    helper.make_tensor('add_v_weight', TensorProto.FLOAT, [8], add_v_weight),
    helper.make_tensor('add_qkv_weight', TensorProto.FLOAT, [1], [1.0]),
  ]

  graph = helper.make_graph(
    nodes,
    "ConcatThreeInputs",  #name
    [  # inputs
      helper.make_tensor_value_info('input_1', TensorProto.INT64, ['batch', 'seq_len'])
    ],
    [  # outputs
      helper.make_tensor_value_info('predictions', TensorProto.FLOAT, [1,1,8]),
    ],
    initializers)

  model = helper.make_model(graph)
  onnx.save(model, model_name)

GenerateModel('concat_trainable.onnx')

