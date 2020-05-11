import onnx
from onnx import helper
from onnx import TensorProto
from enum import Enum

def GenerateModel(model_name):
    nodes = [  # Attention subgraph
        helper.make_node("LayerNormalization", ["input_1", "layer_norm_weight", "layer_norm_bias"], 
                        ["layernorm_out"],
                         "layernorm",
                         axis=-1,
                         epsion=0.000009999999747378752),
    
        # q nodes
        helper.make_node("MatMul", ["layernorm_out", "matmul_q_weight"], ["matmul_q_out"], "matmul_q"),
        helper.make_node("Add", ["matmul_q_out", "add_q_weight"], ["add_q_out"], "add_q"),
        helper.make_node("Reshape", ["add_q_out", "reshape_weight_1"], ["reshape_q_out"], "reshape_q"),
        helper.make_node("Transpose", ["reshape_q_out"], ["transpose_q_out"], "transpose_q",
            perm=[0,2,1,3]),

        # k nodes
        helper.make_node("MatMul", ["layernorm_out", "matmul_k_weight"], ["matmul_k_out"], "matmul_k"),
        helper.make_node("Add", ["matmul_k_out", "add_k_weight"], ["add_k_out"], "add_k"),
        helper.make_node("Reshape", ["add_k_out", "reshape_weight_1"], ["reshape_k_out"], "reshape_k"),
        helper.make_node("Transpose", ["reshape_k_out"], ["transpose_k_out"], "transpose_k", 
            perm=[0,2,3,1]),

        # mask nodes
        helper.make_node("Constant", [], ["mask_input"], "constant", 
            value=helper.make_tensor('mask', TensorProto.FLOAT, 
            [1, 3], [0.0, 0.0, 0.0])),
        helper.make_node("Unsqueeze", ["mask_input"], ["unsqueeze0_out"], "unsqueeze0", axes=[1]),
        helper.make_node("Unsqueeze", ["unsqueeze0_out"], ["unsqueeze1_out"], "unsqueeze1", axes=[2]),
        helper.make_node("Sub", ["sub_weight", "unsqueeze1_out"], ["sub_out"], "sub"),
        helper.make_node("Mul", ["sub_out", "mul_weight"], ["mul_mask_out"], "mul_mask"),

        # qk nodes
        helper.make_node("MatMul", ["transpose_q_out", "transpose_k_out"], ["matmul_qk_out"], "matmul_qk"),
        helper.make_node("Div", ["matmul_qk_out", "div_weight"], ["div_qk_out"], "div_qk"),
        helper.make_node("Add", ["div_qk_out", "mul_mask_out"], ["add_qk_out"], "add_qk"),
        helper.make_node("Softmax", ["add_qk_out"], ["softmax_qk_out"], "softmax_qk", axis=3),

        # v nodes
        helper.make_node("MatMul", ["layernorm_out", "matmul_v_weight"], ["matmul_v_out"], "matmul_v"),
        helper.make_node("Add", ["matmul_v_out", "add_v_weight"], ["add_v_out"], "add_v"),
        helper.make_node("Reshape", ["add_v_out", "reshape_weight_1"], ["reshape_v_out"], "reshape_v"),
        helper.make_node("Transpose", ["reshape_v_out"], ["transpose_v_out"], "transpose_v", 
            perm=[0,2,1,3]),

        # qkv nodes
        helper.make_node("MatMul", ["softmax_qk_out", "transpose_v_out"], ["matmul_qkv_1_out"], "matmul_qkv_1"),
        helper.make_node("Transpose", ["matmul_qkv_1_out"], ["transpose_qkv_out"], "transpose_qkv",
            perm=[0,2,1,3]
        ),
        helper.make_node("Reshape", ["transpose_qkv_out", "reshape_weight_2"], ["reshape_qkv_out"], "reshape_qkv"),
        helper.make_node("MatMul", ["reshape_qkv_out", "matmul_qkv_weight"], ["matmul_qkv_2_out"], "matmul_qkv_2"),
        helper.make_node("Add", ["matmul_qkv_2_out", "add_qkv_weight"], ["add_qkv_out"], "add_qkv"),
        
        helper.make_node("Add", ["add_qkv_out", "layernorm_out"], ["output"], "add"),
    ]

    matmul_q_weights = [
        -0.10791015625, -0.04193115234375, 0.09051513671875, 0.025787353515625, 
        -0.11572265625, -0.126953125, -0.043304443359375, -0.02984619140625,
        0.033538818359375, -0.05755615234375, -0.04986572265625, -0.01558685302734375, 
        -0.0352783203125, 0.03546142578125, 0.05218505859375, 0.005565643310546875,
        -0.05950927734375, 0.0172119140625, 0.06646728515625, 0.046630859375, 
        0.031524658203125, 0.048614501953125, -0.11102294921875, -0.018463134765625,
        -0.0352783203125, 0.037200927734375, 0.082763671875, 0.1260986328125, 
        -0.1087646484375, 0.00566864013671875, -0.027191162109375, -0.0027103424072265625,
        -0.1256103515625, -0.0245361328125, 0.04437255859375, -0.05267333984375, 
        -0.0606689453125, 0.009735107421875, 0.01100921630859375, 0.045928955078125,
        -0.036834716796875, 0.005405426025390625, 0.04571533203125, 0.11767578125, 
        0.0286102294921875, -0.01071929931640625, -0.006378173828125, 0.0213470458984375,
        -0.1434326171875, -0.0975341796875, 0.031402587890625, 0.02880859375, 
        0.048004150390625, -0.028289794921875, 0.018157958984375, 0.061981201171875,
        -0.126953125, -0.03350830078125, 0.1297607421875, -0.0093841552734375, 
        -0.0258026123046875, -0.000560760498046875, 0.1123046875, -0.0560302734375
    ]

    matmul_k_weights = [
        0.022125244140625, -0.017730712890625, -0.03265380859375, -0.05108642578125, 
        0.0423583984375, 0.112060546875, 0.080810546875, 0.09375,
        -0.043182373046875, -0.05010986328125, -0.063720703125, -0.00824737548828125, 
        0.1492919921875, 0.048431396484375, -0.0482177734375, -0.1123046875,
        -0.00719451904296875, -0.0229949951171875, -0.03424072265625, 0.0152435302734375, 
        0.023468017578125, 0.0301513671875, -0.04656982421875, -0.043701171875,
        0.040313720703125, 0.00644683837890625, -0.0186614990234375, 0.0261383056640625, 
        0.09063720703125, -0.078369140625, -0.05841064453125, -0.0743408203125,
        0.040130615234375, -0.0782470703125, 0.03729248046875, -0.07537841796875, 
        -0.0006098747253417969, 0.0285186767578125, -0.0518798828125, -0.01404571533203125,
        -0.08001708984375, 0.015960693359375, -0.0357666015625, -0.048065185546875, 
        0.01461029052734375, 0.06365966796875, 0.10125732421875, -0.00481414794921875,
        0.056182861328125, 0.072998046875, -0.06591796875, -0.035064697265625, 
        -0.1356201171875, -0.055877685546875, 0.06793212890625, -0.1292724609375,
        0.054901123046875, -0.0021762847900390625, 0.059783935546875, -0.035430908203125, 
        0.0528564453125, 0.035125732421875, -0.0186767578125, -0.062286376953125
    ]
    matmul_v_weights = [
        -0.03643798828125, 0.02862548828125, 0.039764404296875, 0.06097412109375, 
        -0.002288818359375, -0.10797119140625, -0.01171875, 0.041717529296875,
        0.032196044921875, 0.0135650634765625, 0.020233154296875, -0.05084228515625, 
        -0.011260986328125, -0.1241455078125, -0.0101165771484375, -0.00490570068359375,
        -0.01361083984375, -0.01454925537109375, -0.000637054443359375, -0.01534271240234375, 
        -0.0438232421875, 0.034332275390625, 0.011962890625, -0.0139617919921875,
        0.03363037109375, 0.0265350341796875, 0.039947509765625, -0.0268707275390625, 
        0.03900146484375, 0.08172607421875, 0.015625, 0.010986328125,
        0.0240325927734375, -0.029022216796875, 0.01403045654296875, 0.0135650634765625, 
        -0.0174102783203125, 0.07305908203125, -0.0231170654296875, 0.011444091796875,
        0.006130218505859375, 0.06268310546875, -0.05902099609375, -0.0109100341796875, 
        0.0185089111328125, 0.0161590576171875, 0.0185546875, 0.032440185546875,
        0.0011491775512695312, 0.01153564453125, 0.005832672119140625, -0.0538330078125,
        -0.008056640625, 0.01096343994140625, 0.037811279296875, 0.05902099609375,
        0.0394287109375, 0.00004678964614868164, -0.03778076171875, 0.004573822021484375, 
        -0.0237274169921875, -0.0124969482421875, -0.045013427734375, -0.04217529296875
        ]

    matmul_qkv_weights = [
        -0.04888916015625, 0.0143280029296875, 0.066650390625,-0.0343017578125,
        -0.0010356903076171875, -0.00048232078552246094, 0.07470703125, -0.04736328125,
        0.01454925537109375, -0.0086669921875, -0.051971435546875, -0.0201568603515625,
        0.040435791015625, -0.019256591796875, 0.0205078125, 0.0111541748046875,
        0.0071868896484375, -0.0298309326171875, -0.0306549072265625, -0.0225372314453125,
        -0.04193115234375, 0.07073974609375, -0.048065185546875, 0.0198822021484375,
        -0.035552978515625, -0.022796630859375, 0.03839111328125, 0.007099151611328125,
        -0.0080108642578125, -0.0017957687377929688, 0.0266265869140625,-0.028289794921875,
        0.0032901763916015625, 0.0208740234375, -0.01529693603515625, -0.046600341796875,
        -0.034637451171875, 0.011322021484375, -0.026458740234375, 0.04656982421875,
        -0.0091705322265625, 0.017913818359375, -0.019256591796875, -0.001216888427734375,
        -0.08245849609375, -0.023162841796875, -0.04132080078125, -0.03363037109375,
        0.0029315948486328125, 0.03173828125, -0.004024505615234375, 0.04534912109375,
        -0.0036163330078125, -0.03912353515625, -0.00800323486328125, 0.058197021484375,
        0.05572509765625, 0.01165771484375, 0.06756591796875, 0.05816650390625,
        -0.0654296875, -0.0241851806640625, 0.0205535888671875, -0.031707763671875
    ]

    add_q_weight = [-0.23681640625, -0.16552734375, 0.2191162109375, -0.1756591796875,
                    -0.03460693359375, -0.05316162109375, -0.336181640625, -0.253662109375]

    add_k_weight = [0.0246734619140625, 0.011993408203125, 0.0178375244140625, 0.00998687744140625,
                    0.0255126953125, 0.076416015625, -0.040771484375, 0.0107879638671875]

    add_v_weight = [-0.005893707275390625, -0.00916290283203125, 0.04541015625, 0.0159454345703125,
                    -0.0029163360595703125, -0.03472900390625, 0.0535888671875, 0.0091094970703125]
    
    add_qkv_weight = [-0.1146240234375, -0.06768798828125, -0.10040283203125, -0.07012939453125,
                    -0.08624267578125, 0.1507568359375, -0.06634521484375, -0.0194549560546875]

    initializers = [  # initializers
        helper.make_tensor('layer_norm_weight', TensorProto.FLOAT, [8], [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0]),
        helper.make_tensor('layer_norm_bias', TensorProto.FLOAT, [8], [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]),
        helper.make_tensor('matmul_q_weight', TensorProto.FLOAT, [8, 8], matmul_q_weights),
        helper.make_tensor('matmul_k_weight', TensorProto.FLOAT, [8, 8], matmul_k_weights),
        helper.make_tensor('matmul_v_weight', TensorProto.FLOAT, [8, 8], matmul_v_weights),        
        helper.make_tensor('matmul_qkv_weight', TensorProto.FLOAT, [8, 8], matmul_qkv_weights),
        helper.make_tensor('div_weight', TensorProto.FLOAT, [1], [2]),
        helper.make_tensor('sub_weight', TensorProto.FLOAT, [1], [1.0]),
        helper.make_tensor('mul_weight', TensorProto.FLOAT, [1], [-10000]),
        helper.make_tensor('add_q_weight', TensorProto.FLOAT, [8], add_q_weight),
        helper.make_tensor('add_k_weight', TensorProto.FLOAT, [8], add_k_weight),
        helper.make_tensor('add_v_weight', TensorProto.FLOAT, [8], add_v_weight),
        helper.make_tensor('add_qkv_weight', TensorProto.FLOAT, [8], add_qkv_weight),
        helper.make_tensor('reshape_weight_1', TensorProto.INT64, [4], [0, 0, 2, 4]),
        helper.make_tensor('reshape_weight_2', TensorProto.INT64, [3], [0, 0, 8]),
    ]

    graph = helper.make_graph(
        nodes,
        "AttentionFusionOneInput",  #name
        [  # inputs
            helper.make_tensor_value_info('input_1', TensorProto.FLOAT, [1, 3, 8])
        ],
        [  # outputs
            helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3, 8]),
        ],
        initializers)

    model = helper.make_model(graph)
    onnx.save(model, model_name)


GenerateModel('attention_mask_no_cast.onnx')

