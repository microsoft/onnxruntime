from onnxruntime.transformers.onnx_model import OnnxModel
import onnx
from onnx import helper
import numpy as np
import os


input_onnx_path = "D:\\fork\\onnxruntime\\onnxruntime\\python\\tools\\transformers\\onnx_models\\gpt2_past.onnx"
output_onnx_path = "D:\\fork\\onnxruntime\\onnxruntime\\python\\tools\\transformers\\onnx_models\\gpt2_faster_transformer.onnx"
test_data_dir = "D:\\fork\\onnxruntime\\onnxruntime\\python\\tools\\transformers\\onnx_models"


def convert_model():
    #if not os.path.exists(output_onnx_path):
    #    return

    model = onnx.load(input_onnx_path)
    onnx_model = OnnxModel(model)

    num_layers = 12
    max_sequence_length = 128

    nodes_to_add = []

    tensor_map = {
        "self_beta" : "transformer.h.{}.ln_1.bias",
        "self_gamma" : "transformer.h.{}.ln_1.weight",
        "self_q_kernel" : "transformer.h.{}.attn.c_attn.weight",
        "self_q_bias" : "transformer.h.{}.attn.c_attn.bias",
        "self_k_kernel": "self_q_kernel", #None,
        "self_k_bias": "self_q_bias", #None,
        "self_v_kernel": "self_q_kernel", #None,
        "self_v_bias": "self_q_bias", #None,
        "self_output_kernel": "transformer.h.{}.attn.c_proj.weight",
        "self_output_bias": "transformer.h.{}.attn.c_proj.bias",
        "ffn_beta": "transformer.h.{}.ln_2.bias",
        "ffn_gamma": "transformer.h.{}.ln_2.weight",
        "ffn_kernel1": "transformer.h.{}.mlp.c_fc.weight",
        "ffn_bias1": "transformer.h.{}.mlp.c_fc.bias",
        "ffn_kernel2": "transformer.h.{}.mlp.c_proj.weight",
        "ffn_bias2": "transformer.h.{}.mlp.c_proj.bias",
        "decoding_beta": "transformer.ln_f.bias",
        "decoding_gamma": "transformer.ln_f.weight",
        "embedding_table": "transformer.wte.weight",
        "embedding_kernel": "transformer.wte.weight",
        "position_encoding_table":"transformer.wpe.weight",
        # graph inputs that are passed through to the custom op.
        "attention_mask" : "",
        "start_ids": "",
        "min_start_length": "",
        "max_start_length": "",
        "start_lengths": ""
    }

    initializers = []
    initializer_names = []
    inputs = []
    for key, value in tensor_map.items():
        print(key)
        if value is None:
            inputs.append("")
            continue

        if value in tensor_map:
            inputs.append(value)
            continue

        if "{}" in value:
            layer_inputs = [value.format(i) for i in range(num_layers)]
            for input in layer_inputs:
                initializer = onnx_model.get_initializer(input)
                if initializer is None:
                    raise RuntimeError(f"Initializer does not exist: {input}")
                if initializer.name not in initializer_names:
                    initializers.append(initializer)
                    initializer_names.append(initializer.name)
            node = helper.make_node('Concat',
                                    inputs=layer_inputs,
                                    outputs=[key],
                                    name='Concat_' + key,
                                    axis=0)
            nodes_to_add.append(node)
        elif value:
            input = value
            initializer = onnx_model.get_initializer(input)
            if initializer is None:
                raise RuntimeError(f"Initializer does not exist: {input}")
            if initializer.name not in initializer_names:
                initializers.append(initializer)
                initializer_names.append(initializer.name)
            inputs.append(input)
            continue

        inputs.append(key)

    node = helper.make_node('DecodingGpt',
                            inputs=inputs,
                            outputs=["output_ids"],
                            name='DecodingGpt_1')
    node.domain = "com.microsoft"
    node.attribute.extend(
        [helper.make_attribute("batch_size", 2),
         helper.make_attribute("candidate_num", 1),
         helper.make_attribute("probability_threshold", 0.0),
         helper.make_attribute("max_seq_len", max_sequence_length),
         helper.make_attribute("head_num", 12),
         helper.make_attribute("size_per_head", 64),
         helper.make_attribute("num_layer", 12),
         helper.make_attribute("start_id", 50256),
         helper.make_attribute("end_id", 50256),
         helper.make_attribute("temperature", 1.0),
         helper.make_attribute("is_fuse_qkv", 1)
         ])
    nodes_to_add.append(node)

    from onnx import TensorProto
    # graph inputs
    attention_mask = helper.make_tensor_value_info('attention_mask', TensorProto.FLOAT, ['batch_size', 'sequence_length'])
    start_ids = helper.make_tensor_value_info('start_ids', TensorProto.INT32, ['batch_size', 'sequence_length'])
    min_start_length = helper.make_tensor_value_info('min_start_length', TensorProto.INT32, [1])
    max_start_length = helper.make_tensor_value_info('max_start_length', TensorProto.INT32, [1])
    start_lengths = helper.make_tensor_value_info('start_lengths', TensorProto.INT32, ['batch_size'])

    # graph outputs
    output_ids = helper.make_tensor_value_info('output_ids', TensorProto.INT32, [max_sequence_length, 'batch_size'])

    graph_def = helper.make_graph(
        nodes_to_add,
        'gpt2-faster-transformer',
        [attention_mask, start_ids, min_start_length, max_start_length, start_lengths],
        [output_ids],
        initializers
    )

    # Create the model (ModelProto)
    model_def = helper.make_model(graph_def, producer_name='onnxruntime.transformers', opset_imports=[helper.make_opsetid('', 11), helper.make_opsetid('com.microsoft', 1)])
    onnx.save(model_def, output_onnx_path)

def test_model():
    from onnxruntime import SessionOptions, InferenceSession, __version__ as ort_version, GraphOptimizationLevel
    sess_options = SessionOptions()
    execution_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

    # Need 16GB GPU memory with grpha optimization disabled, or more if enabled.
    ort_session = InferenceSession(
        output_onnx_path,
        sess_options,
        providers=execution_providers)
    

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", cache_dir="./cache_models",)
    from transformers import GPT2LMHeadModel
    #add the EOS token as PAD token to avoid warnings
    model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="./cache_models", pad_token_id=tokenizer.eos_token_id)
    input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt')
    """
    beam_outputs = model.generate(
        input_ids, 
        max_length=32,
        num_beams=2, 
        early_stopping=False
    )

    print("input_ids", input_ids)
    print("Output:", beam_outputs)
    #print(tokenizer.decode(beam_outputs[0], skip_special_tokens=True))
    for i, beam_output in enumerate(beam_outputs):
        print("{}: {}".format(i, tokenizer.decode(beam_output, skip_special_tokens=True)))
    """
    sample_output = model.generate(
        input_ids, 
        do_sample=True, 
        max_length=32, 
        top_k=2, 
        temperature=1.0
    )

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(sample_output[0], skip_special_tokens=True))


    #names = [i.name for i in sess.get_inputs()]
    old_batch_size, sequence_length = input_ids.shape
    batch_size = 2
    input_ids = input_ids.repeat(batch_size, 1)

    print("sequence_length", sequence_length)
    #sequence_length = len(input_ids)
    inputs = {
        "attention_mask" : np.ones((batch_size, sequence_length), dtype=np.float32),
        "start_ids": input_ids.cpu().numpy().astype(np.int32), #np.random.randint(50256, size=(batch_size, sequence_length), dtype=np.int32),
        "min_start_length": np.array([sequence_length], dtype=np.int32),
        "max_start_length": np.array([sequence_length], dtype=np.int32),
        "start_lengths": np.array([sequence_length] * batch_size, dtype=np.int32)
    }

    from bert_test_data import output_test_data
    all_inputs = [inputs]
    for i, inputs in enumerate(all_inputs):
        dir = os.path.join(test_data_dir, 'test_data_set_' + str(i))
        output_test_data(dir, inputs)

    print("inputs", inputs)
    result = ort_session.run(None, inputs)
    print("outputs", result)
    #print(tokenizer.decode(result[0][0], skip_special_tokens=True))

convert_model()
test_model()