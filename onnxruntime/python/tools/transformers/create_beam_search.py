import argparse
import onnx
from onnx import helper, TensorProto

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument('-m',
                        '--model_name',
                        required=True,
                        type=str,
                        help='Model name')

    parser.add_argument('-o',
                        '--output_path',
                        required=True,
                        type=str,
                        help='Path where the model has to be saved')

    parser.add_argument('-s',
                        '--subgraph',
                        required=True,
                        type=str,
                        help='subgraph path for the custom kernel')

    args = parser.parse_args(argv)

    return args

def make_custombeamsearchop(args):
    model_type = "CustomBeamsearchOp"
    domain = "test.beamsearchop"
    inputs = ["input_ids", "num_beams"]
    
    outputs = ["logits"]

    model = onnx.load(args.subgraph)
    model.graph.name = f"customsubgraph"

    node = helper.make_node(model_type, inputs=inputs, outputs=outputs, name=f'{model_type}_0')
    node.domain = domain
    node.attribute.extend([
        helper.make_attribute("customsubgraph", model.graph),
        helper.make_attribute("sampleint", 64)
    ])

    input_ids = helper.make_tensor_value_info('input_ids', TensorProto.FLOAT, ['batch_size', 'sequence_length'])
    num_beams = helper.make_tensor_value_info('num_beams', TensorProto.FLOAT, [1])

    graph_inputs = [input_ids, num_beams]

    logits = helper.make_tensor_value_info("logits", TensorProto.INT32, ['batch_size', 'sequence_length'])
    graph_outputs = [logits]

    initializers = []
    graph = helper.make_graph([node], f'Testing_{model_type}', graph_inputs, graph_outputs, initializer=initializers)
    model = helper.make_model(graph, producer_name='onnxruntime.transformers')
    onnx.save(model, args.output_path)

def main(argv=None, sentences=None):
    args = parse_arguments(argv)
    
    if args.model_name == "custombeamsearchop":
        make_custombeamsearchop(args)
    else:
        raise Exception("Nothing else is supported")

if __name__ == '__main__':
    main()