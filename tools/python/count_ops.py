import onnx
import argparse
import os

num_ops = 0
op_counts = {}

def count_op(node):
    global num_ops
    global op_counts

    if node.op_type not in op_counts:
        op_counts[node.op_type] = 0

    num_ops += 1
    op_counts[node.op_type] += 1

def iterate_graph(graph):

    for node in graph.node:
        count_op(node)
        if node.op_type == "Scan" or node.op_type == "Loop":
            body_attribute = list(filter(lambda attr: attr.name == 'body', node.attribute))[0]
            iterate_graph(body_attribute.g)
        if node.op_type == "If":
            then_attribute = list(filter(lambda attr: attr.name == 'then_branch', node.attribute))[0]
            else_attribute = list(filter(lambda attr: attr.name == 'else_branch', node.attribute))[0]
            iterate_graph(then_attribute.g)
            iterate_graph(else_attribute.g)


def parse_args():
    parser = argparse.ArgumentParser(os.path.basename(__file__),
                                     description='Count all the operators in the model.')
    parser.add_argument('model', help='model file')
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = args.model
    model = onnx.load_model(model_path)

    iterate_graph(model.graph)

    print("Total operators: " + str(num_ops))
    print(sorted(op_counts.items()))

if __name__ == '__main__':
    main()
