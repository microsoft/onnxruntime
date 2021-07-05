import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto


# Create graph with Add and Sub nodes that can be used to test partitioning when one of the operators
# can run using the test EP and the other cannot.
# As the operators take 2 inputs and produce one output we can easily create edges to test different scenarios
def create_model_1():
    # Assume the EP can handle either Add or Sub but not both, and we need to minimize the partitions for nodes
    # the EP can handle (as going to/from the EP has significant performance cost).
    #
    #   graph inputs
    #     /   \  \
    #    a1   s1  \
    #    |    / \  \
    #    |  a2  s2  \
    #    \     /    |
    #     \   /     |
    #      a3      a4
    #
    # Assuming the initial topological sort is top down, left to right, we get a1, s1, a2, s2, a3, a4
    #
    # Naively creating groups based on iterating this order and whether a node is supported gives the following groups
    # (a1), (s1), (a2), (s2), (a3, a4). This is similar to what most EPs do currently.
    #
    # If we also consider downstream nodes with all inputs available when adding via the topological sort we get two
    # less groups as s2 gets added with s1.
    # (a1), (s1, s2), (a2, a3, a4)
    #
    # If the EP handles Sub that's fine. If the EP handles Add that's not.
    #
    # Finally, if we do a partition aware sort, and prefer unhandled nodes first to maximize the inputs that would be
    # available each time we go to the EP, we can choose either of the root nodes (a1 or s1) to start at.
    #
    # If the EP is handling Sub we would start with a1 and get the same groups as above - which is perfectly fine as
    # there's a single partition with (s1, s2) run on the EP.
    #
    # If the EP is handling Add we would start with s1 (due to preferring unhandled nodes first) and get the following
    # groups, which also achieves a single partition of the handled nodes.
    # (s2, s2), (a1, a2, a3, a4)
    #
    # So if this model is loaded in a partitioning test, there should only be one partition running on the EP regardless
    # of whether Add or Sub is supported by it.
    graph = helper.make_graph(
        nodes=
        [
            helper.make_node("Add", ['input0', 'input1'], ['1'], "A1"),
            helper.make_node("Sub", ['input0', 'input1'], ["2"], "S1"),
            helper.make_node("Add", ['2', 'input1'], ['3_out'], "A2"),
            helper.make_node("Sub", ['2', 'input1'], ['4'], "S2"),
            helper.make_node("Add", ['1', '4'], ['5_out'], "A3"),
            helper.make_node("Add", ['input1', 'input2'], ['6_out'], "A4"),
        ],
        name="graph",
        inputs=
        [
            helper.make_tensor_value_info('input0', TensorProto.INT64, [1]),
            helper.make_tensor_value_info('input1', TensorProto.INT64, [1]),
            helper.make_tensor_value_info('input2', TensorProto.INT64, [1]),
        ],
        outputs=
        [
            helper.make_tensor_value_info('3_out', TensorProto.INT64, [1]),
            helper.make_tensor_value_info('5_out', TensorProto.INT64, [1]),
            helper.make_tensor_value_info('6_out', TensorProto.INT64, [1]),
        ],
        initializer=[]
    )

    model = helper.make_model(graph)
    return model


def create_model_2():
    # Create a model where there's a node that can't be run breaking up the partition.
    # Partition aware topo sort should give us
    #  s1, [a1, a2, a3, a5], s2, [a4, a6, a7]
    #
    #           graph inputs
    #               s1
    #                |
    #               a1
    #              /   \
    #             a2    s2
    #           /   \   /
    #          a3    a4
    #          |     |
    #         a5     a6
    #           \   /
    #            a7
    graph = helper.make_graph(
        nodes=
        [
            helper.make_node("Sub", ['input0', 'input1'], ['s1_out'], "S1"),
            helper.make_node("Add", ['s1_out', 'input2'], ['a1_out'], "A1"),
            helper.make_node("Add", ['a1_out', 'input0'], ['a2_out'], "A2"),
            helper.make_node("Sub", ['a1_out', 'input1'], ['s2_out'], "S2"),
            helper.make_node("Add", ['a2_out', 'input2'], ['a3_out'], "A3"),
            helper.make_node("Add", ['a2_out', 's2_out'], ['a4_out'], "A4"),
            helper.make_node("Add", ['a3_out', 'input0'], ['a5_out'], "A5"),
            helper.make_node("Add", ['a4_out', 'input1'], ['a6_out'], "A6"),
            helper.make_node("Add", ['a5_out', 'a6_out'], ['a7_out'], "A7"),
        ],
        name="graph",
        inputs=
        [
            helper.make_tensor_value_info('input0', TensorProto.INT64, [1]),
            helper.make_tensor_value_info('input1', TensorProto.INT64, [1]),
            helper.make_tensor_value_info('input2', TensorProto.INT64, [1]),
        ],
        outputs=
        [
            helper.make_tensor_value_info('a7_out', TensorProto.INT64, [1]),
        ],
        initializer=[]
    )

    model = helper.make_model(graph)
    return model


if __name__ == '__main__':
    model = create_model_1()
    onnx.save(model, 'ep_partitioning_test_1.onnx')
    model = create_model_2()
    onnx.save(model, 'ep_partitioning_test_2.onnx')
