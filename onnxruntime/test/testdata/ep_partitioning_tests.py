import numpy as np
import onnx
from onnx import helper
from onnx import TensorProto


# Create graph with Add and Sub nodes that can be used to test partitioning when one of the operators
# can run using the test EP and the other cannot.
# As the operators take 2 inputs and produce one output we can easily create edges to test different scenarios
def create_model_1():
    # Graph where the 3 'add' nodes should be able to be run as a single partition.
    # Naively you get 2 partitions if the topo sort is left to right, top down and by looking at edges as the bottom
    # right 'add' a4 is last in the topo sort so is separated from the first partition (a1 + a3 via edge, + a2 if next
    # in topo sort and added to the group).
    # But if you move the second 'sub' node up so it runs immediately after the first one, all the 'add' nodes can
    # be merged as all their inputs are available at that point
    #
    #        input0, input1, input2
    #         /  \  \
    #        s1  a1  \
    #       / \   \   \
    #     a2   s2  |   \
    #           \  |    \
    #             a3     a4
    graph = helper.make_graph(
        nodes=
        [
            helper.make_node("Sub", ['input0', 'input1'], ["1"], "S1"),
            helper.make_node("Add", ['input0', 'input1'], ['2'], "A1"),
            helper.make_node("Add", ['1', 'input1'], ['3_out'], "A2"),
            helper.make_node("Sub", ['1', 'input1'], ['4'], "S2"),
            helper.make_node("Add", ['2', '4'], ['5_out'], "A3"),
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
    #        input0, input1, input2
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
