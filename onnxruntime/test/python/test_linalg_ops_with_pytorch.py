import unittest
import onnx
from onnx import helper
import onnxruntime as ort
import torch

import numpy as np
import parameterized

# set generate_testcases to True to print C++ test cases
generate_testcases = True

def create_model(op, inputs, outputs, opset_version, node_kwargs):
    # create an onnx model with the given op
    input_names = [i.name for i in inputs]
    output_names = [o.name for o in outputs]
    node = helper.make_node(op, input_names, output_names, **node_kwargs)
    graph = helper.make_graph([node], f"test_graph_with_linalg_{op}", inputs, outputs)

    # TODO: remove onnx opset import
    opset_imports = [
        onnx.helper.make_opsetid("", opset_version),
        onnx.helper.make_opsetid("com.microsoft", 1)]
    meta = {
        "ir_version": 9,
        "opset_imports": opset_imports,
        "producer_name": "onnxruntime test",
        }
    model = onnx.helper.make_model(graph, **meta)
    onnx.checker.check_model(model)
    return model

def create_svd_model(use_batch, full_matrices):
    a_shape = ["B"] if use_batch else []
    if full_matrices:
        a_shape = [*a_shape, "M", "N"]
        u_shape = [*a_shape, "M", "M"]
        s_shape = [*a_shape, "N"]
        v_shape = [*a_shape, "N", "N"]
    else:
        a_shape = [*a_shape, "M", "N"]
        u_shape = [*a_shape, "M", "K"]
        s_shape = [*a_shape, "K"]
        v_shape = [*a_shape, "K", "N"]

    a_value_info = helper.make_tensor_value_info("A", onnx.TensorProto.FLOAT, a_shape)
    u_value_info = helper.make_tensor_value_info("U", onnx.TensorProto.FLOAT, u_shape)
    s_value_info = helper.make_tensor_value_info("S", onnx.TensorProto.FLOAT, s_shape)
    v_value_info = helper.make_tensor_value_info("V", onnx.TensorProto.FLOAT, v_shape)

    onnx_model = create_model(
        "LinalgSVD",
        [a_value_info],
        [u_value_info, s_value_info, v_value_info],
        opset_version=17,
        node_kwargs={"full_matrices": full_matrices, "domain": "com.microsoft",},
    )
    return onnx_model

def normalize_signs(a, b):
    signs = np.sign(a[0]) * np.sign(b[0])
    return a, b * signs, signs == -1

def validate_base_equal(actual, expected):
    sign_changes = 0
    if len(expected.shape) > 2:
        for actual_, expected_ in zip(actual, expected):
            validate_base_equal(actual_, expected_)
        return
    for i in range(expected.shape[1]):
        actual[:, i], expected[:, i], sign_changed = normalize_signs(actual[:, i], expected[:, i])
        if sign_changed:
            sign_changes += 1
    np.testing.assert_allclose(expected, actual, rtol=1e-5, atol=1e-7)
    assert(sign_changes % 2 == 0)

def format_tensor(tensor):
    # Reshape the tensor and convert it to a list of lists
    tensor_list = tensor.reshape(-1, tensor.shape[-1]).tolist()

    # Format each row of the tensor as a string
    tensor_str = ',\n      '.join([', '.join([f"{val:.6f}f" for val in row]) for row in tensor_list])

    return '{\n      ' + tensor_str + '\n    }'

class TestLinalgOps(unittest.TestCase):
    def setUp(self):
        self.opset_version = 17

    @parameterized.parameterized.expand([
        (True, True),
        (True, False),
        (False, True),
        (False, False),
        ])
    def test_linalg_svd(self, use_batch, full_matrices):
        torch.manual_seed(0)
        if use_batch:
            A = torch.randn(2, 3, 4)
        else:
            A = torch.randn(3, 4)
        U, S, Vh = torch.linalg.svd(A, full_matrices=full_matrices)

        onnx_model = create_svd_model(use_batch=use_batch, full_matrices=full_matrices)
        session = ort.InferenceSession(onnx_model.SerializeToString())
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        input_data = {input_name: A.numpy()}
        actual_u, actual_s, actual_vh = session.run(output_names, input_data)

        expected_u = U.numpy()
        expected_s = S.numpy()
        expected_vh = Vh.numpy()
        validate_base_equal(actual_u, expected_u)
        if len(expected_vh.shape) == 3:
            validate_base_equal(actual_vh.transpose((0, -1, -2)), expected_vh.transpose((0, -1, -2)))
        else:
            validate_base_equal(actual_vh.transpose(), expected_vh.transpose())
        np.testing.assert_allclose(actual_s, expected_s, rtol=1e-5, atol=1e-7)

        if generate_testcases:
            # Print the C++ test case
            A_str = format_tensor(A)
            U_str = format_tensor(actual_u)
            S_str = format_tensor(actual_s)
            Vh_str = format_tensor(actual_vh)

            batch_str = 'batch' if use_batch else 'no_batch'
            full_matrices_str = 'full_matrices' if full_matrices else 'no_full_matrices'
            test_case_name = f'{batch_str}_{full_matrices_str}'

            print(f'TYPED_TEST(LinalgSVDContribOpTest, {test_case_name}) {{\n'
                f'  OpTester test("LinalgSVD", 1, kMSDomain);\n'
                f'  test.AddAttribute("full_matrices", (int64_t){"1" if full_matrices else "0"});\n'
                f'  test.AddInput<TypeParam>("A", {{{", ".join(map(str, A.shape))}}}, {A_str});\n'
                f'  test.AddOutput<TypeParam>("U", {{{", ".join(map(str, U.shape))}}}, {U_str});\n'
                f'  test.AddOutput<TypeParam>("S", {{{", ".join(map(str, S.shape))}}}, {{{S_str}}});\n'
                f'  test.AddOutput<TypeParam>("Vh", {{{", ".join(map(str, Vh.shape))}}}, {Vh_str});\n'
                f'  test.Run();\n'
                f'}}')


if __name__ == '__main__':
    unittest.main()
