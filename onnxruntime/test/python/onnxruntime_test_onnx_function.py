import onnx
from onnx import parser, GraphProto, checker
from onnxruntime import InferenceSession
import unittest

class TestOnnxFunctions(unittest.TestCase):

    def test_parse_model(self) -> None:
        input = '''
            <
            ir_version: 8,
            opset_import: [ "" : 10, "local" : 1 ]
            >
            agraph (float[N, 128] X, float[128,10] W, float[10] B) => (float[N,10] C)
            {
            T = local.foo (X, W, B)
            C = local.square(T)
            }

            <
            opset_import: [ "" : 10 ],
            domain: "local",
            doc_string: "Function foo."
            >
            foo (x, w, b) => (c) {
            T = MatMul(x, w)
            S = Add(T, b)
            c = Softmax(S)
            }

            <
            opset_import: [ "" : 10 ],
            domain: "local",
            doc_string: "Function square."
            >
            square (x) => (y) {
            y = Mul (x, x)
            }
           '''
        model = onnx.parser.parse_model(input)
        checker.check_model(model)
        sess = InferenceSession(model.SerializeToString(), providers=['CPUExecutionProvider'])

if __name__ == '__main__':
    unittest.main()
