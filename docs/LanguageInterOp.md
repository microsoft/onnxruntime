# LANGUAGE INTEROP OPERATORS
## Introduction
To facilitate Python developers on model inferencing, onnxruntime provides a way to invoke operators implemented in Python.
To use the feature, model designer needs to do following things:
1. Define the node with all required attributes;
2. Implement the Python operator in right format;
3. Before inferencing, place the Python module where the operator(s) are defined under python system path, then set PYTHONHOME environment variable;

## Example
To showcase the usage, let's first create an onnx model with muliple Python operators:

A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [3,3])
B = helper.make_tensor_value_info('B', TensorProto.FLOAT, [3,3])
C = helper.make_tensor_value_info('C', TensorProto.FLOAT, [3,3])
D = helper.make_tensor_value_info('D', TensorProto.FLOAT, [3,3])
E = helper.make_tensor_value_info('E', TensorProto.FLOAT, [3,3])
F = helper.make_tensor_value_info('F', TensorProto.FLOAT, [3,3])

ad1_node = helper.make_node('Add', ['A','B'], ['S'])
mul_node = helper.make_node('Mul', ['C','D'], ['P'])
py1_node = helper.make_node('PyOp',['S','P'], ['L','M','N'], domain = 'pyopmulti_1',
                            input_types  = [TensorProto.FLOAT, TensorProto.FLOAT],
                            output_types = [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT],
                            module = 'mymodule', class_name='Multi_1', W1='5', W2='7', W3='9')
ad2_node = helper.make_node('Add', ['L','M'], ['H'])
py2_node = helper.make_node('PyOp',['H','N','E'],['O','W'], domain = 'pyopmulti_2',
                            input_types  = [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT],
                            output_types = [TensorProto.FLOAT, TensorProto.FLOAT],
                            module = 'mymodule', class_name='Multi_2')
sub_node = helper.make_node('Sub', ['O','W'], ['F'])
graph = helper.make_graph([ad1_node,mul_node,py1_node,ad2_node,py2_node,sub_node], 'multi_pyop_graph', [A,B,C,D,E], [F])
model = helper.make_model(graph, producer_name='pyop_model')
onnx.save(model, './model.onnx')

Here is the graph layout:

A     B   C     D     E
 \   /     \   /     /
  Add       Mul     /
    \      /       /
      PyOp        /
      | | \      /
      Add  |    /
       \   |   /
         PyOp
         |  |
         Subt
           |
           F

Next, implement mymodule.py:

class Multi_1:
    def __init__(self,W1,W2,W3):
        self.W1 = int(W1)
        self.W2 = int(W2)
        self.W3 = int(W3)
    def compute(self,S,P):
        ret = S + P
        return ret+self.W1, ret+self.W2, ret+self.W3
class Multi_2:
    def compute(self,H,N,E):
        r1,r2 = H+N,N+E
        return r1,r2

Before inferencing, place mymodule.py into Python system path, then set PYTHONHOME as environment variable properly.
Finally, inference model.onnx with onnxruntime, Multi_1 and Multi_2 will each be instantiated with compute function triggered.

## Limitations
1. Currently, Python operator is only published via source, developers need to compile onnxruntime with --enable_language_interop_ops to enable the functionality. When compile is complete, override existing onnxruntime binary with the latest one from build. Also depends on the os, please copy onnxruntime_pywrapper.dll/libonnxruntime_pywrapper.so/libonnxruntime_pywrapper.dylib to the path where onnxruntime binary file is placed. 
2. On windows, Python operator will fail with memory exception when cmake config set to Debug. Please go with "--config RelWithDebInfo" if you need a binary with debug information;
3. Please specify a unique domain for each Python operator referred in the graph;
4. Due to limitation imposed by python C API, multi-threading is disabled, meaning multiple Python operators have to run sequentially;

## Pitfalls
1. In python module, developers are free to import any installed python packages for all kinds of purposes, this may introduce security issues, please go with caution;
2. Python modules are called via Python C API, the inferencing may be significantly less performant when any such operator is included in the graph;

## Tests

<TBD>
