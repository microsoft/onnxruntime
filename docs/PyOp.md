# Python Operator 
## Introduction
To facilitate Python coders on model developing, onnxruntime provides a way to invoke operators implemented in Python.

## Usage
Step 1, since Python operator is only published via source under onnxruntime/core/language_interop_ops, developers need to compile with“-enable_language_interop_ops”and override existing onnxruntime binary. Meanwhile, please also copy onnxruntime_pywrapper.dll or libonnxruntime_pywrapper.so or libonnxruntime_pywrapper.dylib to the path where onnxruntime binary is placed.
Note that it is suggested to compile within the Python environment where inferencing will happen. For example, if inferencing will happen in a conda env named myconda1, please compile the binary within that environment as well.
Step 2, create an onnx model containing Python operator nodes:
```python
ad1_node = helper.make_node('Add', ['A','B'], ['S'])
mul_node = helper.make_node('Mul', ['C','D'], ['P'])
py1_node = helper.make_node(op_type = 'PyOp', #required, must be 'PyOp'
                            inputs = ['S','P'], #required
                            outputs = ['L','M','N'], #required
                            domain = 'pyopmulti_1', #required, must be unique
                            input_types = [TensorProto.FLOAT, TensorProto.FLOAT], #required
                            output_types = [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT], #required
                            module = 'mymodule', #required
                            class_name = 'Multi_1', #required
                            compute = 'compute', #optional, 'compute' by default
                            W1 = '5', W2 = '7', W3 = '9') #optional, must be strings
ad2_node = helper.make_node('Add', ['L','M'], ['H'])
py2_node = helper.make_node('PyOp',['H','N','E'],['O','W'], domain = 'pyopmulti_2',
                            input_types = [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT],
                            output_types = [TensorProto.FLOAT, TensorProto.FLOAT],
                            module = 'mymodule', class_name = 'Multi_2')
sub_node = helper.make_node('Sub', ['O','W'], ['F'])
graph = helper.make_graph([ad1_node,mul_node,py1_node,ad2_node,py2_node,sub_node], 'multi_pyop_graph', [A,B,C,D,E], [F])
model = helper.make_model(graph, producer_name = 'pyop_model')
onnx.save(model, './model.onnx')
```
Step 3, implement mymodule.py:
```python
class Multi_1:
    def __init__(self, W1, W2, W3):
        self.W1 = int(W1)
        self.W2 = int(W2)
        self.W3 = int(W3)
    def compute(self, S, P):
        ret = S + P
        return ret + self.W1, ret + self.W2, ret + self.W3
class Multi_2:
    def compute(self, H, N, E):
        r1, r2 = H + N, N + E
        return r1, r2
```
Step 4, copy mymodule.py into one of the Python sys path, and do referencing with onnxruntime. On windows, please set PYTHONHOME beforehand. It points to path where the python is installed, such as C:\Python37 or C:\ProgramData\Anaconda3\envs\myconda1 if it is in conda.

## Supported Data Types
* TensorProto.BOOL,
* TensorProto.UINT8,
* TensorProto.UINT16,
* TensorProto.UINT32,
* TensorProto.INT16,
* TensorProto.INT32,
* TensorProto.FLOAT,
* TensorProto.DOUBLE

## Limitations
* On Windows,  "-config Debug" has known issues, so please compile with "-config RelWithDebInfo" if need debug information;
* Please specify a unique domain for each Python operator referred in the graph;
* Due to restrictions imposed by python C API, multi-threading is disabled, meaning multiple Python operators will run sequentially.

## Test
We haved tested the operator in multiple environments, with or without conda:

Platforms | Python 3.5 | Python 3.6 | Python 3.7
----------- | ------------| -----------  | -----------
Windows | (conda) passed | (conda) passed | passed
Linux | (conda) passed | (conda) passed | passed

