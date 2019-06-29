# Python Operator 
The Python Operator provides the capability to easily invoke any custom Python code within a single node of an ONNX graph using ONNX Runtime. This can be useful for quicker experimentation when a model requires operators that are not officially supported in ONNX and ONNX Runtime, particularly if there is already a Python implementation for the required functionality. This should be used with discretion in production scenarios, and all security or other risks should be considered.

## Design Overview
The feature can be found under [onnxruntime/core/language_interop_ops](onnxruntime/core/language_interop_ops).
All Python C API dependent code are compiled into a dynamic linked library named pywrapper.
Before calling into Python script, pywrapper will convert onnxruntime tensor(s) to numpy(s), which is converted back when completed.
<p>Here is a chart illustrating the calling sequence:
<pre>
onnxruntime                          pywrapper                          script
     |                                  |                                 |
     | ------------------------------>  |                                 |
     |       call with tensor(s)        | ------------------------------> |
     |                                  |         call with numpy(s)      | 
     |                                  |                                 | compute
     |                                  |  <----------------------------- |
     | <------------------------------  |           return numpys(s)      |
     |         return tensor(s)         |                                 |
</pre>

## How to Use
### Step 1
Build onnxruntime with `--config Release --enable_language_interop_ops --build_shared_lib` and override the existing onnxruntime binary with the latest. Then, copy onnxruntime_pywrapper.dll, libonnxruntime_pywrapper.so, or libonnxruntime_pywrapper.dylib to the path where the onnxruntime binary is located. 
**Notes:**
* It is recommended to compile within the Python environment where inferencing will happen. For example, if inferencing will happen in a conda env named myconda1, please compile the binary within that environment as well
* If `--numpy_version=...` is specified, the Python operator will build with that version.

### Step 2
Create an onnx model containing Python operator nodes:
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
                            W1 = '5', W2 = '7', W3 = '9') #optional, must all be strings
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
### Step 3
Implement mymodule.py:
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
### Step 4
Copy mymodule.py into Python sys.path, then reference with onnxruntime. On Windows, please set PYTHONHOME beforehand. It should point to directory where the python is installed, such as C:\Python37 or C:\ProgramData\Anaconda3\envs\myconda1 if it is in conda.

## Supported Data Types
* TensorProto.BOOL
* TensorProto.UINT8
* TensorProto.UINT16
* TensorProto.UINT32
* TensorProto.INT16
* TensorProto.INT32
* TensorProto.FLOAT
* TensorProto.DOUBLE

## Limitations
* On Windows, `--config Debug` has known issues. Please build with `--config RelWithDebInfo` if debugging symbols are needed.
* Due to Python C API restrictions, multi-threading is disabled so Python operators will run sequentially.

## Test Coverage
The operator has been tested on multiple platforms, with or without conda:

Platform | Python 3.5 | Python 3.6 | Python 3.7
----------- | ------------| -----------  | -----------
Windows | (conda) passed | (conda) passed | passed
Linux | (conda) passed | (conda) passed | passed
Mac |  (conda) passed | (conda) passed | (conda) passed

