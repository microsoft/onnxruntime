import onnx
import numpy as np
class TransformerUtils:
    @staticmethod
    def make_initializer(name, data_type, dims, vals):
        data_arry = np.asarray(vals,
                                dtype=onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[data_type])
        return onnx.numpy_helper.from_array(data_arry.reshape(dims), name)
