# DataFrameTool overview

This tool helps to feed data from an an instance of pandas DataFrame to a loaded ONNX model using ONNX Runtime API.

## Example of usage

See example of usage in feed_inputs_test.py in the same directory.

```python
import onnxruntime as onnxrt
import numpy as np
import pandas as pd

from feed_inputs import DataFrameTool

# Load the onnx model
sess_options = onnxrt.SessionOptions()
sess_options.enable_profiling = args.profile

df_tool = DataFrameTool(args.model_path, sess_options)

# Create a DataFrame that holds 3 inputs, string, bool, float in their respective columns
df = pd.DataFrame([['string_input', 3.25, 8, 16, 32, 64, True, 0.25]], 
                  columns=['StringInput', 'DoubleInput', 'Int8Input', 'Int16Input', 'Int32Input', 'Int64Input', 'BoolInput', 'Float32Input'])

outputs = df_tool.execute(df, [])
print('Outputs: ', outputs)

```
