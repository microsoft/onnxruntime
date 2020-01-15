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
sess = onnxrt.InferenceSession(args.model_path, sess_options)

df = pd.DataFrame([['string_input', True, np.float32(0.25)]], index=[0], columns=['F2', 'Label', 'F1'])

feed_helper = DataFrameTool(sess)
feeds = feed_helper.feed_nputs(df)

sess.run([], feeds)
```
