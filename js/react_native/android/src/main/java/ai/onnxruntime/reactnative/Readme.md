NOTE: The raw resources contain a mix of .ORT and .ONNX format models.

Originally the ORT React Native package used the ORT Mobile build, which only supports .ORT format models, and does not support the double or bool types.

We're now using the 'full' ORT build in the React Native package, so both model formats and all types are supported.

ONNX format models were added for the unsupported types.