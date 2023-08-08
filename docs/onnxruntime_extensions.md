# ONNXRuntime Extensions

ONNXRuntime Extensions is a comprehensive package to extend the capability of the ONNX conversion and inference. Please visit the documentation [onnxruntime-extensions](https://github.com/microsoft/onnxruntime-extensions) to learn more about ONNXRuntime Extensions.

## Custom Operators Supported
onnxruntime-extensions supports many useful custom operators to enhance the text processing capability of ONNXRuntime, which include some widely used **string operators** and popular **tokenizers**. For custom operators supported and how to use them, please check the documentation [custom operators](https://github.com/microsoft/onnxruntime-extensions/blob/main/docs/custom_text_ops.md).

## Build ONNXRuntime with Extensions
We have supported build onnxruntime-extensions as a static library and link it into ONNXRuntime. To enable custom operators from onnxruntime-extensions, you should add argument `--use_extensions`, which will use onnxruntime-extensions from git submodule in path cmake/external/onnxruntime-extensions **by default**.

If you want to build ONNXRuntime with a pre-pulled onnxruntime-extensions, pass extra argument `--extensions_overridden_path <path-to-onnxruntime-extensions>`.

Note: Please remember to use `--minimal_build custom_ops` when you build minimal runtime with custom operators from onnxruntime-extensions.

### Build with Operators Config
Also, you could pass the **required operators config** file by argument `--include_ops_by_config` to customize the operators you want to build in both onnxruntime and onnxruntime-extensions. Example content of **required_operators.config** are:
```json
# Generated from model/s
# domain;opset;op1,op2...
ai.onnx;12;Add,Cast,Concat,Squeeze
ai.onnx.contrib;1;GPT2Tokenizer,
```

In above operators config, `ai.onnx.contrib` is the domain name of operators in onnxruntime-extensions. We would parse this line to generate required operators in onnxruntime-extensions for build.

### Generate Operators Config
To generate the **required_operators.config** file from model, please follow the guidance [Converting ONNX models to ORT format](https://onnxruntime.ai/docs/reference/ort-format-models.html#convert-onnx-models-to-ort-format).

If your model contains operators from onnxruntime-extensions, please add argument `--custom_op_library` and pass the path to **ortcustomops** shared library built following guidance [share library](https://github.com/microsoft/onnxruntime-extensions#the-share-library-for-non-python).

You could even manually edit the **required_operators.config** if you know the custom operators required and don't want to build the shared library.

### Build and Disable Exceptions
You could add argument `--disable_exceptions` to disable exceptions in both onnxruntime and onnxruntime-extensions.

However, if the custom operators you used in onnxruntime-extensions (such as BlingFireTokenizer) use c++ exceptions, then you will also need to add argument `--enable_wasm_exception_throwing_override` to enable **Emscripten** to link in exception throwing support library. If this argument is not set, Emscripten will throw linking errors.

### Example Build Command
```console
D:\onnxruntime> build.bat --config Release --build_wasm --enable_wasm_threads --enable_wasm_simd --skip_tests --disable_exceptions --disable_wasm_exception_catching --enable_wasm_exception_throwing_override --disable_rtti --use_extensions --parallel --minimal_build custom_ops --include_ops_by_config D:\required_operators.config
```

## E2E Example using Custom Operators
A common NLP task would probably contain several steps, including pre-processing, DL model and post-processing. It would be very efficient and productive to convert the pre/post processing code snippets into ONNX model since ONNX graph is actually a computation graph, and it can represent the most programming code, theoretically.

Here is an E2E NLP example to show the usage of onnxruntime-extensions:
### Create E2E Model
You could use ONNX helper functions to create an ONNX model with custom operators.
```python
import onnx
from onnx import helper

# ...
e2e_nodes = []

# tokenizer node
tokenizer_node = helper.make_node(
    'GPT2Tokenizer', # custom operator supported in onnxruntime-extensions
    inputs=['input_str'],
    outputs=['token_ids', 'attention_mask'],
    vocab=get_file_content(vocab_file),
    merges=get_file_content(merges_file),
    name='gpt2_tokenizer',
    domain='ai.onnx.contrib' # domain of custom operator
)
e2e_nodes.append(tokenizer_node)

# deep learning model
dl_model = onnx.load("dl_model.onnx")
dl_nodes = dl_model.graph.node
e2e_nodes.extend(dl_nodes)

# construct E2E ONNX graph and model
e2e_graph = helper.make_graph(
    e2e_nodes,
    'e2e_graph',
    [input_tensors],
    [output_tensors],
)
# ...
```
For more usage of ONNX helper, please visit the document [Python API Overview](https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md).

### Run E2E Model in Python
```python
import onnxruntime as _ort
from onnxruntime_extensions import get_library_path as _lib_path

so = _ort.SessionOptions()
# register onnxruntime-extensions library
so.register_custom_ops_library(_lib_path())

# run onnxruntime session
sess = _ort.InferenceSession(e2e_model, so)
sess.run(...)
```

### Run E2E Model in JavaScript
To run E2E ONNX model in JavaScript, you need to first [prepare ONNX Runtime WebAssembly artifacts](https://github.com/microsoft/onnxruntime/blob/main/js), include the generated `ort.min.js`, and then load and run the model in JS.
```js
// use an async context to call onnxruntime functions
async function main() {
    try {
        // create a new session and load the e2e model
        const session = await ort.InferenceSession.create('./e2e_model.onnx');

        // prepare inputs
        const tensorA = new ort.Tensor(...);
        const tensorB = new ort.Tensor(...);

        // prepare feeds: use model input names as keys
        const feeds = { a: tensorA, b: tensorB };

        // feed inputs and run
        const results = await session.run(feeds);

        // read from results
        const dataC = results.c.data;
        document.write(`data of result tensor 'c': ${dataC}`);

    } catch (e) {
        document.write(`failed to inference ONNX model: ${e}.`);
    }
}
```
