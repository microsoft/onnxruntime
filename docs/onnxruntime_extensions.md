# ONNXRuntime Extensions

ONNXRuntime Extensions is a comprehensive package to extend the capability of the ONNX conversion and inference. Please visit the documentation [onnxruntime-extensions](https://github.com/microsoft/onnxruntime-extensions) to learn more about ONNXRuntime Extensions.

## Custom Operators Supported
onnxruntime-extensions supports many useful custom operators to enhance the text processing capability of ONNXRuntime, which include some widely used **string operators** and popular **tokenizers**. For custom operators supported and how to use them, please check the documentation [custom operators](https://github.com/microsoft/onnxruntime-extensions/blob/main/docs/custom_text_ops.md).

## Build ONNXRuntime with Extensions
We have supported build onnxruntime-extensions as a static library and link it into ONNXRuntime. To enable custom operators in onnxruntime-extensions, you should add argument `--enable_onnxruntime_extensions` when build ONNXRuntime.

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
For more usage of ONNX helper, please visit the document [Python API Overview](https://github.com/onnx/onnx/blob/master/docs/PythonAPIOverview.md).

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
To run E2E ONNX model in JavaScript, you need to first [prepare ONNX Runtime WebAssembly artifacts](https://github.com/microsoft/onnxruntime/tree/master/js), include the generated `ort.min.js`, and then load and run the model in JS.
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