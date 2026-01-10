# ONNXRuntime EP Context Model Generation with Weight Sharing

> [!NOTE]
> This tool is deprecated. Please use the public ONNX Runtime Python APIs to compile models with resource sharing. Refer to the example Python script at the end of this document.

[EP context with weight sharing design doc](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html#epcontext-with-weight-sharing)

OnnxRuntime provides the ep_weight_sharing_ctx_gen tool to automate the weight-sharing workflow. This tool handles the entire process. This tool is specifically designed for weight sharing scenarios, streamlining the EPContext model generation process.

Example command line:

```
ep_weight_sharing_ctx_gen [options...] model1_path,model2_path

Example: ./ep_weight_sharing_ctx_gen -e qnn -i "soc_model|60 htp_graph_finalization_optimization_mode|3" -C "ep.context_node_name_prefix|_part1" ./model1.onnx,./model2.onnx

Options:
        -e [qnn|tensorrt|openvino|vitisai]: Specifies the compile based provider 'qnn', 'tensorrt', 'openvino', 'vitisai'. Default: 'qnn'.
        -p [plugin_ep_config_json_file]: Specify JSON configuration file for a plugin EP. Takes precedence over the '-e' and '-i' options.

                                         Example JSON configuration that selects plugin EP devices via name:
                                           {
                                             "ep_library_registration_name": "example_plugin_ep",
                                             "ep_library_path": "example_plugin_ep.dll",
                                             "selected_ep_name": "example_plugin_ep",
                                             "default_ep_options": { "key": "value" }
                                           }

                                         Example JSON configuration that selects plugin EP devices via index:
                                           {
                                             "ep_library_registration_name": "example_plugin_ep",
                                             "ep_library_path": "example_plugin_ep.dll",
                                             "selected_ep_device_indices": [ 0 ],
                                             "default_ep_options": { "key": "value" }
                                           }
        -v: Show verbose information.
        -C: Specify session configuration entries as key-value pairs: -C "<key1>|<value1> <key2>|<value2>"
            Refer to onnxruntime_session_options_config_keys.h for valid keys and values.
            Force ep.context_enable to 1 and ep.context_embed_mode to 0. Change ep.context_file_path is not allowed.
            [Example] -C "ep.context_node_name_prefix|_part1"
        -i: Specify EP specific runtime options as key value pairs. Different runtime options available are:
            [Usage]: -i '<key1>|<value1> <key2>|<value2>'

            [QNN only] [backend_type]: QNN backend type. E.g., 'cpu', 'htp'. Mutually exclusive with 'backend_path'.
            [QNN only] [backend_path]: QNN backend path. E.g., '/folderpath/libQnnHtp.so', '/winfolderpath/QnnHtp.dll'. Mutually exclusive with 'backend_type'.
            [QNN only] [vtcm_mb]: QNN VTCM size in MB. default to 0(not set).
            [QNN only] [htp_graph_finalization_optimization_mode]: QNN graph finalization optimization mode, options: '0', '1', '2', '3', default is '0'.
            [QNN only] [soc_model]: The SoC Model number. Refer to QNN SDK documentation for specific values. Defaults to '0' (unknown).
            [QNN only] [htp_arch]: The minimum HTP architecture. The driver will use ops compatible with this architecture. eg: '0', '68', '69', '73', '75'. Defaults to '0' (none).
            [QNN only] [enable_htp_fp16_precision]: Enable the HTP_FP16 precision so that the float32 model will be inferenced with fp16 precision.
            Otherwise, it will be fp32 precision. Works for float32 model for HTP backend. Defaults to '1' (with FP16 precision.).
            [QNN only] [offload_graph_io_quantization]: Offload graph input quantization and graph output dequantization to another EP (typically CPU EP).
            Defaults to '1' (another EP (typically CPU EP) handles the graph I/O quantization and dequantization).
            [QNN only] [enable_htp_spill_fill_buffer]: Enable HTP spill file buffer, used while generating QNN context binary.
            [Example] -i "vtcm_mb|8 htp_arch|73"

        -h: help
```

# Example: Use Python APIs to compile models with resource sharing
Use of the public ORT Python APIs is now recommended for compiling models with resource (e.g., "weight") sharing.
The following snippet shows an example that compiles two models using an example plugin EP.

```Python
import onnxruntime
import os

def main():
    ep_name = "example_ep"
    ep_lib_path = "example_plugin_ep.dll"

    onnxruntime.register_execution_provider_library(ep_name, os.path.realpath(ep_lib_path))

    # Find one or more EP devices that correspond to the EP of interest.
    # In this example, we pick the first one.
    ep_device = next((d for d in onnxruntime.get_ep_devices() if d.ep_name == ep_name), None)

    # These are the names/paths to the input and output models.
    input_models = ["model_0.onnx", "model_1.onnx"]
    output_models = ["model_0_ctx.onnx", "model_1_ctx.onnx"]

    num_models = len(input_models)
    session_options = onnxruntime.SessionOptions()
    provider_options = {}  # Empty for this example

    # Set option that tells EP to share resources (e.g., weights) across sessions.
    session_options.add_session_config_entry("ep.share_ep_contexts", "1")
    session_options.add_provider_for_devices([ep_device], provider_options)

    # Compile individual models
    for i in range(len(input_models)):
        if i == num_models - 1:
            # Tell EP that this is the last compiling session that will be sharing resources.
            session_options.add_session_config_entry("ep.stop_share_ep_contexts", "1")

        model_compiler = onnxruntime.ModelCompiler(
            session_options,
            input_models[i],
            embed_compiled_data_into_model=False,
        )
        model_compiler.compile_to_file(output_models[i])

    onnxruntime.unregister_execution_provider_library(ep_name)
```
