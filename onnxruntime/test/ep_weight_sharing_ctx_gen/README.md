# ONNXRuntime EP Context Model Generation with Weight Sharing

[EP context with weight sharing design doc](https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html#epcontext-with-weight-sharing)

OnnxRuntime provides the ep_weight_sharing_ctx_gen tool to automate the weight-sharing workflow. This tool handles the entire process. This tool is specifically designed for weight sharing scenarios, streamlining the EPContext model generation process.

Example command line:

`ep_weight_sharing_ctx_gen [options...] model_1_path,model_2_path`

./ep_weight_sharing_ctx_gen -e qnn -v -i "soc_model|60 htp_graph_finalization_optimization_mode|3" /mnt/c/model1.onnx,/mnt/c/model2.onnx

Options:

    -e [qnn|tensorrt|openvino|vitisai]: Specifies the compile based provider qnn, tensorrt, openvino, vitisai. Default is qnn.

    -v: Show verbose information.

    -C: [session_config_entries]: Specify session configuration entries as key-value pairs: -C "<key1>|<val1> <key2>|<val2>"
                                  Refer to onnxruntime_session_options_config_keys.h for valid keys and values.
                                  [Example] -C "ep.context_enable|1 ep.context_embed_mode|0". These are set as default so can be ignored.

    -i: [provider_options]: Specify QNN EP specific runtime options as key value pairs. Different runtime options available are:
            [Usage]: -i '<key1>|<value1> <key2>|<value2>'

            [backend_path]: QNN backend path. e.g '/folderpath/libQnnHtp.so', '/winfolderpath/QnnHtp.dll'. Default to HTP backend lib in current folder.
            [vtcm_mb]: QNN VTCM size in MB. default to 0(not set).
            [htp_graph_finalization_optimization_mode]: QNN graph finalization optimization mode, options: '0', '1', '2', '3', default is '0'.
            [soc_model]: The SoC Model number. Refer to QNN SDK documentation for specific values. Defaults to '0' (unknown).
            [htp_arch]: The minimum HTP architecture. The driver will use ops compatible with this architecture. eg: '0', '68', '69', '73', '75'. Defaults to '0' (none).
            [enable_htp_fp16_precision]: Enable the HTP_FP16 precision so that the float32 model will be inferenced with fp16 precision.
            Otherwise, it will be fp32 precision. Only works for float32 model. Defaults to '0' (with FP32 precision.).
            [enable_htp_weight_sharing]: Allows common weights across graphs to be shared and stored in a single context binary. Defaults to '1' (enabled).
            [Example] -i "vtcm_mb|8 htp_arch|73"

    -h: help.

