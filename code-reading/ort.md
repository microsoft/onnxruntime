# data flow
https://github.com/microsoft/onnxruntime/blob/668c70ee11b6b20c56997a9bc68e93317674e803/onnxruntime/core/session/inference_session.h#L78
1. session.ctor(sess_option, provider)
2. session.load()
3. session.init()
4. session.run()

> python example
> c++ example
# doc files
# python apis
https://onnxruntime.ai/docs/api/python/index.html
    - onnxruntime.InferenceSession
    - onnxruntime.SessionOptions
      - add_free_dimension_override_by_name
      - enable_mem_pattern
      - enable_mem_reuse
      - enable_profiling
      - graph_optimization_level
      - inter_op_num_threads, intra_op_num_threads
      - register_custom_ops_library
    - onnxruntime.IOBinding
      - session.run_with_iobinding
      - onnxruntime.OrtValue
    - onnxruntime.RunOptions
      - log_severity_level
      - only_execute_path_to_fetches
      -
# c++ apis
