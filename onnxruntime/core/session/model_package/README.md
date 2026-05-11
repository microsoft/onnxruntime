# Model Package Format

This document describes the model package directory layout and the JSON files used by ONNX Runtime to discover and load model packages. All JSON files must be UTF-8 encoded.

## Definitions

- Model Package

  - A model package defines the overall logical ‘model’
  - A model package contains one or more ‘component models’
  - The component models are executed when running the model package to provide the overall functionality of the logical model
  - A model package may contain configuration information to support running multiple component models

- Component Model
  - A component model comprises one or more ‘model variants’
  - All variants have the same model inputs and outputs with the same shapes.
    - The data types may vary.

- Model Variant
  - A ‘model variant’ is a single ONNX or ORT format model.

## Directory layout

````
<model>.ortpackage/ 
├── manifest.json
├── pipeline.json
├── configs/ 
|   ├── genai_config.json 
|   └── chat_template.jinja
└── models/ 
    └── model_name/ 
        ├── metadata.json
        |   └── Contains general information on the component model,
        |       and specific information about each model variant
        |       such as data types, quantization algo, EP, etc. that
        |       is updated on add/remove of model variant
        └── shared_weights/ (shared weights from all variants)
            └── <checksum of weights file A>/
                └── model.data
            └── <checksum of weights file B>/
                └── model.data
            └── ...
        └── base model /   
            ├── model.onnx  
        └── variant A / 
            ├── optimized model.onnx (contains EPContext nodes) 
            └── [Compilation artifacts] 
        └── variant B / 
            ├── optimized model.onnx (contains EPContext nodes) 
            └── [Compilation artifacts] 
````


## Notes:
- Shared weights is not yet supported, but the format allows for it in the future.

## `manifest.json` (required)

Location: `<package_root>/manifest.json`

Purpose: Provides the overall package identity and (optionally) lists component models available in the package.

Schema:
- `model_name` (string, required): Logical package name.
- `model_version` (string, optional): Version of the model package.
- `component_models` (array of strings, optional): List of component model names. If this field is omitted, ONNX Runtime will discover component models by enumerating subdirectories under `models/`. If present, the names listed here must match the subdirectory names under `models/`.

### `manifest.json` example

```json
{
    "model_name":  <logical_model_name>,
    "model_version": "1.0",
    "component_models": [
        <component_model_name_1>,
        <component_model_name_2>
    ]
}
```

## `metadata.json` (required per component model)

Location: `<package_root>/models/<component_model>/metadata.json`

Purpose: Describes the variants available for a specific component model.

Schema:
- `component_model_name` (string, required): Name of the component model.
- `model_variants` (object, required): Map of variant names to variant descriptors.
  - `<variant_name>` (object, required):
    - `model_type` (string, optional): Type of the model (e.g., `"onnx"`, `"ORT"`). If omitted, ORT will treat it as an ONNX model by default.
    - `model_file` (string, optional): Path relative to the model variant directory. Can point to an ONNX model file or a directory. If it is a directory, or if `model_file` is omitted, ORT will discover the ONNX model file within that directory.
    - `model_id` (string, optional): Unique identifier for the model variant. It should match a catalog value if the model comes from a catalog. If `model_id` is present, the model will be in the <component_model_name>/`model_id`/ directory.
    - `constraints` (object, required):
      - `ep` (string, required (except base model)): Execution provider name (e.g., `"TensorrtExecutionProvider"`, `"QNNExecutionProvider"`, `"OpenVINOExecutionProvider"`).
      - `device` (string, optional): Target device type (e.g., `"cpu"`, `"gpu"`, `"npu"`). Must match a supported `OrtHardwareDevice`. If the EPContext model can support multiple device types, this field can be omitted and EP should record supported device types in `ep_compatibility_info` instead.
      - `architecture` (string, optional): Hardware architecture hint; interpreted by the EP if needed.
      - `ep_compatibility_info` (string, optional): EP-specific compatibility string (as produced by `OrtEp::GetCompiledModelCompatibilityInfo()`); validated by the EP when selecting a variant. **The compatibility value returned by the EP is critical—ORT uses it to rank and choose the model variant.**

### `metadata.json` example
```json
{
    "component_model_name":  <component_model_name>,
    "model_variants": {
        <variant_1>: {
            "model_type": "onnx",
            "model_file": "model_ctx.onnx",
            "constraints": {
                "ep": "TensorrtExecutionProvider",
                "ep_compatibility_info": "..."
            }
        },
        <variant_2>: {
            "model_type": "onnx",
            "model_file": "model_ctx.onnx",
             "constraints": {
                 "ep": "OpenVINOExecutionProvider",
                 "device": "cpu",
                 "ep_compatibility_info": "..."
             }
        }
    }
}
```


## Processing rules (runtime expectations)

- ONNX Runtime reads `manifest.json` if the path passed in is the package root directory; if `component_models` is present, it uses that to determine which component models to load. If `component_models` is not present, ONNX Runtime discovers component models by enumerating subdirectories under `models/`. (In this case, ONNX Runtime expects only one component model exist in the model package.)
- ONNX Runtime reads component model's `metadata.json` and ignores `manifest.json` if the path passed in points directly to a component model directory.
- For each component model, `metadata.json` supplies the definitive list of variants and constraints.
- Variant selection is performed by matching constraints (EP, device, `ep_compatibility_info`, and optionally architecture). **The EP’s returned compatibility value (e.g., `EP_SUPPORTED_OPTIMAL`, `EP_SUPPORTED_PREFER_RECOMPILATION`) is used to score and pick the winning model variant.**
- All file paths must be relative paths; avoid absolute paths to keep packages portable
