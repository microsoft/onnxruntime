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

- Model Variant
  - A ‘model variant’ is typically a single ONNX or ORT format model, however we allow some flexibility here
    - An ORT GenAI ‘model variant’ is the collection of files required by ORT GenAI such as one or more ONNX models and related configuration files.



## Directory layout

````
<model>.ortpackage/ 
├── manifest.json
├── pipeline.json  
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
        └── model_name.generic-gpu/   
            ├── model.onnx 
            ├── [GenAI config and data files]  
        └── variant A / 
            ├── optimized model.onnx (contains EPContext nodes) 
            ├── [GenAI config and data files]  
            └── [Compilation artifacts] 
        └── variant B / 
            ├── optimized model.onnx (contains EPContext nodes) 
            ├── [GenAI config and data files]  
            └── [Compilation artifacts] 
````


## `manifest.json` (required)

Location: `<package_root>/manifest.json`

Purpose: Provides the overall package identity and (optionally) lists component models available in the package.

Schema:
- `name` (string, required): Logical package name.
- `component_models` (object, optional): Map of component model names to their descriptors.
  - `<component_model_name>` (object, required if present):
    - `model_variants` (object, optional): Map of variant names to variant descriptors.
      - `<variant_name>` (object, required if present):
        - `file` (string, optional): Path relative to the component model directory. Can point to an ONNX model file or a directory. If it is a directory, or if `file` is omitted, ORT will discover the ONNX model file within that directory.
        - `constraints` (object, required):
          - `ep` (string, required): Execution provider name (e.g., `"TensorrtExecutionProvider"`, `"QNNExecutionProvider"`, `"OpenVINOExecutionProvider"`).
          - `device` (string, optional): Target device type (e.g., `"cpu"`, `"gpu"`, `"npu"`). Must match a supported `OrtHardwareDevice`.
          - `architecture` (string, optional): Hardware architecture hint; interpreted by the EP if needed.
          - `ep_compatibility_info` (string, optional): EP-specific compatibility string (as produced by `OrtEp::GetCompiledModelCompatibilityInfo()`); validated by the EP when selecting a variant. **The compatibility value returned by the EP is critical—ORT uses it to rank and choose the model variant.**

Notes:
- `component_models` may be omitted. In that case, ORT will discover component models and rely on each component model’s `metadata.json` to enumerate variants.
- If `component_models` is present but a given component omits `model_variants`, its variants must be defined in that component’s `metadata.json`.
- All file paths are relative to the component model’s directory unless stated otherwise.

### `manifest.json` examples

**Minimal with no component list (metadata.json drives discovery):**
```json
{
    "name":  <logical_model_name>,
    "component_models": { // optional, if missing, ORT will discover component models by looking for folders with metadata.json under model_package_root/models
        <model_name_1>: {
           …  // could be empty.
        },
    }
}
```


**Multiple variants with differing constraints:**
```json
{
    "name":  <logical_model_name>,
    "component_models": {
        <model_name_1>: {
            "model_variants": {
                "variant_1": {
                    "file": "model_ctx_.onnx",
                    "constraints": {
                        "ep": "TensorrtExecutionProvider",
                        "device": "gpu",
                        "ep_compatibility_info": "device=gpu,npu;cuda_driver_version_support=..."
                    }
                },
                "variant_2": {
                    "file": "model_ctx_.onnx",
                    "constraints": {
                        "ep": "OpenVINOExecutionProvider",
                        "device": "cpu",
                        "ep_compatibility_info": "device=cpu;hardware_architecture=panther_lake;..."
                    }
                }
            }
        },
        <model_name_2>: { … }
    }
}
```


## `metadata.json` (required per component model)

Location: `<package_root>/<component_model>/metadata.json`

Purpose: Describes the variants available for a specific component model.

Schema:
- `model_name` (string, required): Name of the component model.
- `model_variants` (object, required): Map of variant names to variant descriptors.
  - `<variant_name>` (object, required):
    - `file` (string, required): Path relative to the component model directory. Can point to an ONNX model file or a directory. If it is a directory, or if `file` is omitted, ORT will discover the ONNX model file within that directory.
    - `constraints` (object, required):
      - `ep` (string, required: Execution provider name.
      - `device` (string, optional): Target device type (e.g., `"cpu"`, `"gpu"`, `"npu"`).
      - `architecture` (string, optional): Hardware architecture hint.
      - `ep_compatibility_info` (string, optional): EP-specific compatibility string (as produced by `OrtEp::GetCompiledModelCompatibilityInfo()`); validated by the EP when selecting a variant. **The compatibility value returned by the EP is critical—ORT uses it to rank and choose the model variant.**

### `metadata.json` example
```json
{
    "name":  <logical_model_name>,
    "model_variants": {
        "variant_1": {
            "file": "model_ctx_.onnx",
            "constraints": {
                "ep": "TensorrtExecutionProvider",
                "device": "gpu",
                "ep_compatibility_info": "device=gpu,npu;cuda_driver_version_support=..."
            }
        },
        "variant_2": {
            "file": "model_ctx_.onnx",
             "constraints": {
                 "ep": "OpenVINOExecutionProvider",
                 "device": "cpu",
                 "ep_compatibility_info": "device=cpu;hardware_architecture=panther_lake;..."
             }
        }
    }
}
```


## Processing rules (runtime expectations)

- ONNX Runtime reads `manifest.json` first to enumerate component models and any declared variants.
- For each component model, `metadata.json` supplies the definitive list of variants and constraints.
- Variant selection is performed by matching constraints (EP, device, `ep_compatibility_info`, and optionally architecture). **The EP’s returned compatibility value (e.g., `EP_SUPPORTED_OPTIMAL`, `EP_SUPPORTED_PREFER_RECOMPILATION`) is used to score and pick the winning model variant.**
- All file paths must be relative to the component model directory; avoid absolute paths to keep packages portable
