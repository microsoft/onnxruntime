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

# Proposed Model Package APIs
```C++
ORT_RUNTIME_CLASS(ModelPackageOptions);
ORT_RUNTIME_CLASS(ModelPackageContext);

/** \brief APIs for loading, inspecting, and creating sessions from ONNX model packages.
 *
 * Obtain via OrtApi::GetModelPackageApi. Mirrors the shape of OrtCompileApi.
 *
 * Typical flow:
 *
 *   const OrtModelPackageApi* pkg = g_ort->GetModelPackageApi();
 *
 *   // 1. Capture EP selection (and other session-level settings) from session_options.
 *   OrtModelPackageOptions* options = nullptr;
 *   pkg->CreateModelPackageOptionsFromSessionOptions(env, session_options, &options);
 *
 *   // 2. Open the package and resolve variants using the captured EP selection.
 *   OrtModelPackageContext* ctx = nullptr;
 *   pkg->CreateModelPackageContext(env, package_root, options, &ctx);
 *
 *   // 3. Optionally inspect the package via query APIs.
 *   size_t component_count = 0;
 *   pkg->ModelPackageContext_GetComponentModelCount(ctx, &component_count);
 *
 *   const char* component_name = nullptr;
 *   pkg->ModelPackageContext_GetComponentModelName(ctx, 0, &component_name);
 *
 *   // 4. Create a session for a specific component model (and file, if the variant
 *   //    declares multiple files). Pass NULL for session_options to reuse the options
 *   //    captured in step 1 plus any variant-specific settings from the package metadata.
 *   OrtSession* session = nullptr;
 *   pkg->CreateSession(env, ctx, component_name, /*file_identifier*/ nullptr,
 *                      /*session_options*/ nullptr, &session);
 *
 *   // 5. Release in reverse order of creation.
 *   g_ort->ReleaseSession(session);
 *   pkg->ReleaseModelPackageContext(ctx);
 *   pkg->ReleaseModelPackageOptions(options);
 *
 * \since Version 1.XX.
 */
struct OrtModelPackageApi {
  ORT_CLASS_RELEASE(ModelPackageOptions);

  /** \brief Create an OrtModelPackageOptions from an OrtSessionOptions.
   *
   * Captures (by copy) the session-level settings that will be needed when creating a
   * session from the package. In particular, EP selection is captured from
   * `session_options`:
   *  - if `SessionOptionsAppendExecutionProvider_V2` was used, the appended OrtEpDevices
   *    and their EP options are captured directly;
   *  - else if `SessionOptionsSetEpSelectionPolicy` was used, the policy is resolved
   *    against `env`'s currently registered OrtEpDevices and the resulting OrtEpDevices
   *    are captured;
   *  - otherwise, no EP selection is captured (only unconstrained variants are
   *    eligible and the session falls back to CPU).
   *
   * After this call returns, `session_options` may be released by the caller.
   *
   * The resolved OrtEpDevices are cached on the options and reused by:
   *  - `CreateModelPackageContext` for variant selection;
   *  - `CreateSession` for actual session creation,
   *
   * ensuring the two never drift.
   */
  ORT_API2_STATUS(CreateModelPackageOptionsFromSessionOptions,
                  _In_ const OrtEnv* env,
                  _In_ const OrtSessionOptions* session_options,
                  _Outptr_ OrtModelPackageOptions** out);

  ORT_CLASS_RELEASE(ModelPackageContext);

  /** \brief Open and parse a model package, resolving variants against `options`.
   *
   * On success, the returned context caches the parsed manifest/metadata and the
   * variant chosen (per component model) using the EP selection captured on
   * `options`. `options` must have been created via
   * `CreateModelPackageOptionsFromSessionOptions`.
   *
   * `options` may be released by the caller after this call; the context captures
   * what it needs internally.
   */
  ORT_API2_STATUS(CreateModelPackageContext,
                  _In_ const OrtEnv* env,
                  _In_ const ORTCHAR_T* package_root,
                  _In_ const OrtModelPackageOptions* options,
                  _Outptr_ OrtModelPackageContext** out);

  // -- Query APIs (on the context) --------------------------------------------
  // Names kept in line with the proposed design doc.

  /** \brief Number of component models discovered in the package. */
  ORT_API2_STATUS(ModelPackageContext_GetComponentModelCount,
                  _In_ const OrtModelPackageContext* ctx,
                  _Out_ size_t* out_count);

  /** \brief Name of the component model at `index` (UTF-8). Pointer is owned by `ctx`. */
  ORT_API2_STATUS(ModelPackageContext_GetComponentModelName,
                  _In_ const OrtModelPackageContext* ctx,
                  _In_ size_t index,
                  _Outptr_ const char** out_name);

  /** \brief Number of variants declared for the component model at `component_index`. */
  ORT_API2_STATUS(ModelPackageContext_GetModelVariantCount,
                  _In_ const OrtModelPackageContext* ctx,
                  _In_ size_t component_index,
                  _Out_ size_t* out_count);

  /** \brief Get descriptive info for a given variant (ep/device/architecture/path). */
  ORT_API2_STATUS(ModelPackageContext_GetModelVariantInfo,
                  _In_ const OrtModelPackageContext* ctx,
                  _In_ size_t component_index,
                  _In_ size_t variant_index,
                  _Outptr_ const OrtModelVariantInfo** out_info);

  /** \brief Path of the variant selected for component model `component_index`.
   *
   * Two-call idiom: pass `path_buf=NULL` first to get `*required_size` in ORTCHARs.
   * Returns an error if no variant was selectable for the given EP selection.
   */
  ORT_API2_STATUS(ModelPackageContext_GetSelectedVariantPath,
                  _In_ const OrtModelPackageContext* ctx,
                  _In_ size_t component_index,
                  _Out_writes_opt_(path_buf_size) ORTCHAR_T* path_buf,
                  _In_ size_t path_buf_size,
                  _Out_ size_t* required_size);
};

struct OrtApi {
  ...

  /** \brief Create an OrtSession from a specific file within a component model variant.
   *
   * Session options precedence:
   *   1. session_options == NULL (default path):
   *      ORT uses the OrtSessionOptions that was captured when `context` was created.
   *      Any variant-specific session and provider options declared in the variant
   *      metadata are merged on top.
   *
   *   2. session_options != NULL (advanced path):
   *      ORT uses the caller-provided OrtSessionOptions as-is. Variant-specific
   *      session and provider options from the variant metadata are NOT applied.
   *      Use this when custom EP setup is required (e.g., shared CUDA streams,
   *      shared QNN EP contexts, custom allocators).
   *
   * \param env             Environment. Must be the same OrtEnv used to create `context`.
   * \param context         Loaded model package providing the resolved EP selection and
   *                        the chosen variant for `component_name`.
   * \param component_name  Component model whose selected variant should be loaded.
   * \param file_identifier Optional. Selects a file within the variant when the variant
   *                        declares multiple files. May be NULL if the variant has
   *                        exactly one file.
   * \param session_options Optional. See "Session options precedence" above.
   * \param[out] session    The created session.
   */
  ORT_API2_STATUS(CreateSession,
                  _In_ const OrtEnv* env,
                  _In_ const OrtModelPackageContext* context,
                  _In_ const char* component_name,
                  _In_opt_ const char* file_identifier,
                  _In_opt_ const OrtSessionOptions* session_options,
                  _Outptr_ OrtSession** session);

  ...

}

```
