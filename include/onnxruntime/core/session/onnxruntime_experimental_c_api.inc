// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Central declaration list for experimental C API functions.
//
// This file is included multiple times with different definitions of ORT_EXPERIMENTAL_API to generate constructs like
// typedefs, name constants, registration table entries, and C++ accessors.
//
// The X-macro (https://en.wikipedia.org/wiki/X_macro) ORT_EXPERIMENTAL_API has the following signature:
//   ORT_EXPERIMENTAL_API(SinceVersion, ReturnType, Name, Params...)
//
//     SinceVersion: The ORT API version in which the function was first introduced.
//                   Must be a numeric literal (e.g., 28), not a macro like ORT_API_VERSION.
//     ReturnType:   The return type of the function (typically OrtStatusPtr).
//     Name:         The base name (without version suffix). By convention, prefixed with the target stable struct
//                   (e.g., OrtApi_Foo for functions destined for OrtApi).
//     Params:       The parameter list, using SAL annotations as in other ORT C API declarations.
//
// The function lookup name is mechanically generated as Name_SinceV<SinceVersion>.
//
// Experimental function lifecycle rules:
//   - Adding: Add one ORT_EXPERIMENTAL_API line with the current API version and implement it.
//   - Removing: Delete the line. The versioned name is inherently unique and cannot be reused.
//   - Changing a signature: Add a new entry at the current API version (producing a new name) and optionally delete
//     the old entry. Both can coexist during a transition period.
//   - Promoting to stable: Move the function to the stable API struct (append-only, drop the _SinceV<SinceVersion>
//     suffix). Delete the experimental entry.

/** \brief Test-only experimental function that exercises the lookup mechanism end-to-end.
 *
 * This entry is kept permanently for regression coverage of the experimental API mechanism.
 *
 * \param[out] out The output of this function, the value 12345.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtApi_ExperimentalApiTest, _Out_ int64_t* out)

// ---------------------------------------------------------------------------
// OrtModelPackageApi
//
// A model package is a directory containing one or more *components* (logical models).
// Each component has one or more *variants*, where each variant targets a single
// execution provider (EP). The package manifest declares the EP name, device type,
// and an optional compatibility string for every variant so that the runtime can
// automatically select the best variant for the hardware and EPs available in the
// caller's session options.
//
// The functions below support:
// - creating model package options that capture EP configuration from OrtSessionOptions,
// - loading a package context (manifest + metadata) from a package root path,
// - querying component/variant metadata including per-variant EP information,
// - selecting a component (which also resolves the best-matching variant),
// - querying the selected variant's name and folder path,
// - creating an OrtSession from the selected component context.
//
// Typical flow:
// 1) Create model package options:
//    - OrtModelPackageApi_CreateModelPackageOptionsFromSessionOptions
// 2) Load package metadata:
//    - OrtModelPackageApi_CreateModelPackageContext
// 3) Query metadata (optional):
//    - OrtModelPackageApi_ModelPackage_GetSchemaVersion
//    - OrtModelPackageApi_ModelPackage_GetComponentCount
//    - OrtModelPackageApi_ModelPackage_GetComponentNames
//    - OrtModelPackageApi_ModelPackage_GetVariantCount
//    - OrtModelPackageApi_ModelPackage_GetVariantNames
//    - OrtModelPackageApi_ModelPackage_GetVariantEpName
// 4) Select a component and resolve variant:
//    - OrtModelPackageApi_SelectComponent
// 5) Query selected variant info (optional):
//    - OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantName
//    - OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantFolderPath
// 6) Create session:
//    - OrtModelPackageApi_CreateSession
//
// Ownership:
// - Release objects created by this API with the corresponding release functions:
//   OrtModelPackageApi_ReleaseModelPackageOptions,
//   OrtModelPackageApi_ReleaseModelPackageContext,
//   OrtModelPackageApi_ReleaseModelPackageComponentContext.
//
// Opaque handles (OrtModelPackageOptions, OrtModelPackageContext, OrtModelPackageComponentContext)
// are declared in onnxruntime_experimental_c_api.h.
// ---------------------------------------------------------------------------

/** \brief Create model package options from an existing OrtSessionOptions.
 *
 * Captures EP configuration (registered execution providers and their devices) from
 * the session options for use during variant selection. The resulting OrtModelPackageOptions
 * is passed to OrtModelPackageApi_SelectComponent to resolve the best variant for the
 * available EPs.
 *
 * \param[in] env The ORT environment.
 * \param[in] session_options Session options containing registered EPs.
 * \param[out] out Receives the newly created OrtModelPackageOptions. Must be released
 *             with OrtModelPackageApi_ReleaseModelPackageOptions.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_CreateModelPackageOptionsFromSessionOptions,
                     _In_ const OrtEnv* env,
                     _In_ const OrtSessionOptions* session_options,
                     _Outptr_ OrtModelPackageOptions** out)

/** \brief Release an OrtModelPackageOptions created by
 *         OrtModelPackageApi_CreateModelPackageOptionsFromSessionOptions.
 */
ORT_EXPERIMENTAL_API(28, void, OrtModelPackageApi_ReleaseModelPackageOptions,
                     _Frees_ptr_opt_ OrtModelPackageOptions* options)

/** \brief Create a model package context by parsing the package at the given root path.
 *
 * Parses the manifest.json and component metadata from the specified directory.
 * The returned context provides read-only access to the package structure (components,
 * variants, EP declarations).
 *
 * \param[in] package_root Path to the model package root directory (containing manifest.json).
 * \param[out] out Receives the newly created OrtModelPackageContext. Must be released
 *             with OrtModelPackageApi_ReleaseModelPackageContext.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_CreateModelPackageContext,
                     _In_ const ORTCHAR_T* package_root,
                     _Outptr_ OrtModelPackageContext** out)

/** \brief Release an OrtModelPackageContext created by OrtModelPackageApi_CreateModelPackageContext. */
ORT_EXPERIMENTAL_API(28, void, OrtModelPackageApi_ReleaseModelPackageContext,
                     _Frees_ptr_opt_ OrtModelPackageContext* ctx)

/** \brief Get the schema version declared in the model package manifest.
 *
 * \param[in] ctx The model package context.
 * \param[out] out_version Receives the schema version number.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetSchemaVersion,
                     _In_ const OrtModelPackageContext* ctx,
                     _Out_ int64_t* out_version)

/** \brief Get the number of components in the model package.
 *
 * \param[in] ctx The model package context.
 * \param[out] out_count Receives the component count.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetComponentCount,
                     _In_ const OrtModelPackageContext* ctx,
                     _Out_ size_t* out_count)

/** \brief Get the names of all components in the model package.
 *
 * Returns a pointer to an array of UTF-8 component name strings. The array and its
 * strings are owned by `ctx` and remain valid until the context is released.
 *
 * \param[in] ctx The model package context.
 * \param[out] out_names Receives a pointer to an array of component name strings.
 * \param[out] out_count Receives the number of elements in the array.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetComponentNames,
                     _In_ const OrtModelPackageContext* ctx,
                     _Outptr_result_buffer_maybenull_(*out_count) const char* const** out_names,
                     _Out_ size_t* out_count)

/** \brief Get the number of variants for a given component.
 *
 * \param[in] ctx The model package context.
 * \param[in] component_name Name of the component to query.
 * \param[out] out_count Receives the variant count.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetVariantCount,
                     _In_ const OrtModelPackageContext* ctx,
                     _In_ const char* component_name,
                     _Out_ size_t* out_count)

/** \brief Get the names of all variants for a given component.
 *
 * Returns a pointer to an array of UTF-8 variant name strings. The array and its
 * strings are owned by `ctx` and remain valid until the context is released.
 *
 * \param[in] ctx The model package context.
 * \param[in] component_name Name of the component to query.
 * \param[out] out_variant_names Receives a pointer to an array of variant name strings.
 * \param[out] out_count Receives the number of elements in the array.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetVariantNames,
                     _In_ const OrtModelPackageContext* ctx,
                     _In_ const char* component_name,
                     _Outptr_result_buffer_maybenull_(*out_count) const char* const** out_variant_names,
                     _Out_ size_t* out_count)

/** \brief Get the EP name declared for a (component, variant) pair.
 *
 * Each variant targets a single EP. `out_ep` receives the EP name string.
 * When the variant does not declare an EP, the returned pointer is NULL.
 * String memory is owned by `ctx` and remains valid until the context is released.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetVariantEpName,
                     _In_ const OrtModelPackageContext* ctx,
                     _In_ const char* component_name,
                     _In_ const char* variant_name,
                     _Outptr_result_maybenull_ const char** out_ep)

/** \brief Select a component model and return an opaque component instance.
 *
 * The variant selection is also performed during this call based on the component
 * metadata and the provided options. The returned `OrtModelPackageComponentContext*` is
 * independent of `context` lifetime and must be released via
 * OrtModelPackageApi_ReleaseModelPackageComponentContext.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_SelectComponent,
                     _In_ const OrtModelPackageContext* context,
                     _In_ const char* component_name,
                     _In_ const OrtModelPackageOptions* options,
                     _Outptr_ OrtModelPackageComponentContext** out)

/** \brief Release an OrtModelPackageComponentContext created by OrtModelPackageApi_SelectComponent. */
ORT_EXPERIMENTAL_API(28, void, OrtModelPackageApi_ReleaseModelPackageComponentContext,
                     _Frees_ptr_opt_ OrtModelPackageComponentContext* ctx)

/** \brief Get the name of the selected variant after OrtModelPackageApi_SelectComponent has been called.
 *
 * String memory is owned by `ctx` and remains valid until the context is released.
 *
 * \param[in] ctx The component context returned by OrtModelPackageApi_SelectComponent.
 * \param[out] out_name Receives the selected variant's name string.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantName,
                     _In_ const OrtModelPackageComponentContext* ctx,
                     _Outptr_ const char** out_name)

/** \brief Get the folder path of the selected variant.
 *
 * Returns the resolved absolute path to the variant's directory on disk.
 * The string is owned by `ctx` and remains valid until the context is released.
 *
 * \param[in] ctx The component context returned by OrtModelPackageApi_SelectComponent.
 * \param[out] folder_path Receives the variant folder path string.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantFolderPath,
                     _In_ const OrtModelPackageComponentContext* ctx,
                     _Outptr_ const ORTCHAR_T** folder_path)

/** \brief Create an OrtSession for a selected file within a component model variant.
 *
 * The chosen variant (and thus its EP selection) is determined by `context`, which
 * was built from an OrtSessionOptions via OrtModelPackageApi_CreateModelPackageOptionsFromSessionOptions.
 *
 * Session options precedence:
 *   1. session_options == NULL (default path):
 *      ORT uses the OrtSessionOptions that was captured when `context` was created.
 *      Any variant-specific session and provider options declared in the package
 *      metadata are merged on top.
 *
 *   2. session_options != NULL (advanced path):
 *      ORT uses the caller-provided OrtSessionOptions as-is. Variant-specific
 *      session and provider options from the package metadata are NOT applied.
 *      Use this when custom EP setup is required (e.g., shared CUDA streams,
 *      shared QNN EP contexts, custom allocators).
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_CreateSession,
                     _In_ const OrtEnv* env,
                     _In_ OrtModelPackageComponentContext* context,
                     _In_opt_ const OrtSessionOptions* session_options,
                     _Outptr_ OrtSession** session)
