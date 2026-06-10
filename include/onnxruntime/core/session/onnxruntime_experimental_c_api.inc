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
// A model package is a directory containing one or more components (logical models).
// Each component has one or more variants, where each variant targets a single execution
// provider (EP). The manifest declares the EP, device, and an optional compatibility
// string per variant so the runtime can pick the best-matching variant for the EPs
// configured in OrtSessionOptions.
//
// Typical flow:
//   1) CreateModelPackageOptionsFromSessionOptions(env, session_options, &opts)
//   2) CreateModelPackageContext(package_root, &ctx)
//   3) (optional) inspect components/variants via ModelPackage_GetXxx
//   4) SelectComponent(ctx, component_name, opts, &comp_ctx)
//   5) (optional) ModelPackageComponent_GetSelectedVariantName / FolderPath
//   6) CreateSession(env, comp_ctx, session_options_or_null, &session)
//
// Opaque handles (OrtModelPackageOptions, OrtModelPackageContext, OrtModelPackageComponentContext)
// are declared in onnxruntime_experimental_c_api.h.
// ---------------------------------------------------------------------------

/** \brief Create model package options from an existing OrtSessionOptions.
 *
 * Captures EP configuration from the session options for variant selection.
 *
 * \param[in] env The ORT environment.
 * \param[in] session_options Session options containing registered EPs.
 * \param[out] out Receives the OrtModelPackageOptions. Release with the
 *             OrtModelPackageApi_ReleaseModelPackageOptions experimental function.
 */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_CreateModelPackageOptionsFromSessionOptions,
                     _In_ const OrtEnv* env,
                     _In_ const OrtSessionOptions* session_options,
                     _Outptr_ OrtModelPackageOptions** out)

/** \brief Release model package options created by CreateModelPackageOptionsFromSessionOptions. */
ORT_EXPERIMENTAL_API(28, void, OrtModelPackageApi_ReleaseModelPackageOptions,
                     _Frees_ptr_opt_ OrtModelPackageOptions* options)

/** \brief Create a model package context by parsing the package at the given root path.
 *
 * Parses manifest.json and component metadata under `package_root`.
 *
 * \param[in] package_root Path to the model package root directory.
 * \param[out] out Receives the OrtModelPackageContext. Release with the
 *             OrtModelPackageApi_ReleaseModelPackageContext experimental function.
 */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_CreateModelPackageContext,
                     _In_ const ORTCHAR_T* package_root,
                     _Outptr_ OrtModelPackageContext** out)

/** \brief Release a model package context created by CreateModelPackageContext. */
ORT_EXPERIMENTAL_API(28, void, OrtModelPackageApi_ReleaseModelPackageContext,
                     _Frees_ptr_opt_ OrtModelPackageContext* ctx)

/** \brief Get the schema version declared in the manifest. */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetSchemaVersion,
                     _In_ const OrtModelPackageContext* ctx,
                     _Out_ int64_t* out_version)

/** \brief Get the number of components in the model package. */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetComponentCount,
                     _In_ const OrtModelPackageContext* ctx,
                     _Out_ size_t* out_count)

/** \brief Get the component names as a const char* array owned by ctx. */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetComponentNames,
                     _In_ const OrtModelPackageContext* ctx,
                     _Outptr_result_buffer_maybenull_(*out_count) const char* const** out_names,
                     _Out_ size_t* out_count)

/** \brief Get the number of variants for a given component. */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetVariantCount,
                     _In_ const OrtModelPackageContext* ctx,
                     _In_ const char* component_name,
                     _Out_ size_t* out_count)

/** \brief Get the variant names for a component as a const char* array owned by ctx. */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetVariantNames,
                     _In_ const OrtModelPackageContext* ctx,
                     _In_ const char* component_name,
                     _Outptr_result_buffer_maybenull_(*out_count) const char* const** out_variant_names,
                     _Out_ size_t* out_count)

/** \brief Get the EP name declared on a (component, variant). NULL when undeclared. */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetVariantEpName,
                     _In_ const OrtModelPackageContext* ctx,
                     _In_ const char* component_name,
                     _In_ const char* variant_name,
                     _Outptr_result_maybenull_ const char** out_ep)

/** \brief Select a component and resolve its best-matching variant.
 *
 * \param[in] context Model package context.
 * \param[in] component_name Name of the component to select.
 * \param[in] options Model package options carrying the EP configuration.
 * \param[out] out Receives the OrtModelPackageComponentContext. Release with the
 *             OrtModelPackageApi_ReleaseModelPackageComponentContext experimental function.
 */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_SelectComponent,
                     _In_ const OrtModelPackageContext* context,
                     _In_ const char* component_name,
                     _In_ const OrtModelPackageOptions* options,
                     _Outptr_ OrtModelPackageComponentContext** out)

/** \brief Release a component context created by SelectComponent. */
ORT_EXPERIMENTAL_API(28, void, OrtModelPackageApi_ReleaseModelPackageComponentContext,
                     _Frees_ptr_opt_ OrtModelPackageComponentContext* ctx)

/** \brief Get the selected variant's name from a component context. */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantName,
                     _In_ const OrtModelPackageComponentContext* ctx,
                     _Outptr_ const char** out_name)

/** \brief Get the selected variant's folder path from a component context. */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantFolderPath,
                     _In_ const OrtModelPackageComponentContext* ctx,
                     _Outptr_ const ORTCHAR_T** folder_path)

/** \brief Create an OrtSession for the selected variant's model file.
 *
 * Variant and EP selection are determined by `context`. When `session_options` is NULL,
 * ORT starts from a fresh OrtSessionOptions and merges the variant's session and provider
 * options from the package metadata. When `session_options` is non-NULL it is used as-is
 * (variant session/provider options from the package metadata are NOT merged).
 */
ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_CreateSession,
                     _In_ const OrtEnv* env,
                     _In_ OrtModelPackageComponentContext* context,
                     _In_opt_ const OrtSessionOptions* session_options,
                     _Outptr_ OrtSession** session)
