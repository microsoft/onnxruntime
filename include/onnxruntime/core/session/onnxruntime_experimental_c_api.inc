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

ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_CreateModelPackageOptionsFromSessionOptions,
                     _In_ const OrtEnv* env,
                     _In_ const OrtSessionOptions* session_options,
                     _Outptr_ OrtModelPackageOptions** out)

ORT_EXPERIMENTAL_API(28, void, OrtModelPackageApi_ReleaseModelPackageOptions,
                     _Frees_ptr_opt_ OrtModelPackageOptions* options)

ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_CreateModelPackageContext,
                     _In_ const ORTCHAR_T* package_root,
                     _Outptr_ OrtModelPackageContext** out)

ORT_EXPERIMENTAL_API(28, void, OrtModelPackageApi_ReleaseModelPackageContext,
                     _Frees_ptr_opt_ OrtModelPackageContext* ctx)

ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetSchemaVersion,
                     _In_ const OrtModelPackageContext* ctx,
                     _Out_ int64_t* out_version)

ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetComponentCount,
                     _In_ const OrtModelPackageContext* ctx,
                     _Out_ size_t* out_count)

ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetComponentNames,
                     _In_ const OrtModelPackageContext* ctx,
                     _Outptr_result_buffer_maybenull_(*out_count) const char* const** out_names,
                     _Out_ size_t* out_count)

ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetVariantCount,
                     _In_ const OrtModelPackageContext* ctx,
                     _In_ const char* component_name,
                     _Out_ size_t* out_count)

ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetVariantNames,
                     _In_ const OrtModelPackageContext* ctx,
                     _In_ const char* component_name,
                     _Outptr_result_buffer_maybenull_(*out_count) const char* const** out_variant_names,
                     _Out_ size_t* out_count)

ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackage_GetVariantEpName,
                     _In_ const OrtModelPackageContext* ctx,
                     _In_ const char* component_name,
                     _In_ const char* variant_name,
                     _Outptr_result_maybenull_ const char** out_ep)

ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_SelectComponent,
                     _In_ const OrtModelPackageContext* context,
                     _In_ const char* component_name,
                     _In_ const OrtModelPackageOptions* options,
                     _Outptr_ OrtModelPackageComponentContext** out)

ORT_EXPERIMENTAL_API(28, void, OrtModelPackageApi_ReleaseModelPackageComponentContext,
                     _Frees_ptr_opt_ OrtModelPackageComponentContext* ctx)

ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantName,
                     _In_ const OrtModelPackageComponentContext* ctx,
                     _Outptr_ const char** out_name)

ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_ModelPackageComponent_GetSelectedVariantFolderPath,
                     _In_ const OrtModelPackageComponentContext* ctx,
                     _Outptr_ const ORTCHAR_T** folder_path)

ORT_EXPERIMENTAL_API(28, OrtStatusPtr, OrtModelPackageApi_CreateSession,
                     _In_ const OrtEnv* env,
                     _In_ OrtModelPackageComponentContext* context,
                     _In_opt_ const OrtSessionOptions* session_options,
                     _Outptr_ OrtSession** session)
