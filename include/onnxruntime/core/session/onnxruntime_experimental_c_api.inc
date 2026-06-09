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
//   - Promoting to stable: Move the function to the stable OrtApi struct (append-only, drop the _SinceV<SinceVersion>
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
