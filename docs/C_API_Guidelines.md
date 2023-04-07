# ORT API Guidelines

## Introduction

Our public C API is the main interface of ONNX runtime with our customers. This document endeavors to describe our expectations with respect to the documentation and the quality of the API entry points so we are better equipped to code review new contributions and address existing shortcomings.

The document is entitled guidelines. However, the expectation is that everyone understands it and adheres to it when implementing a new API or while reviewing contributions from others.

## Guidelines

### 1. Public API must be properly documented

All APIs must have a proper documentation header that includes:

* API summary includes any limitations, such as types that it operates on.
* Description of each of its arguments and whether it is an in, out or in/out argument. Please, document if the user is responsible for memory deallocation or object destruction and how that can be done. Document that strings are UTF-8 encoded.
* Describe its return value.

XML format is automatically supported by Visual Studio when one types 3 consecutive slashes. Both C++ and C# compilers can generate XML documentation with /doc switch as described here. The documentation then can be converted to HTML pages using tools such as Sandcastle. We will use XML format to document C#.

XML format does not support C. We will use Doxygen style to document C API and C++ warappers.

### 2. Public API must be declared using appropriate macros to ensure that they all have proper calling convention

Most of our APIs (application programming interfaces) are exported using a pointer table. Such APIs must be declared using ORT_API2_STATUS macro. APIs that exported directly from the shared library must be declared using ORT_API_STATUS macro. Example: OrtSessionOptionsAppendExecutionProvider_CUDA.

API implementation must be declared using ORT_API_STATUS_IMPL macro.

All new APIs that are exported via a pointer table must be added at the end of the table to maintain backward compatibility.

### 3. Public APIs that create/destroy an instance of an object must be declared using an established pattern and signature

If an API such as CreateSession creates an Ort object such as Session, Session class must be declared using ORT_RUNTIME_CLASS macro. The API must supply an entry point that destroys the instance of such an object. The entry point must be declared using ORT_CLASS_RELEASE and must return void.

### 4. Public API that may error out must return OrtStatus pointer on error or nullptr on success

No C++ exceptions must propagate through the C++/C boundaries. All C++ exceptions must be converted to OrtStatus instances at API boundaries. Such functions should return nullptr on success.

Macros API_IMPL_BEGIN and API_IMPL_END are helpful in this regard.

Cleanup API that destroys objects or simply deallocates memory must return void. Most of the time such API can never error out. Adding return status creates more uncertainty for the client and does not help in exception scenarios such as try/finally in C#. Returning void helps clients to write cleaner code and preserve original exception if any with its meaningful error message rather than memory deallocation failure.

This requirement will also help us to create C++ API wrappers that are exception safe.

Consider logging errors if you must rather than return them to the client.

Example: on Windows delete operator is implemented on top of HeapFree() which may return an error. However, delete never returns anything and can be relied upon as a no throw primitive for cleanup purposes.

### 5. Public API must not require calling code to cleanup anything when it errors out

When API errors out it must leave all its out parameters and buffers untouched, in its original condition. All memory allocations must be cleaned up and no memory leaks result.

The obvious exception in this rule is the actual OrtStatus that is dynamically allocated and must be released by the client using the corresponding API.

Some of the client code, notably in C#, attempts to detect which out arguments need a cleanup when an API errors out. The way it is done, out arguments are pre-set to a specific value, such as zero. If the API errors out, the client code attempts to cleanup if the out argument has changed.

Such a technique is error prone and dangerous, as the client has no way of finding out if the out argument has already been cleaned up by the API as should be the case. It may result in double free. One reason for this is our insufficient documentation. This also results in a convoluted hard to read code with nested try/finally/catch clauses.

It seems that most of our API is compliant with it. Some API zero out the out arguments right away. It is fine.

Examples of an API that are compliant with this requirement are: GetBoundOutputNames and GetBoundOutputValues.

### 6. Public API that allocates memory must take an allocator parameter to use during the allocation

APIs that require memory allocation to return results, must take the instance of an OrtAllocator to use for such allocations. This serves two purposes:

* The user may want to supply their own allocator to use. Many of our APIs do that.

* The API does not have to declare a separate entry point for deallocating memory specifically for its type of allocation as we already have such entry points. OrtAllocatorAlloc/OrtAllocatorFree.

Consider established patterns when APIs return multiple allocations.

### 7. Public API must document that all strings they accept, and return are UTF-8 encoded

All APIs must return and accept strings in UTF-8 encodings. We must be mindful of that when maintaining language bindings.

### 8. Use appropriate types

Use types that fall into established patterns. For example, we use int64_t for dimensions internally and in the API everywhere so no casting is required. Use size_t for counts and memory sizes.

### 9.  Adding a new API

Follow these guidelines and instructions in the source code.  "Rules on how to add a new Ort API version" in [onnxruntime_c_api.cc](https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/session/onnxruntime_c_api.cc).
