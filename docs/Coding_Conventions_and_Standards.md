# ONNX Runtime coding conventions and standards

## C++ Code Style

Google style from <https://google.github.io/styleguide/cppguide.html> with a few minor alterations:

* Max line length 120
  * Aim for 80, but up to 120 is fine.
* Exceptions
  * Allowed to throw fatal errors that are expected to result in a top level handler catching them, logging them and terminating the program.
* Non-const references
  * Allowed
  * Use a non-const reference for arguments that are modifiable but cannot be `nullptr` so the API clearly advertises the intent
  * Const correctness and usage of smart pointers (`shared_ptr` and `unique_ptr`) is expected. A non-const reference equates to "this is a non-null object that you can change but are not being given ownership of".
* Prefer passing `gsl::span<const T>` by value (or `std::span` when supported) as input arguments when passing const references to containers with contiguous storage (like `std::vector`). This allows the function to be container independent, and the argument to represent arbitrary memory spans or sub-spans. The below examples allow the client code to use either `std::vector` or `InlinedVector`. An instance of a `gsl::span` would be created automatically.
```cpp
/// Instead of
void foo(const std::vector<int64_t>&);

/// Use to pass any contiguous const container containing int64_t
// Now you can seamlessly pass either `std::vector`, `InlinedVector`, `std::array` or `gsl::span` as an argument.
void foo(gsl::span<const int64_t>);

// Example with pointer to const data. Instead of
void foo(const std::vector<const Node*>&);

// Use
void foo(gsl::span<const Node* const>);
```
* Prefer returning `gsl::span<const T>` by value instead of a const reference to a contiguous member container. Prefer returning `gsl::span` instead of a pointer referring to a chunk of memory. The size is also included in the span.
For example,
```cpp
// Instead of
const std::vector<int64_t>& foo();

// Return a span by value
gsl::span<const int64_t> foo();

// Instead of
const int64_t* foo();

// Return a span by value
gsl::span<const int64_t> foo();

```
* However, `std::initializer_list<T>` is not automatically convertible to a `gsl::span<const T>`.  Use `AsSpan({1, 2, 3})` defined at `core/common/span_utils.h` to convert `std::initializer_list<T>` to a span. You can also use `std::array`. For example,
```cpp
// Original code
void foo(const std::vector<std::string>&);

foo({"abc", "dbf"}); // Works

// After refactoring to gsl::span it would no longer compile. Use AsSpan().
void foo(gsl::span<const std::string>);

foo(AsSpan<std::string>{"abc", "dbf"}); // Works
```
* Prefer passing `std::string_view` by value instead of `const std::string&`. Make sure that the lifespan of a `std::string` instance ecplises the lifespan of the corresponding `std::string_view` instance.

* `using namespace` permitted with limited scope
  * Not allowing `using namespace` at all is overly restrictive. Follow the C++ Core Guidelines:
    * [SF.6: Use using namespace directives for transition, for foundation libraries (such as std), or within a local scope (only)](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-using)
    * [SF.7: Don't write using namespace at global scope in a header file](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-using-directive)

### Containers to use

Onnxruntime aims to reduce latency and latency variance by minimizing the amount of dynamic memory allocations.

* The use of the following container `typedef`s to reduce memory allocations is required:
  * Use `TensorShapeVector` typedef to build or modify shapes from `core/framework/tensor_shape.h`. It is based on a vector implementation that features small buffer optimization. Its small buffer size is the same to that of in TensorShape.
  * Use `InlinedVector<T>` typedef instead of std::vector defined at `core/common/inlined_containers_fwd.h`. By default, it provides 64 bytes of inlined storage. You can customize inlined size with the second template non-type parameter N.
  * Use `InlinedHashSet<T>` and `InlinedHashMap<T>` typedefs from `core/common/inlined_containers.h`. These are drop-in replacements for `std::unordered_set/map` that store their keys and values in one continuous buffer and reduce the number of allocations. They also do not allocate an `end` node when default constructed. Note, that these Hash containers do not provide pointer stability. `std::map` and `std::set` can often be replaced by hash containers as well.
  * For the node based containers where pointer stability is required, use `NodeHashSet` and `NodeHashMap`. Although node based, they are more cache friendly.
  * Use `core/common/inlined_containers_fwd.h` to forward declare any of the above container types.
  * Consider using `std::string_view` for use in containers to reduce the number of allocations and avoid string duplication. Keep in mind that the lifespan of the objects being referred to must eclipse the lifespan of the corresponding `std::string_view`.
  * We have selected to use `Abseil` library for the above typedefs. Abseil container documentation is [here](https://abseil.io/docs/cpp/guides/container#abseil-containers).
  * Do not use `Abseil` library or `absl` namespace directly. We should be able to build Onnxruntime without Abseil.
  * Use `onnxruntime/tools/natvis/abseil-cpp.natvis` for the above containers visualizations and debugging help in `VS Studio` and `VS Code`.
* Prefer using `reserve()` and not `resize()` on vectors. `resize()` default constructs all the elements for the size which can be expensive/noticeable even if the type is trivial. Default values are rarely used in practice and it becomes a waste. Construction like `std::vector<int>(10, 0)` is the same as `resize()` and is potentially wasteful.
* Use `reserve()` on hash containers and vectors. For example,
```cpp
#include "core/common/inlined_containers.h"

void foo(gsl::span<const std::string> names) {
  // For local processing, names are still valid
  // use std::string_view to avoid duplicate memory allocations.
  // same code would work with std::unordered_set if built without Abseil
  InlinedHashSet<std::string_view> unique_names;
  unique_names.reserve(names.size());
  unique_names.insert(names.cbegin(), names.cend());
}
```


### Other

* Qualify usages of `auto` with `const`, `*`, `&` and `&&` where applicable to more clearly express the intent
* When adding a new class, disable copy/assignment/move until you have a proven need for these capabilities. If a need arises, enable copy/assignment/move selectively, and when doing so validate that the implementation of the class supports what is being enabled.
  * Use `ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE` initially
  * See the other `ORT_DISALLOW_*` macros in <https://github.com/microsoft/onnxruntime/blob/main/include/onnxruntime/core/common/common.h>
* Sometimes, `std::unique_ptr` might be considered for delayed or optional construction of objects or members of classes. Instead, use `std::optional` as appropriate to reduce the number of allocations.
* Don't use `else` after `return`. see: [https://llvm.org/docs/CodingStandards.html#don-t-use-else-after-a-return](https://llvm.org/docs/CodingStandards.html#don-t-use-else-after-a-return)
* Don't overuse `std::shared_ptr`. Use `std::shared_ptr` only if it's not clear when and where the object will be de-allocated. See also: [https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rf-shared_ptr](https://isocpp.github.io/CppCoreGuidelines/CppCoreGuidelines#Rf-shared_ptr)
* Avoid using the `long` type, which could be either 32 bits or 64 bits.
* If there is a legitimate need to allocate objects on the heap, prefer using `std::make_unique()`. References for the reasoning:
  * <https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rh-make_unique>
  * <https://herbsutter.com/2013/05/29/gotw-89-solution-smart-pointers/>
  * <https://abseil.io/tips/126>
* Use [SafeInt](https://github.com/dcleblanc/SafeInt) when calculating the size of memory to allocate to protect against overflow errors
  * `#include "core/common/safeint.h"`
  * search for `SafeInt<size_t>` in the code for examples
* The following C++ warnings should never be disabled in onnxruntime VC++ projects(Required by [Binskim](https://github.com/microsoft/binskim/blob/d9afb65c89a621411efded74c27999281d87867e/src/BinSkim.Rules/PERules/BA2007.EnableCriticalCompilerWarnings.cs)).
  1. [4018](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-3-c4018) 'token' : signed/unsigned mismatch
  2. [4146](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-2-c4146?view=msvc-160) unary minus operator applied to unsigned type, result still unsigned
  3. [4244](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-2-c4244?view=msvc-160) 'argument' : conversion from 'type1' to 'type2', possible loss of data. For example, casting a int64_t to size_t.
  4. [4267](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-3-c4267?view=msvc-160) 'var' : conversion from 'size_t' to 'type', possible loss of data.
  5. [4302](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-2-c4302?view=msvc-160) 'conversion' : truncation from 'type 1' to 'type 2'
  6. [4308](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-2-c4308?view=msvc-160) negative integral constant converted to unsigned type
  7. [4532](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-1-c4532?view=msvc-160) 'continue' : jump out of \_\_finally/finally block has undefined behavior during termination handling
  8. [4533](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-1-c4533?view=msvc-160) initialization of 'variable' is skipped by 'instruction'
  9. [4700](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-1-and-level-4-c4700?view=msvc-160) uninitialized local variable 'name' used
  10. [4789](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-1-c4789?view=msvc-160) buffer 'identifier' of size N bytes will be overrun; M bytes will be written starting at offset L
  11. [4995](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-3-c4995?view=msvc-160) 'function': name was marked as #pragma deprecated
  12. [4996](https://docs.microsoft.com/en-us/cpp/error-messages/compiler-warnings/compiler-warning-level-3-c4996?view=msvc-160) Your code uses a function, class member, variable, or typedef that's marked deprecated

#### Clang-format

Clang-format will handle automatically formatting code to these rules. There’s a Visual Studio plugin that can format on save at <https://marketplace.visualstudio.com/items?itemName=LLVMExtensions.ClangFormat>, or alternatively the latest versions of Visual Studio 2017 include [clang-format support](https://blogs.msdn.microsoft.com/vcblog/2018/03/13/clangformat-support-in-visual-studio-2017-15-7-preview-1/).

There is a `.clang-format` file in the root directory that has the max line length override and defaults to the google rules. This should be automatically discovered by the clang-format tools.

## Code analysis

Visual Studio Code Analysis with [C++ Core guidelines](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md) rules enabled is configured to run on build for the `onnxruntime_common`, `onnxruntime_graph` and `onnxruntime_util` libraries. Updating the `onnxruntime_framework` and `onnxruntime_provider` libraries to enable Code Analysis and build warning free is pending.

Code changes should build with no Code Analysis warnings, however this is somewhat difficult to achieve consistently as the Code Analysis implementation is in fairly constant flux. Different minor releases may have less false positives (a build with the latest version may be warning free, and a build with an earlier version may not), or detect additional problems (an earlier version builds warning free and a later version doesn't).

We use [BinSkim Binary Analyzer](https://github.com/microsoft/binskim) to scan our binaries.

## Unit Testing and Code Coverage

There should be unit tests that cover the core functionality of the product, expected edge cases, and expected errors.
Code coverage from these tests should aim at maintaining over 80% coverage.

All changes should be covered by new or existing unit tests.

In order to check that all the code you expect to be covered by testing is covered, run code coverage in Visual Studio using 'Analyze Code Coverage' under the Test menu.

There is a configuration file in `onnxruntime/VSCodeCoverage.runsettings` that can be used to configure code coverage so that it reports numbers for just the onnxruntime code. Select that file in Visual Studio via the Test menu: `Test` -> `Test Settings` -> `Select Test Settings File`.

Using `Show Code Coverage Coloring` will allow you to visually inspect which lines were hit by the tests. See <https://docs.microsoft.com/en-us/visualstudio/test/using-code-coverage-to-determine-how-much-code-is-being-tested?view=vs-2017>.

## Linting

This project uses [lintrunner](https://github.com/suo/lintrunner) for linting. It provides a consistent linting experience locally and in CI. You can install the dependencies and initialize with

```sh
pip install lintrunner lintrunner-adapters
lintrunner init
```

This will install lintrunner on your system and download all the necessary
dependencies to run linters locally.
If you want to see what lintrunner init will install, run
`lintrunner init --dry-run`.

To lint local changes:

```bash
lintrunner
```

To format files and apply suggestions:

```bash
lintrunner -a
```

To lint all files:

```bash
lintrunner --all-files
```

To show help text:

```bash
lintrunner -h
```

To read more about lintrunner, see [wiki](https://github.com/pytorch/pytorch/wiki/lintrunner).
To update an existing linting rule or create a new one, modify `.lintrunner.toml` or create a
new adapter following examples in https://github.com/justinchuby/lintrunner-adapters.

## Python Code Style

Follow the [Black formatter](https://black.readthedocs.io)'s coding style when possible. A maximum line length of 120 characters is allowed for consistency with the C++ code.

Please adhere to the [PEP8 Style Guide](https://www.python.org/dev/peps/pep-0008/). We use [Google's python style guide](https://google.github.io/styleguide/pyguide.html) as the style guide which is an extension to PEP8.

Use `pyright`, which is provided as a component of the `pylance` extension in VS Code for static type checking.

Auto-formatting is done with `black` and `isort`. The tools are configured in `pyproject.toml`. From the root of the repository, you can run

```sh
lintrunner f --all-files
```

to format Python files.

Use `pydocstyle` to lint documentation styles. `pydocstyle` is enabled in VS Code.

## IDEs

### VS Code

VS Code is automatically configured with workspace configurations.

For Python development is VS Code, read
[this tutorial](https://code.visualstudio.com/docs/python/python-tutorial) for
more information.

### PyCharm

Follow [black's documentation](https://black.readthedocs.io/en/stable/integrations/editors.html#pycharm-intellij-idea) to set up the black formatter for PyCharm.

## Testing

We use the Python built-in [`unittest`](https://docs.python.org/3/library/unittest.html) framework for creating unit tests and [`pytest`](https://pytest.org) to run them. Use `pytest` to create tests only when `unittest` does not fit the need.

### Style

Test the *behavior*, instead of the *implementation*. To make what a test is testing clear, the test methods should be named following the pattern `test_<method or function name>_<expected behavior>_[when_<condition>]`.

e.g. `test_method_x_raises_error_when_dims_is_not_a_sequence`

## Objective-C/C++ Code Style

Please follow the [Google Objective-C/C++ Style Guide](https://google.github.io/styleguide/objcguide.html) with the following alteration(s):
- The maximum line length is 120 for consistency with the C++ code.

Clang-format can be used to format Objective-C/C++ code. The `.clang-format` file is in the repository root directory.
