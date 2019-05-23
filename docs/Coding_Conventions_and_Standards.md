# ONNX Runtime coding conventions and standards


## Code Style

Google style from https://google.github.io/styleguide/cppguide.html with a few minor alterations:

* Max line length 120
  *	Aim for 80, but up to 120 is fine.
* Exceptions
  *	Allowed to throw fatal errors that are expected to result in a top level handler catching them, logging them and terminating the program.
* Non-const references
  *	Allowed
  * Use a non-const reference for arguments that are modifiable but cannot be nullptr so the API clearly advertises the intent
  *	Const correctness and usage of smart pointers (shared_ptr and unique_ptr) is expected. A non-const reference equates to “this is a non-null object that you can change but are not being given ownership of”.
* 'using namespace' permitted with limited scope
  * Not allowing 'using namespace' at all is overly restrictive. Follow the C++ Core Guidelines:
    * [SF.6: Use using namespace directives for transition, for foundation libraries (such as std), or within a local scope (only)](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-using)
    * [SF.7: Don't write using namespace at global scope in a header file](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md#Rs-using-directive)

Other
* Qualify usages of 'auto' with 'const', '*', '&' and '&&' where applicable to more clearly express the intent
* When adding a new class, disable copy/assignment/move until you have a proven need for these capabilities. If a need arises, enable copy/assignment/move selectively, and when doing so validate that the implementation of the class supports what is being enabled.
  * Use ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE initially
  * See the other ORT_DISALLOW_* macros in https://github.com/microsoft/onnxruntime/blob/master/include/onnxruntime/core/common/common.h

#### Clang-format

Clang-format will handle automatically formatting code to these rules. There’s a Visual Studio plugin that can format on save at https://marketplace.visualstudio.com/items?itemName=LLVMExtensions.ClangFormat, or alternatively the latest versions of Visual Studio 2017 include [clang-format support](https://blogs.msdn.microsoft.com/vcblog/2018/03/13/clangformat-support-in-visual-studio-2017-15-7-preview-1/).  

There is a .clang-format file in the root directory that has the max line length override and defaults to the google rules. This should be automatically discovered by the clang-format tools. 

## Code analysis

Visual Studio Code Analysis with [C++ Core guidelines](https://github.com/isocpp/CppCoreGuidelines/blob/master/CppCoreGuidelines.md) rules enabled is configured to run on build for the onnxruntime_common, onnxruntime_graph and onnxruntime_util libraries. Updating the onnxruntime_framework and onnxruntime_provider libraries to enable Code Analysis and build warning free is pending. 

Code changes should build with no Code Analysis warnings, however this is somewhat difficult to achieve consistently as the Code Analysis implementation is in fairly constant flux. Different minor releases may have less false positives (a build with the latest version may be warning free, and a build with an earlier version may not), or detect additional problems (an earlier version builds warning free and a later version doesn't). 

## Unit Testing and Code Coverage

There should be unit tests that cover the core functionality of the product, expected edge cases, and expected errors. 
Code coverage from these tests should aim at maintaining over 80% coverage. 

All changes should be covered by new or existing unit tests. 

In order to check that all the code you expect to be covered by testing is covered, run code coverage in Visual Studio using 'Analyze Code Coverage' under the Test menu. 

There is a configuration file in onnxruntime\VSCodeCoverage.runsettings that can be used to configure code coverage so that it reports numbers for just the onnxruntime code. Select that file in Visual Studio via the Test menu: 'Test' -> 'Test Settings' -> 'Select Test Settings File'. 

Using 'Show Code Coverage Coloring' will allow you to visually inspect which lines were hit by the tests. See <https://docs.microsoft.com/en-us/visualstudio/test/using-code-coverage-to-determine-how-much-code-is-being-tested?view=vs-2017>.
