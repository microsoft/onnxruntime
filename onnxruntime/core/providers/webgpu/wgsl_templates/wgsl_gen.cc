// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/**
 * @file wgsl_gen.cc
 * @brief WGSL shader generation for WebGPU execution provider
 *
 * This file implements both static and dynamic WGSL template generation:
 * - Static generation: Uses pre-compiled C++ template functions
 * - Dynamic generation (ORT_WGSL_TEMPLATE_DYNAMIC=1): Uses JavaScript templates executed via Duktape
 *
 * For dynamic generation, it bridges C++ objects (ShaderHelper, ShaderVariableHelper)
 * with JavaScript template functions to generate optimized WGSL shader code.
 *
 * Key features:
 * - Configurable static vs dynamic template generation
 * - Dynamic loading of templates.js at runtime (dynamic mode)
 * - JavaScript-C++ interop for parameter and variable passing
 * - Robust error handling with stack traces
 * - Memory-safe pointer management between JS and C++
 */

#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "core/providers/webgpu/wgsl_templates/wgsl_gen.h"

#include "core/providers/webgpu/shader_helper.h"
#include "core/providers/webgpu/shader_variable.h"
#include "core/providers/webgpu/webgpu_utils.h"

namespace onnxruntime {
namespace webgpu {
namespace wgsl_gen {

// ============================================================================
// Template Generation Implementation Selection
// ============================================================================
// This file supports two modes of WGSL template generation controlled by
// the ORT_WGSL_TEMPLATE_DYNAMIC macro:
// - ORT_WGSL_TEMPLATE_DYNAMIC not defined: Static generation using pre-compiled C++ functions
// - ORT_WGSL_TEMPLATE_DYNAMIC=1: Dynamic generation using JavaScript templates via Duktape
// ============================================================================

#ifndef ORT_WGSL_TEMPLATE_DYNAMIC  // Use static generator

#if defined(INCLUDED_BY_WGSL_GEN_IMPL)
#error "macro INCLUDED_BY_WGSL_GEN_IMPL should not be defined yet."
#endif

#define INCLUDED_BY_WGSL_GEN_IMPL
#include "wgsl_template_gen/index_impl.h"
#undef INCLUDED_BY_WGSL_GEN_IMPL

#else  // Use dynamic generator

#include "duktape.h"

namespace {

// ============================================================================
// Constants and Utilities
// ============================================================================

// Hidden property name used to store C++ object pointers in JavaScript functions
constexpr const char* kHelperPointerProperty =
    "\xFF"
    "helper_ptr";

// ============================================================================
// Duktape Context Management
// ============================================================================

/**
 * @brief RAII wrapper for Duktape JavaScript context with WGSL template support
 *
 * This class manages the lifecycle of a Duktape JavaScript context and loads
 * the auto-generated templates.js file containing WGSL template functions.
 */
struct DuktapeContext {
  DuktapeContext() {
    ctx = duk_create_heap_default();
    if (!ctx) {
      throw std::runtime_error("Failed to create Duktape heap");
    }
    LoadTemplatesJS();
  }
  ~DuktapeContext() {
    if (ctx) {
      duk_destroy_heap(ctx);
    }
  }

  DuktapeContext(const DuktapeContext&) = delete;
  DuktapeContext& operator=(const DuktapeContext&) = delete;

 private:
  /**
   * @brief Load and initialize templates.js in the JavaScript context
   *
   * This method:
   * 1. Loads the templates.js file using CMake-generated path
   * 2. Fixes JavaScript scoping issues for Duktape compatibility
   * 3. Evaluates the JavaScript code to register template functions
   * 4. Registers global helper functions needed by templates
   */
  void LoadTemplatesJS() {
    // Load and evaluate the templates.js file using CMake-generated absolute path
#ifndef ORT_WGSL_TEMPLATES_JS_PATH
#error "ORT_WGSL_TEMPLATES_JS_PATH must be defined by CMake"
#endif

    std::string templates_js_path = ORT_WGSL_TEMPLATES_JS_PATH;

    // Read the templates.js file
    std::ifstream file(templates_js_path);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to open templates.js file at: " + templates_js_path);
    }

    std::string js_content((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());
    file.close();

    // Evaluate the JavaScript code
    if (duk_peval_string(ctx, js_content.c_str()) != 0) {
      std::string error_msg = "Failed to evaluate templates.js: ";
      if (duk_is_string(ctx, -1)) {
        error_msg += duk_get_string(ctx, -1);
      }
      duk_pop(ctx);
      throw std::runtime_error(error_msg);
    }
    duk_pop(ctx);  // Pop the result

    // Ensure $templates is globally accessible (fallback for Duktape scoping quirks)
    duk_eval_string(ctx, "$templates");
    if (duk_is_object(ctx, -1)) {
      duk_put_global_string(ctx, "$templates");
    } else {
      duk_pop(ctx);
    }

    RegisterGlobalHelperFunctions();
  }

  /**
   * @brief Register global helper functions required by WGSL templates
   */
  void RegisterGlobalHelperFunctions() {
    // Register GetElementAt function used by templates for array indexing
    duk_push_c_function(ctx, [](duk_context* ctx) -> duk_ret_t {
      const char* var = duk_require_string(ctx, 0);
      const char* idx = duk_require_string(ctx, 1);
      int rank = duk_require_int(ctx, 2);
      bool is_f16 = duk_opt_boolean(ctx, 3, false);

      std::string result = onnxruntime::webgpu::GetElementAt(var, idx, rank, is_f16);
      duk_push_string(ctx, result.c_str());
      return 1; }, 4);
    duk_put_global_string(ctx, "GetElementAt");
  }

 public:
  duk_context* ctx = nullptr;
};

// ============================================================================
// JavaScript Method Binding Helpers
// ============================================================================

/**
 * @brief Create a JavaScript function bound to a C++ ShaderVariableHelper method
 *
 * @param ctx Duktape context
 * @param method_name Name of the JavaScript method
 * @param func C function implementing the method
 * @param nargs Number of arguments the function expects
 * @param helper Pointer to the C++ ShaderVariableHelper instance
 */
void CreateShaderVariableMethod(duk_context* ctx, const char* method_name,
                                duk_c_function func, duk_idx_t nargs,
                                const ShaderVariableHelper* helper) {
  duk_push_string(ctx, method_name);
  duk_push_c_function(ctx, func, nargs);
  // Store helper pointer for later retrieval
  duk_push_pointer(ctx, const_cast<ShaderVariableHelper*>(helper));
  duk_put_prop_string(ctx, -2, kHelperPointerProperty);
  duk_put_prop(ctx, -3);
}

/**
 * @brief Create a JavaScript function bound to a C++ ShaderHelper method
 *
 * @param ctx Duktape context
 * @param method_name Name of the JavaScript method
 * @param func C function implementing the method
 * @param nargs Number of arguments the function expects
 * @param helper Pointer to the C++ ShaderHelper instance
 */
void CreateShaderHelperMethod(duk_context* ctx, const char* method_name,
                              duk_c_function func, duk_idx_t nargs,
                              ShaderHelper* helper) {
  duk_push_string(ctx, method_name);
  duk_push_c_function(ctx, func, nargs);
  // Store helper pointer for later retrieval
  duk_push_pointer(ctx, helper);
  duk_put_prop_string(ctx, -2, kHelperPointerProperty);
  duk_put_prop(ctx, -3);
}

/**
 * @brief Safely retrieve a C++ helper pointer from a JavaScript function
 *
 * @tparam T Type of the helper pointer to retrieve
 * @param ctx Duktape context
 * @return Pointer to the C++ helper object
 * @throws JavaScript error if the pointer is invalid
 */
template <typename T>
T* GetHelperFromFunction(duk_context* ctx) {
  duk_push_current_function(ctx);
  duk_get_prop_string(ctx, -1, kHelperPointerProperty);
  T* helper = static_cast<T*>(duk_get_pointer(ctx, -1));
  if (!helper) {
    duk_error(ctx, DUK_ERR_ERROR, "Failed to get helper pointer");
  }
  duk_pop_2(ctx);  // Pop pointer and function
  return helper;
}

// ============================================================================
// JavaScript Callback Functions
// ============================================================================

}  // namespace

// ShaderVariableHelper JavaScript bindings
namespace {
/** @brief JavaScript binding for ShaderVariableHelper::OffsetToIndices */
duk_ret_t ShaderVariable_OffsetToIndices(duk_context* ctx) {
  const char* offset_expr = duk_require_string(ctx, 0);
  const ShaderVariableHelper* helper = GetHelperFromFunction<const ShaderVariableHelper>(ctx);
  std::string result = helper->OffsetToIndices(offset_expr);
  duk_push_string(ctx, result.c_str());
  return 1;
}

/** @brief JavaScript binding for ShaderVariableHelper::SetByOffset */
duk_ret_t ShaderVariable_SetByOffset(duk_context* ctx) {
  const char* offset_expr = duk_require_string(ctx, 0);
  const char* value_expr = duk_require_string(ctx, 1);
  const ShaderVariableHelper* helper = GetHelperFromFunction<const ShaderVariableHelper>(ctx);
  std::string result = helper->SetByOffset(offset_expr, value_expr);
  duk_push_string(ctx, result.c_str());
  return 1;
}

/** @brief JavaScript binding for ShaderVariableHelper::Rank */
duk_ret_t ShaderVariable_Rank(duk_context* ctx) {
  const ShaderVariableHelper* helper = GetHelperFromFunction<const ShaderVariableHelper>(ctx);
  int rank = helper->Rank();
  duk_push_int(ctx, rank);
  return 1;
}

// ShaderHelper JavaScript bindings
/** @brief JavaScript binding for ShaderHelper::AdditionalImplementation */
duk_ret_t ShaderHelper_AppendAdditionalImplementation(duk_context* ctx) {
  const char* code = duk_to_string(ctx, 0);  // Allow automatic type conversion
  ShaderHelper* helper = GetHelperFromFunction<ShaderHelper>(ctx);
  helper->AdditionalImplementation() << code;
  return 0;
}

/** @brief JavaScript binding for ShaderHelper::MainFunctionBody */
duk_ret_t ShaderHelper_AppendMainFunctionBody(duk_context* ctx) {
  const char* code = duk_to_string(ctx, 0);  // Allow automatic type conversion
  ShaderHelper* helper = GetHelperFromFunction<ShaderHelper>(ctx);
  helper->MainFunctionBody() << code;
  return 0;
}

/** @brief JavaScript binding for ShaderHelper::GuardAgainstOutOfBoundsWorkgroupSizes */
duk_ret_t ShaderHelper_GuardAgainstOutOfBoundsWorkgroupSizes(duk_context* ctx) {
  const char* size = duk_require_string(ctx, 0);
  ShaderHelper* helper = GetHelperFromFunction<ShaderHelper>(ctx);
  std::string result = helper->GuardAgainstOutOfBoundsWorkgroupSizes(size);
  duk_push_string(ctx, result.c_str());
  return 1;
}
}  // namespace

// ============================================================================
// Template Argument Constructors
// ============================================================================

TemplateArgument TemplateParam(std::string_view name, int value) {
  return TemplateArgument{.name = std::string(name), .type = TemplateArgument::Type::Param, .param_value = value};
}

TemplateArgument TemplateVariable(std::string_view name, const void* value) {
  return TemplateArgument{.name = std::string(name), .type = TemplateArgument::Type::Variable, .variable_value = value};
}

/**
 * @brief Apply a dynamic WGSL template using JavaScript execution
 *
 * This function loads a JavaScript template function from the templates.js file
 * and executes it with the provided parameters and variable helpers. The template
 * generates WGSL shader code by calling methods on the shader_helper object.
 *
 * @param shader_helper Reference to the ShaderHelper that collects generated code
 * @param template_filepath Path/key of the template function (e.g., "tensor/pad.wgsl.template")
 * @param args Template arguments including parameters and variable helpers
 * @return Status indicating success or failure with error details
 */
Status ApplyTemplateDynamic(ShaderHelper& shader_helper,
                            std::string_view template_filepath,
                            const std::initializer_list<TemplateArgument>& args) {
  static DuktapeContext duktape_context;
  duk_context* ctx = duktape_context.ctx;

  try {
    // Get the $templates object
    duk_get_global_string(ctx, "$templates");
    if (!duk_is_object(ctx, -1)) {
      duk_pop(ctx);
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "$templates object not found in JavaScript context");
    }

    // Get the template function
    duk_get_prop_string(ctx, -1, std::string(template_filepath).c_str());
    if (!duk_is_function(ctx, -1)) {
      duk_pop_2(ctx);
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Template function not found: ", template_filepath);
    }

    // Create param object
    duk_push_object(ctx);
    for (const auto& arg : args) {
      if (arg.type == TemplateArgument::Type::Param) {
        duk_push_int(ctx, arg.param_value);
        duk_put_prop_string(ctx, -2, arg.name.c_str());
      }
    }

    // Create variable object
    duk_push_object(ctx);
    for (const auto& arg : args) {
      if (arg.type == TemplateArgument::Type::Variable) {
        // Create object that wraps the ShaderVariableHelper
        duk_push_object(ctx);

        // Add methods that are called by the template
        const auto* var_helper = static_cast<const ShaderVariableHelper*>(arg.variable_value);
        if (!var_helper) {
          duk_pop_n(ctx, 3);  // Clean up stack: object, variable object, $templates
          return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Invalid variable helper for argument: ", arg.name);
        }

        CreateShaderVariableMethod(ctx, "OffsetToIndices", ShaderVariable_OffsetToIndices, 1, var_helper);
        CreateShaderVariableMethod(ctx, "SetByOffset", ShaderVariable_SetByOffset, 2, var_helper);
        CreateShaderVariableMethod(ctx, "Rank", ShaderVariable_Rank, 0, var_helper);
        duk_put_prop_string(ctx, -2, arg.name.c_str());
      }
    }

    // Create shader_helper object
    duk_push_object(ctx);

    CreateShaderHelperMethod(ctx, "appendAdditionalImplementation",
                             ShaderHelper_AppendAdditionalImplementation, 1, &shader_helper);
    CreateShaderHelperMethod(ctx, "appendMainFunctionBody",
                             ShaderHelper_AppendMainFunctionBody, 1, &shader_helper);
    CreateShaderHelperMethod(ctx, "GuardAgainstOutOfBoundsWorkgroupSizes",
                             ShaderHelper_GuardAgainstOutOfBoundsWorkgroupSizes, 1, &shader_helper);

    // Call the template function: function(param, variable, shader_helper)
    if (duk_pcall(ctx, 3) != 0) {
      std::string error_msg = "Template execution failed: ";

      // Get error message and stack trace if available
      if (duk_is_error(ctx, -1)) {
        duk_get_prop_string(ctx, -1, "message");
        if (duk_is_string(ctx, -1)) {
          error_msg += duk_get_string(ctx, -1);
        }
        duk_pop(ctx);  // Pop message

        duk_get_prop_string(ctx, -1, "stack");
        if (duk_is_string(ctx, -1)) {
          error_msg += "\nStack trace:\n";
          error_msg += duk_get_string(ctx, -1);
        }
        duk_pop(ctx);  // Pop stack
      } else if (duk_is_string(ctx, -1)) {
        error_msg += duk_get_string(ctx, -1);
      } else {
        error_msg += "Unknown JavaScript error";
      }

      duk_pop_2(ctx);  // Pop error and $templates
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, error_msg);
    }

    duk_pop_2(ctx);  // Pop result and $templates
    return Status::OK();

  } catch (const std::exception& e) {
    // Clean up stack
    duk_set_top(ctx, 0);
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Exception in template execution: ", e.what());
  }
}

#endif

}  // namespace wgsl_gen
}  // namespace webgpu
}  // namespace onnxruntime
