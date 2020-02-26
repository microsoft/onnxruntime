// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
//TODO(): delete this file from public interface
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#else
#pragma warning(push)
#pragma warning(disable : 4018) /*'expression' : signed/unsigned mismatch */
#pragma warning(disable : 4065) /*switch statement contains 'default' but no 'case' labels*/
#pragma warning(disable : 4100)
#pragma warning(disable : 4146) /*unary minus operator applied to unsigned type, result still unsigned*/
#pragma warning(disable : 4127)
#pragma warning(disable : 4244) /*'conversion' conversion from 'type1' to 'type2', possible loss of data*/
#pragma warning(disable : 4251) /*'identifier' : class 'type' needs to have dll-interface to be used by clients of class 'type2'*/
#pragma warning(disable : 4267) /*'var' : conversion from 'size_t' to 'type', possible loss of data*/
#pragma warning(disable : 4305) /*'identifier' : truncation from 'type1' to 'type2'*/
#pragma warning(disable : 4307) /*'operator' : integral constant overflow*/
#pragma warning(disable : 4309) /*'conversion' : truncation of constant value*/
#pragma warning(disable : 4334) /*'operator' : result of 32-bit shift implicitly converted to 64 bits (was 64-bit shift intended?)*/
#pragma warning(disable : 4355) /*'this' : used in base member initializer list*/
#pragma warning(disable : 4506) /*no definition for inline function 'function'*/
#pragma warning(disable : 4800) /*'type' : forcing value to bool 'true' or 'false' (performance warning)*/
#pragma warning(disable : 4996) /*The compiler encountered a deprecated declaration.*/
#endif
#include "onnx/defs/schema.h"
#include "onnx/onnx_pb.h"
#ifdef __GNUC__
#pragma GCC diagnostic pop
#else
#pragma warning(pop)
#endif
