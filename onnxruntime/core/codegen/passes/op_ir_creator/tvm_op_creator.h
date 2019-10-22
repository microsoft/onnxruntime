// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/common/creator.h"
#include "core/codegen/common/dispatcher.h"
#include "core/codegen/common/registry.h"
#include "core/graph/graph.h"
#include <tvm/tvm.h>

namespace onnxruntime {
namespace tvm_codegen {

class CodeGenContext;

// OpIRCreator lowers an Ort Node to its corresponding TVM IRs
using OpIRCreator = codegen::CreatorBase<
    const tvm::Array<tvm::Tensor>&,
    const Node&,
    CodeGenContext&,
    tvm::Array<tvm::Tensor>&,
    Status>;

// OpIRDispatcher is the base dispatcher for TVM IR Builder
// It checks whether an Ort Node satisfying a criteria (in Find)
// and dispatches a corresponding OpIRCreator.
class OpIRDispatcher : public codegen::DispatcherBase<OpIRCreator*> {
 public:
  OpIRDispatcher(const std::string& name)
      : DispatcherBase(name) {}

  virtual ~OpIRDispatcher() = default;

  virtual OpIRCreator* Find(const Node&) = 0;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OpIRDispatcher);
};

// Macro returns an OpIRCreators' dispatcher's name
#define OP_IR_DISPATCHER_CLASS(OP) \
  TVM##OP##IRCreator

// Macro declares an OpIRCreators' dispatcher
#define DECLARE_OP_IR_DISPATCHER_CLASS(OP)                             \
  class OP_IR_DISPATCHER_CLASS(OP) : public OpIRDispatcher {           \
   public:                                                             \
    TVM##OP##IRCreator(const std::string& name)                        \
        : OpIRDispatcher(name) {}                                      \
    ~TVM##OP##IRCreator() = default;                                   \
    OpIRCreator* Find(const Node&) override;                           \
                                                                       \
   private:                                                            \
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(OP_IR_DISPATCHER_CLASS(OP)); \
  };

// Declare two common dispatchers for TVM Op IR builders
// One dispatcher is based on Ort OpType
DECLARE_OP_IR_DISPATCHER_CLASS(OpType)
// Another dispatcher is based Ort NodeArg name
DECLARE_OP_IR_DISPATCHER_CLASS(NodeName)

// OpIRCreator Registry is a registry holds all OpIRCreators
using OpIRRegistry = codegen::RegistryBase<OpIRCreator>;

// Macro declares an OpIRCreator
#define DECLARE_OP_IR_CREATOR_CLASS(OP, PREFIX)         \
  DECLARE_CREATOR_CLASS(OP, PREFIX##IRCreator,          \
                        const tvm::Array<tvm::Tensor>&, \
                        const Node&,                    \
                        tvm_codegen::CodeGenContext&,   \
                        tvm::Array<tvm::Tensor>&,       \
                        Status)

// Macro returns an OpIRCreator's name  with prefix
#define OP_IR_CREATOR_CLASS_EX(OP, PREFIX, ARCH) \
  CREATOR_CLASS(OP, PREFIX##ARCH##IRCreator)

// Macro declares an OpIRCreator with prefix and arch
#define DECLARE_OP_IR_CREATOR_CLASS_EX(OP, PREFIX, ARCH) \
  DECLARE_OP_IR_CREATOR_CLASS(OP, PREFIX##ARCH)

}  // namespace tvm_codegen
}  // namespace onnxruntime
