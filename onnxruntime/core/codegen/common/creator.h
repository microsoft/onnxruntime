// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/common/dispatcher.h"

// TODO rename this file to creator_base
namespace onnxruntime {
namespace codegen {

// It is a base class for TVM Op IR builder, weight layout builder, TVM scheduler
// CreatorBase is a template class of compiler pass
// for 1) TVM IR builder
//     2) Weight layout transformer
//     3) TVM Scheduler, etc.
// CreatorBase is similor to OpXXCreate in llvm IR builder

template <typename INPUT_TYPE,
          typename NODE_TYPE,
          typename CONTEXT_TYPE,
          typename OUTPUT_TYPE,
          typename RETURN_TYPE>
class CreatorBase {
 public:
  CreatorBase(const std::string& name)
      : name_(name) {}

  virtual ~CreatorBase() = default;

  virtual RETURN_TYPE Evaluate(INPUT_TYPE,
                               NODE_TYPE,
                               CONTEXT_TYPE,
                               OUTPUT_TYPE) = 0;

  const std::string& Name() const {
    return name_;
  }

 protected:
  std::string name_;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CreatorBase);
};

// macro to stringize
#define STRINGIZE_NX(OP) #OP
#define STRINGIZE(OP) STRINGIZE_NX(OP)

// macro returns class name
#define CREATOR_CLASS(OP, POSTFIX) \
  OP##POSTFIX

// macro returns class name as string
#define CREATOR_STRING(OP, POSTFIX) \
  STRINGIZE(CREATOR_CLASS(OP, POSTFIX))

// macro returns class constructor name
#define CREATOR_CLASS_FUNC(OP, POSTFIX) \
  OP##POSTFIX()

// macro declares a creator class inheriting the template class CreatorBase
// with corresponding template parameters
#define DECLARE_CREATOR_CLASS(OP, POSTFIX, INPUT, NODE, CONTEXT, OUTPUT, RETURN)                                      \
  class CREATOR_CLASS(OP, POSTFIX) : public onnxruntime::codegen::CreatorBase<INPUT, NODE, CONTEXT, OUTPUT, RETURN> { \
   public:                                                                                                            \
    CREATOR_CLASS_FUNC(OP, POSTFIX) : CreatorBase(CREATOR_STRING(OP, POSTFIX)) {}                                     \
    RETURN Evaluate(INPUT,                                                                                            \
                    NODE,                                                                                             \
                    CONTEXT,                                                                                          \
                    OUTPUT) override;                                                                                 \
                                                                                                                      \
   private:                                                                                                           \
    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CREATOR_CLASS(OP, POSTFIX));                                                \
  };

}  // namespace codegen
}  // namespace onnxruntime
