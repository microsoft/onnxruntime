// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Forward declare to reduce external header dependencies.
interface IMLOperatorKernel;
class MLOperatorKernelCreationContext;

// Forward declares an external creation function.
#define DML_OP_EXTERN_CREATION_FUNCTION(operatorName) extern void CALLBACK Create##operatorName(IMLOperatorKernelCreationContext* kernelInfo, IMLOperatorKernel** opKernel)
#define DML_OP_EXTERN_QUERY_FUNCTION(operatorName) extern void CALLBACK Query##operatorName(IMLOperatorSupportQueryContextPrivate* context, bool* isSupported);

// A specific opset version for registration.
// e.g. 
// DML_OP_DEFINE_CREATION_FUNCTION(RoiAlign10, VersionedKernel<DmlOperatorSlice, 10>);
template <typename BaseClass, uint32_t opsetVersion>
class VersionedKernel : public BaseClass
{
public:
    VersionedKernel(const MLOperatorKernelCreationContext& kernelInfo)
    :   BaseClass(kernelInfo, opsetVersion)
    {
    }
};

// Declares a callback creation function of the given operator class.
// This does not register it, just declares it for usage by registration later.
//
// e.g. DML_OP_DEFINE_CREATION_FUNCTION(Congrats, DmlOperatorCongratulate);
//
// Note the second parameter is the class name, but templated parameters with
// commas in them break the macro, and so they are stuffed into the VA_ARGS.
// 
#define DML_OP_DEFINE_CREATION_FUNCTION(operatorName, ...)\
extern void CALLBACK Create##operatorName(IMLOperatorKernelCreationContext* kernelInfo, IMLOperatorKernel** opKernel)\
{\
    using T = __VA_ARGS__; \
    ORT_THROW_IF_FAILED(MLOperatorKernel<T>::CreateInstance(*kernelInfo, /*out*/ opKernel));\
}