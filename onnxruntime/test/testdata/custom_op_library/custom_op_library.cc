
// /*
//  * The OrtCustomOp structure defines a custom op's schema and its kernel callbacks. The callbacks are filled in by
//  * the implementor of the custom op.
// */
// struct OrtCustomOp {
//   uint32_t version;  // Initialize to ORT_API_VERSION

//   // This callback creates the kernel, which is a user defined parameter that is passed to the Kernel* callbacks below.
//   void*(ORT_API_CALL* CreateKernel)(_In_ struct OrtCustomOp* op, _In_ const OrtApi* api, _In_ const OrtKernelInfo* info);

//   // Returns the name of the op
//   const char*(ORT_API_CALL* GetName)(_In_ struct OrtCustomOp* op);

//   // Returns the type of the execution provider, return nullptr to use CPU execution provider
//   const char*(ORT_API_CALL* GetExecutionProviderType)(_In_ struct OrtCustomOp* op);

//   // Returns the count and types of the input & output tensors
//   ONNXTensorElementDataType(ORT_API_CALL* GetInputType)(_In_ struct OrtCustomOp* op, _In_ size_t index);
//   size_t(ORT_API_CALL* GetInputTypeCount)(_In_ struct OrtCustomOp* op);
//   ONNXTensorElementDataType(ORT_API_CALL* GetOutputType)(_In_ struct OrtCustomOp* op, _In_ size_t index);
//   size_t(ORT_API_CALL* GetOutputTypeCount)(_In_ struct OrtCustomOp* op);

//   // Op kernel callbacks
//   void(ORT_API_CALL* KernelCompute)(_In_ void* op_kernel, _In_ OrtKernelContext* context);
//   void(ORT_API_CALL* KernelDestroy)(_In_ void* op_kernel);
// };

// /*
//  * END EXPERIMENTAL
// */

#include "custom_op_library.h"
#include <vector>

static const char* c_OpDomain = "test.customop";
static const char* c_OpOneName = "CustomOpOne";
static const char* c_OpTwoName = "CustomOpTwo";
static const char* c_EPType = nullptr; // CPU
ONNXTensorElementDataType c_dataElemTypeOne = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
ONNXTensorElementDataType c_dataElemTypeTwo = ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;

#define ORT_API_RETURN_IF_ERROR(expr) \
  do {                                \
    auto _status = (expr);            \
    if (_status) return _status;      \
  } while (0)


// This callback creates the kernel, which is a user defined parameter that is passed to the Kernel* callbacks below.
void* ORT_API_CALL CreateKernel_One(_In_ struct OrtCustomOp* op, _In_ const OrtApi* api, _In_ const OrtKernelInfo* info)
{
    return nullptr;
}

// Returns the name of the op
const char* ORT_API_CALL GetName_One(_In_ struct OrtCustomOp* op)
{
    return c_OpOneName;   
}

// Returns the type of the execution provider, return nullptr to use CPU execution provider
const char* ORT_API_CALL GetExecutionProviderType_One(_In_ struct OrtCustomOp* op)
{
    return c_EPType;
}

// Returns the count and types of the input & output tensors
ONNXTensorElementDataType ORT_API_CALL GetInputType_One(_In_ struct OrtCustomOp* op, _In_ size_t index)
{
    return c_dataElemTypeOne;
}
  
size_t ORT_API_CALL GetInputTypeCount_One(_In_ struct OrtCustomOp* op)
{
    return 2;
}


ONNXTensorElementDataType ORT_API_CALL GetOutputType_One(_In_ struct OrtCustomOp* op, _In_ size_t index)
{
    return c_dataElemTypeOne;
}

size_t ORT_API_CALL GetOutputTypeCount_One(_In_ struct OrtCustomOp* op)
{
    return 1;
}

// Op kernel callbacks
void ORT_API_CALL KernelCompute_One(_In_ void* op_kernel, _In_ OrtKernelContext* context)
{
    


}

void ORT_API_CALL KernelDestroy_One(_In_ void* op_kernel)
{
    // nothing to do
}


// This callback creates the kernel, which is a user defined parameter that is passed to the Kernel* callbacks below.
void* ORT_API_CALL CreateKernel_Two(_In_ struct OrtCustomOp* op, _In_ const OrtApi* api, _In_ const OrtKernelInfo* info)
{
    return nullptr;
}

// Returns the name of the op
const char* ORT_API_CALL GetName_Two(_In_ struct OrtCustomOp* op)
{
    return c_OpTwoName;
}

// Returns the type of the execution provider, return nullptr to use CPU execution provider
const char* ORT_API_CALL GetExecutionProviderType_Two(_In_ struct OrtCustomOp* op)
{
    return c_EPType;
}

// Returns the count and types of the input & output tensors
ONNXTensorElementDataType ORT_API_CALL GetInputType_Two(_In_ struct OrtCustomOp* op, _In_ size_t index)
{
    return c_dataElemTypeTwo;
}
  
size_t ORT_API_CALL GetInputTypeCount_Two(_In_ struct OrtCustomOp* op)
{
    return 2;
}


ONNXTensorElementDataType ORT_API_CALL GetOutputType_Two(_In_ struct OrtCustomOp* op, _In_ size_t index)
{
    return c_dataElemTypeTwo;
}

size_t ORT_API_CALL GetOutputTypeCount_Two(_In_ struct OrtCustomOp* op)
{
    return 1;
}

// Op kernel callbacks
void ORT_API_CALL KernelCompute_Two(_In_ void* op_kernel, _In_ OrtKernelContext* context)
{
    auto api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    size_t inputCount, outputCount;

    if (auto status = api->KernelContext_GetInputCount(context, &inputCount))
    {
        return; // throw exception? where to log the error?
    }

    if (auto status = api->KernelContext_GetOutputCount(context, &outputCount))
    {
        return;
    }

    std::vector<OrtValue*> inputs(inputCount);
    std::vector<OrtValue*> outputs(outputCount);

    for (int i = 0; i < inputCount; i++)
    {
        if (auto status = api->KernelContext_GetInput(context, i, (const OrtValue**)inputs.data()+i))
        {
            return;
        }
    }

    // for (int i = 0; i < outputCount; i++)
    // {
    //     if (auto status = api->KernelContext_GetOutput(context, i, (const OrtValue**)outputs.data()+i))
    //     {
    //         return;
    //     }
    // }



}

void ORT_API_CALL KernelDestroy_Two(_In_ void* op_kernel)
{

}

OrtCustomOp c_CustomOpOne =
{
ORT_API_VERSION, 
CreateKernel_One, 
GetName_One,
GetExecutionProviderType_One,
GetInputType_One,
GetInputTypeCount_One,
GetOutputType_One,
GetOutputTypeCount_One,
KernelCompute_One,
KernelDestroy_One
},
c_CustomOpTwo = 
{
ORT_API_VERSION, 
CreateKernel_Two, 
GetName_Two,
GetExecutionProviderType_Two,
GetInputType_Two,
GetInputTypeCount_Two,
GetOutputType_Two,
GetOutputTypeCount_Two,
KernelCompute_Two,
KernelDestroy_Two
};



// /*
//   Create a custom op domain. After all sessions using it are released, call OrtReleaseCustomOpDomain
//   */
//   OrtStatus*(ORT_API_CALL* CreateCustomOpDomain)(_In_ const char* domain, _Outptr_ OrtCustomOpDomain** out)NO_EXCEPTION;

//   /*
// 	 * Add custom ops to the OrtCustomOpDomain
// 	 *  Note: The OrtCustomOp* pointer must remain valid until the OrtCustomOpDomain using it is released
// 	*/
//   OrtStatus*(ORT_API_CALL* CustomOpDomain_Add)(_Inout_ OrtCustomOpDomain* custom_op_domain, _In_ OrtCustomOp* op)NO_EXCEPTION;

//   /*
// 	 * Add a custom op domain to the OrtSessionOptions
// 	 *  Note: The OrtCustomOpDomain* must not be deleted until the sessions using it are released
// 	*/
//   OrtStatus*(ORT_API_CALL* AddCustomOpDomain)(_Inout_ OrtSessionOptions* options, _In_ OrtCustomOpDomain* custom_op_domain)NO_EXCEPTION;

//   /*
// 	 * Loads a DLL named 'library_path' and looks for this entry point:
// 	 *		OrtStatus* RegisterCustomOps(OrtSessionOptions * options, const OrtApiBase* api);
// 	 * It then passes in the provided session options to this function along with the api base.
// 	 * The handle to the loaded library is returned in library_handle. It can be freed by the caller after all sessions using the passed in
// 	 * session options are destroyed, or if an error occurs and it is non null.
//   */
//   OrtStatus*(ORT_API_CALL* RegisterCustomOpsLibrary)(_Inout_ OrtSessionOptions* options, _In_ const char* library_path, void** library_handle)NO_EXCEPTION;


OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions * options, const OrtApiBase* api)
{
    OrtCustomOpDomain* domain = nullptr;
    if (auto status = api->GetApi(ORT_API_VERSION)->CreateCustomOpDomain(c_OpDomain, &domain))
    {
        return status;
    }

    if (auto status = api->GetApi(ORT_API_VERSION)->CustomOpDomain_Add(domain, &c_CustomOpOne))
    {
        return status;
    }

    if (auto status = api->GetApi(ORT_API_VERSION)->CustomOpDomain_Add(domain, &c_CustomOpTwo))
    {
        return status;
    }

    return api->GetApi(ORT_API_VERSION)->AddCustomOpDomain(options, domain);
}

