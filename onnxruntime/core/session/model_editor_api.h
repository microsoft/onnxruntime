namespace OrtModelEditorAPI {

// implementation that returns the API struct
ORT_API(const OrtModelEditorApi*, GetModelEditorApi);

// APIs to create/edit type info
ORT_API_STATUS_IMPL(CreateTensorTypeInfo, _In_ const OrtTensorTypeAndShapeInfo* tensor_info,
                    _Out_ OrtTypeInfo** type_info);
ORT_API_STATUS_IMPL(CreateSparseTensorTypeInfo, _In_ const OrtTensorTypeAndShapeInfo* tensor_info,
                    _Out_ OrtTypeInfo** type_info);
ORT_API_STATUS_IMPL(CreateMapTypeInfo, ONNXTensorElementDataType map_key_type, _In_ const OrtTypeInfo* map_value_type,
                    _Out_ OrtTypeInfo** type_info);
ORT_API_STATUS_IMPL(CreateSequenceTypeInfo, _In_ const OrtTypeInfo* sequence_type, _Out_ OrtTypeInfo** type_info);
ORT_API_STATUS_IMPL(CreateOptionalTypeInfo, _In_ const OrtTypeInfo* contained_type, _Out_ OrtTypeInfo** type_info);

ORT_API_STATUS_IMPL(CreateValueInfo, _In_ const char* name, _In_ const OrtTypeInfo* type_info,
                    _Outptr_ OrtValueInfo** value_info);

ORT_API_STATUS_IMPL(CreateNode, const char* operator_name, const char* domain_name, _In_ const char* node_name,
                    _In_reads_(input_names_len) const char* const* input_names, size_t input_names_len,
                    _In_reads_(output_names_len) const char* const* output_names, size_t output_names_len,
                    _In_reads_(attribs_len) _Inout_opt_ OrtOpAttr** attributes, _In_opt_ size_t attribs_len,
                    _Outptr_ OrtNode** node);

ORT_API_STATUS_IMPL(CreateGraph, _Outptr_ OrtGraph** graph);
ORT_API_STATUS_IMPL(SetGraphInputs, _In_ OrtGraph* graph,
                    _In_reads_(inputs_len) _In_ OrtValueInfo** inputs, _In_ size_t inputs_len);
ORT_API_STATUS_IMPL(SetGraphOutputs, _In_ OrtGraph* graph,
                    _In_reads_(outputs_len) _In_ OrtValueInfo** outputs, _In_ size_t outputs_len);
ORT_API_STATUS_IMPL(AddInitializerToGraph, _In_ OrtGraph* graph, _In_ const char* name, _Inout_ OrtValue* tensor,
                    bool data_is_external);
ORT_API_STATUS_IMPL(AddNodeToGraph, _In_ OrtGraph* graph, _Inout_ OrtNode* node);

ORT_API_STATUS_IMPL(CreateModel,
                    _In_reads_(opset_entries_len) const char* const* domain_names,
                    _In_reads_(opset_entries_len) const int* opset_versions,
                    size_t opset_entries_len,
                    _Outptr_ OrtModel** model);
ORT_API_STATUS_IMPL(AddGraphToModel, _In_ OrtModel* model, _Inout_ OrtGraph* graph);

ORT_API_STATUS_IMPL(CreateSessionFromModel, _In_ const OrtEnv* env, _In_ const OrtModel* model,
                    _In_ const OrtSessionOptions* options, _Outptr_ OrtSession** out);

//
// Model editing APIs for updating existing model by adding node/s at start or end.
//
ORT_API_STATUS_IMPL(CreateModelEditorSession, _In_ const OrtEnv* env,
                    _In_ const ORTCHAR_T* model_path,
                    _In_ const OrtSessionOptions* options,
                    _Outptr_ OrtSession** out);

ORT_API_STATUS_IMPL(CreateModelEditorSessionFromArray, _In_ const OrtEnv* env,
                    _In_ const void* model_data, size_t model_data_length,
                    _In_ const OrtSessionOptions* options,
                    _Outptr_ OrtSession** out);

ORT_API_STATUS_IMPL(SessionGetOpsetForDomain, _In_ const OrtSession* session, _In_ const char* domain,
                    _Out_ int* opset);

ORT_API_STATUS_IMPL(ApplyModelToModelEditorSession, _In_ OrtSession* session, _In_ OrtModel* model);

ORT_API_STATUS_IMPL(FinalizeModelEditorSession, _In_ OrtSession* session, _In_ const OrtSessionOptions* options,
                    _Inout_ OrtPrepackedWeightsContainer* prepacked_weights_container);

}  // namespace OrtModelEditorAPI
