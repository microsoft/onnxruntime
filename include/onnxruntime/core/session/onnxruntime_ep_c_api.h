// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Do not include this file directly. Please include "onnxruntime_c_api.h" instead.

#ifdef __cplusplus
extern "C" {
#endif

ORT_RUNTIME_CLASS(Ep);
ORT_RUNTIME_CLASS(EpFactory);
ORT_RUNTIME_CLASS(EpGraphSupportInfo);
ORT_RUNTIME_CLASS(NodeComputeContext);

/**
 * \brief The OrtNodeComputeInfo struct provides functions that an OrtEp implements to specify the compute
 * function for a compiled OrtGraph instance.
 * \since Version 1.23.
 */
struct OrtNodeComputeInfo {
  /** \brief The ONNX Runtime version the OrtNodeComputeInfo was compiled with.
   *
   * Implementation should set to ORT_API_VERSION.
   * ORT will use this to ensure it does not call functions that were not available when the library was compiled.
   *
   * \since Version 1.23.
   */
  uint32_t ort_version_supported;

  /** \brief Creates an opaque compute state object that is then passed to the Compute() function during inference.
   * \param[in] this_ptr The OrtNodeComputeInfo instance.
   * \param[in] compute_context OrtNodeComputeContext instance that contains compiled/fused node's name and host
   *                            memory allocation functions. Can optionally be used to build the compute state.
   * \param[out] compute_state Output parameter that is assigned the opaque computation state. ONNX Runtime calls
   *                           ReleaseState() (after calling Compute()) to allow the implementer to release the
   *                           compute state.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   * \since Version 1.23.
   */
  OrtStatus*(ORT_API_CALL* CreateState)(_In_ OrtNodeComputeInfo* this_ptr,
                                        _In_ OrtNodeComputeContext* compute_context,
                                        _Outptr_ void** compute_state);

  /** \brief Computation function called to execute the fused node compiled by an OrtEp instance.
   * \param[in] this_ptr The OrtNodeComputeInfo instance.
   * \param[in] compute_state The opaque computation state returned by CreateState().
   * \param[in] kernel_context The OrtKernelContext instance used to access inputs/outputs.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   * \since Version 1.23.
   */
  OrtStatus*(ORT_API_CALL* Compute)(_In_ OrtNodeComputeInfo* this_ptr, _In_ void* compute_state,
                                    _In_ OrtKernelContext* kernel_context);

  /** \brief Releases the compute state returned by CreateState().
   * \param[in] this_ptr The OrtNodeComputeInfo instance.
   * \param[inout] compute_state The opaque compute state returned by CreateState().
   *
   * \since Version 1.23.
   */
  void(ORT_API_CALL* ReleaseState)(_In_ OrtNodeComputeInfo* this_ptr, _Frees_ptr_opt_ void* compute_state);
};

struct OrtEpApi {
  /** \brief Create an OrtEpDevice for the EP and an OrtHardwareDevice.
   * \param[in] ep_factory Execution provider factory that is creating the instance.
   * \param[in] hardware_device Hardware device that the EP can utilize.
   * \param[in] ep_metadata Optional OrtKeyValuePairs instance for execution provider metadata that may be used
   *                        during execution provider selection and passed to CreateEp.
   *                        ep_device will copy this instance and the user should call ReleaseKeyValuePairs.
   * \param[in] ep_options  Optional OrtKeyValuePairs instance for execution provider options that will be added
   *                        to the Session configuration options if the execution provider is selected.
   *                        ep_device will copy this instance and the user should call ReleaseKeyValuePairs.
   * \param ep_device OrtExecutionDevice that is created.
   *
   * \since Version 1.22.
   */
  ORT_API2_STATUS(CreateEpDevice, _In_ OrtEpFactory* ep_factory,
                  _In_ const OrtHardwareDevice* hardware_device,
                  _In_opt_ const OrtKeyValuePairs* ep_metadata,
                  _In_opt_ const OrtKeyValuePairs* ep_options,
                  _Out_ OrtEpDevice** ep_device);

  ORT_CLASS_RELEASE(EpDevice);

  /** \brief Specify nodes that are supported by an OrtEp and should be fused into one node.
   *
   * IMPORTANT: This is not the final version of this API function. This is currently experimental but will
   * be stabilized by the ONNX Runtime 1.23 release.
   *
   * Because the nodes will be fused into one "fused node", there must not exist an unsupported node in
   * a path between two of the provided nodes. Otherwise, the graph will become invalid.
   *
   * This function can be called multiple times. A subsequent call to this function will force the next set of
   * nodes to be fused into a different node.
   *
   * \param[in] graph_support_info OrtEpGraphSupportInfo instance to which to add the supported nodes.
   * \param[in] nodes Array of nodes supported by the EP that should be fused/compiled.
   * \param[in] num_nodes The number of supported nodes.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   * \since Version 1.23.
   */
  ORT_API2_STATUS(EpGraphSupportInfo_AddNodesToFuse, _In_ OrtEpGraphSupportInfo* graph_support_info,
                  _In_reads_(num_nodes) const OrtNode* const* nodes, _In_ size_t num_nodes
                  /*, OrtFusedNodeSchema* optional_fused_node_schema, OrtNodesToOptimizeInfo* nodes_to_opt*/);

  /** \brief Specify a node that is supported by an OrtEp and should be run with a registered EP kernel.
   *
   * \param[in] graph_support_info OrtEpGraphSupportInfo instance to which to add the supported node.
   * \param[in] node The supported OrtNode instance.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   * \since Version 1.23.
   */
  ORT_API2_STATUS(EpGraphSupportInfo_AddSingleNode, _In_ OrtEpGraphSupportInfo* graph_support_info,
                  _In_ const OrtNode* node);

  /** \brief Query a OrtNodeComputeContext for the name of the node that encapsulates the compiled/fused node.
   *
   * Used in OrtNodeComputeInfo::CreateComputeState().
   *
   * \param[in] context The OrtNodeComputeContext instance to query.
   * \return The node's name.
   *
   * \note Returned string is owned by ORT and valid only while OrtNodeComputeInfo::CreateComputeState() is called.
   *
   * \since Version 1.23.
   */
  ORT_API_T(const char*, NodeComputeContext_NodeName, _In_ const OrtNodeComputeContext* context);
};

/**
 * \brief The data layout type that is preferred by an EP.
 * \since Version 1.23.
 */
typedef enum OrtEpDataLayout {
  OrtEpDataLayout_NCHW = 0,
  OrtEpDataLayout_NHWC,
} OrtEpDataLayout;

/**
 * \brief The OrtEp struct provides functions to implement for an execution provider.
 * \since Version 1.22.
 */
struct OrtEp {
  /** \brief The ONNX Runtime version the execution provider was compiled with.
   *
   * Implementation should set to ORT_API_VERSION.
   * ORT will use this to ensure it does not call functions that were not available when the library was compiled.
   *
   * \since Version 1.22.
   */
  uint32_t ort_version_supported;

  /** \brief Get the execution provider name.
   *
   * \param[in] this_ptr The OrtEp instance.
   * \return The execution provider name.
   *
   * \note Returned string is owned by ORT and valid until UnregisterExecutionProviderLibrary is called.
   *
   * \since Version 1.22.
   */
  const char*(ORT_API_CALL* GetName)(_In_ const OrtEp* this_ptr);

  /** \brief Get information about the nodes supported by the OrtEp instance.
   *
   * IMPORTANT: This is not the final version of this API function. This is currently experimental but will
   * be stabilized by the ONNX Runtime 1.23 release.
   *
   * \param[in] this_ptr The OrtEp instance.
   * \param[in] graph The OrtGraph instance for which to populate node support. The OrtGraph could be a nested subgraph
   *                  contained by a node (e.g., an If or Loop node). ONNX Runtime calls this function separately
   *                  for each nested subgraph.
   * \param[inout] graph_support_info OrtEpGraphSupportInfo instance that the implementer must fill out in order to
   *                                  specify the supported nodes.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   * \since Version 1.23.
   */
  OrtStatus*(ORT_API_CALL* GetCapability)(_In_ OrtEp* this_ptr, _In_ const OrtGraph* graph,
                                          _Inout_ OrtEpGraphSupportInfo* graph_support_info);

  /** \brief Compile OrtGraph instances assigned to the OrtEp. Implementer must set a OrtNodeComputeInfo instance
   * for each OrtGraph in order to define its computation function.
   *
   * If the session is configured to generate a pre-compiled model, the execution provider must return EPContext nodes,
   * as OrtNode instances, that ONNX Runtime uses to create a pre-compiled model, known as an "EPContext model".
   * An EPContext model contains EPContext nodes. Each EPContext node encapsulates the pre-compiled binary data for a
   * OrtGraph compiled for a specific execution provider. For more details about the EPContext design, refer to:
   *  \htmlonly
   *  <a href="https://onnxruntime.ai/docs/execution-providers/EP-Context-Design.html">EPContext design document.</a>
   *  \endhtmlonly
   *
   * \param[in] this_ptr The OrtEp instance.
   * \param[in] graphs Array of `count` OrtGraph instances to compile. Each graph contains only the nodes for
   *                   which the execution provider indicated support. Nested subgraphs contained by a
   *                   node, such as an If or Loop, have separate OrtGraph instances.
   * \param[in] fused_nodes Array of `count` fused nodes that will replace the compiled graphs.
   *                        Each fused node is an OrtNode initialized with the intended fused node name and
   *                        input/output information.
   * \param[in] count The number of OrtGraph instances to compile.
   * \param[out] node_compute_infos Array of `count` OrtNodeComputeInfo instances that define each OrtGraph instance's
   *                                computation function. The implementer allocates the OrtNodeComputeInfo instances.
   *                                ORT calls ReleaseNodeComputeInfos() to release multiple instances in a batch.
   * \param[out] ep_context_nodes Output array of `count` OrtNode instances, each representing an EPContext
   *                              node for a compiled OrtGraph. The execution provider must use
   *                              OrtModelEditorApi::CreateNode to create the OrtNode instances. ONNX Runtime takes
   *                              ownership of the OrtNode instances, so the execution provider must NOT call
   *                              OrtApi::ReleaseNode. Should be ignored if the session is not configured to generate an
   *                              EPContext model.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   * \note Do NOT cache the provided OrtGraph instances in any of the OrtNodeComputeInfo functions because the
   *       graphs are only valid for the duration of the call to Compile. Any graph/node/input/output
   *       names that are needed by the OrtNodeComputeInfo functions must be copied and stored by the OrtEp.
   *
   * \since Version 1.23.
   */
  OrtStatus*(ORT_API_CALL* Compile)(_In_ OrtEp* this_ptr, _In_ const OrtGraph** graphs,
                                    _In_ const OrtNode** fused_nodes, _In_ size_t count,
                                    _Out_writes_all_(count) OrtNodeComputeInfo** node_compute_infos,
                                    _Out_writes_(count) OrtNode** ep_context_nodes);

  /** \brief Release OrtNodeComputeInfo instances.
   *
   * \param[in] this_ptr The OrtEp instance.
   * \param[inout] node_compute_infos The OrtNodeComputeInfo instances to release.
   * \param[in] num_node_compute_infos The number of OrtNodeComputeInfo instances.
   *
   * \since Version 1.23.
   */
  void(ORT_API_CALL* ReleaseNodeComputeInfos)(_In_ OrtEp* this_ptr,
                                              OrtNodeComputeInfo** node_compute_infos,
                                              _In_ size_t num_node_compute_infos);

  /** \brief Get the EP's preferred data layout.
   *
   * \note Implementation of this function is optional.
   *       If not implemented, ORT will assume that this EP prefers the data layout `OrtEpDataLayout::NCHW`.
   *
   * \param[in] this_ptr The OrtEp instance.
   * \param[out] preferred_data_layout The EP's preferred data layout.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   * \since Version 1.23.
   */
  OrtStatus*(ORT_API_CALL* GetPreferredDataLayout)(_In_ OrtEp* this_ptr,
                                                   _Out_ OrtEpDataLayout* preferred_data_layout);

  /** \brief Determine whether a node with `domain` and `op_type` requires its data layout to be converted to NHWC.
   *         If the EP prefers NHWC data layout (see `GetPreferredDataLayout()`), this function will be called during
   *         layout transformation.
   *
   * \note Implementation of this function is optional.
   *       If an EP prefers NHWC data layout, it may implement this to customize the specific NHWC op preferences at a
   *       finer granularity.
   *
   * \param[in] this_ptr The OrtEp instance.
   * \param[in] node_domain The node's op domain. An empty string means the ONNX domain.
   * \param[in] node_op_type The node's op type.
   * \param[out] should_convert Indicates whether the node's layout should be converted to NHWC.
   *                            If greater than 0, convert.
   *                            If 0, don't convert.
   *                            Otherwise, if less than 0, leave the decision to ORT.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   * \since Version 1.23.
   */
  OrtStatus*(ORT_API_CALL* ShouldConvertNodeLayoutToNhwc)(_In_ OrtEp* this_ptr,
                                                          _In_z_ const char* node_domain,
                                                          _In_z_ const char* node_op_type,
                                                          _Outptr_ int* should_convert);

  /** \brief Set dynamic options on this EP.
   *
   * Dynamic options can be set by the user at any time after session creation with `OrtApi::SetEpDynamicOptions()`.
   *
   * \param[in] this_ptr The OrtEp instance.
   * \param[in] option_keys The dynamic option keys.
   * \param[in] option_values The dynamic option values.
   * \param[in] num_options The number of dynamic options.
   *
   * \note Implementation of this function is optional.
   *       An EP should only implement this if it needs to handle any dynamic options.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   * \since Version 1.23.
   */
  OrtStatus*(ORT_API_CALL* SetDynamicOptions)(_In_ OrtEp* this_ptr,
                                              _In_reads_(num_options) const char* const* option_keys,
                                              _In_reads_(num_options) const char* const* option_values,
                                              _In_ size_t num_options);

  /** \brief Called by ORT to notify the EP of the start of a run.
   *
   * \param[in] this_ptr The OrtEp instance.
   * \param[in] run_options The run options for this run.
   *
   * \note Implementation of this function is optional.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   * \since Version 1.23.
   */
  OrtStatus*(ORT_API_CALL* OnRunStart)(_In_ OrtEp* this_ptr,
                                       _In_ const OrtRunOptions* run_options);

  /** \brief Called by ORT to notify the EP of the end of a run.
   *
   * \param[in] this_ptr The OrtEp instance.
   * \param[in] run_options The run options for this run.
   * \param[in] sync_stream Whether any associated stream should be synchronized during this call.
   *                        Only applicable if there is such a stream.
   *
   * \note Implementation of this function is optional.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   * \since Version 1.23.
   */
  OrtStatus*(ORT_API_CALL* OnRunEnd)(_In_ OrtEp* this_ptr,
                                     _In_ const OrtRunOptions* run_options,
                                     _In_ bool sync_stream);
};

/** \brief The function signature that ORT will call to create OrtEpFactory instances.
 *
 * This must be available in a function called 'CreateEpFactories' in the execution provider library.
 *
 * \param[in] registered_name The name the execution library is registered with by RegisterExecutionProviderLibrary
 * \param[in] ort_api_base The OrtApiBase instance that is used by the factory to get the OrtApi instance for the
 *                         version of ORT that the library was compiled against.
 * \param[in,out] factories The implementation should create and add OrtEpFactory instances to this
 *                          pre-allocated array.
 *                          i.e. usage is `factories[0] = new MyEpFactory();`
 * \param[in] max_factories The maximum number of OrtEpFactory instances that can be added to `factories`.
 *                          Current default is to allow 4 factories. This can be increased in the future if needed.
 * \param[out] num_factories The number of OrtEpFactory instances created by the factory and added to `factories`.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 *
 * \since Version 1.22.
 */
typedef OrtStatus* (*CreateEpApiFactoriesFn)(_In_ const char* registered_name, _In_ const OrtApiBase* ort_api_base,
                                             _Inout_ OrtEpFactory** factories, _In_ size_t max_factories,
                                             _Out_ size_t* num_factories);

/** \brief The function signature that ORT will call to release an OrtEpFactory instance.
 *
 * This must be available in a function called 'ReleaseEpFactory' in the execution provider library.
 *
 * \param[in] factory The OrtEpFactory instance to release.
 *
 * \snippet{doc} snippets.dox OrtStatus Return Value
 *
 * \since Version 1.22.
 */
typedef OrtStatus* (*ReleaseEpApiFactoryFn)(_In_ OrtEpFactory* factory);

/**
 * \brief The OrtEpFactory provides functions to create and manage execution providers.
 * \since Version 1.22.
 */
struct OrtEpFactory {
  /** \brief The ONNX Runtime version the execution provider was compiled with.
   *
   * Implementation should set to ORT_API_VERSION.
   * ORT will use this to ensure it does not call functions that were not available when the library was compiled.
   *
   * \since Version 1.22.
   */
  uint32_t ort_version_supported;

  /** \brief Get the name of the execution provider that the factory creates.
   *
   * \param[in] this_ptr The OrtEpFactory instance.
   * \return The name of the execution provider the factory creates.
   *
   * \since Version 1.22.
   */
  const char*(ORT_API_CALL* GetName)(const OrtEpFactory* this_ptr);

  /** \brief Get the name of vendor who owns the execution provider that the factory creates.
   *
   * \param[in] this_ptr The OrtEpFactory instance.
   * \return vendor The vendor name of the execution provider the factory creates.
   *
   * \since Version 1.22.
   */
  const char*(ORT_API_CALL* GetVendor)(const OrtEpFactory* this_ptr);  // return EP vendor

  /** \brief Get information from the execution provider about OrtHardwareDevice support.
   *
   * \param[in] this_ptr The OrtEpFactory instance.
   *                     Non-const as the factory is passed through to the CreateEp call via the OrtEpDevice.
   * \param[in] devices The OrtHardwareDevice instances that are available.
   * \param[in] num_devices The number of OrtHardwareDevice instances.
   * \param[out] ep_devices OrtEpDevice instances for each OrtHardwareDevice that the EP can use.
   *                        The implementation should call OrtEpApi::CreateEpDevice to create, and add the OrtEpDevice
   *                        instances to this pre-allocated array. ORT will take ownership of the values returned.
   *                        i.e. usage is `ep_devices[0] = <ptr to OrtEpDevice created with OrtEpApi::CreateEpDevice>;`
   * \param[in] max_ep_devices The maximum number of OrtEpDevices that can be added to ep_devices.
   *                           Current default is 8. This can be increased if needed.
   * \param[out] num_ep_devices The number of EP devices added to ep_devices.
   * \return true if the factory can create an execution provider that uses `device`.
   *
   * \since Version 1.22.
   */
  OrtStatus*(ORT_API_CALL* GetSupportedDevices)(_In_ OrtEpFactory* this_ptr,
                                                _In_reads_(num_devices) const OrtHardwareDevice* const* devices,
                                                _In_ size_t num_devices,
                                                _Inout_ OrtEpDevice** ep_devices,
                                                _In_ size_t max_ep_devices,
                                                _Out_ size_t* num_ep_devices);

  /** \brief Function to create an OrtEp instance for use in a Session.
   *
   *  ORT will call ReleaseEp to release the instance when it is no longer needed.
   *
   * \param[in] this_ptr The OrtEpFactory instance.
   * \param[in] devices The OrtHardwareDevice instances that the execution provider was selected to use.
   *                    May be a subset of the OrtHardwareDevice instances that the execution provider's factory
   *                    set as supported in the call to OrtEpFactory::GetSupportedDevices.
   * \param[in] ep_metadata_pairs Execution provider metadata that was provided to OrtEpApi::CreateEpDevice, for each
   *                              device.
   * \param[in] num_devices The number of devices the execution provider was selected for.
   * \param[in] session_options The OrtSessionOptions instance that contains the configuration options for the
   *                            session. This will include ep_options from GetSupportedDevices as well as any
   *                            user provided overrides.
   *                            Execution provider options will have been added with a prefix of 'ep.[ep name].'.
   *                            The OrtSessionOptions instance will NOT be valid after this call and should not be
   *                            stored for later use.
   * \param[in] logger The OrtLogger instance for the session that the execution provider should use for logging.
   * \param[out] ep The OrtEp instance created by the factory.
   *
   * \snippet{doc} snippets.dox OrtStatus Return Value
   *
   * \since Version 1.22.
   */
  OrtStatus*(ORT_API_CALL* CreateEp)(_In_ OrtEpFactory* this_ptr,
                                     _In_reads_(num_devices) const OrtHardwareDevice* const* devices,
                                     _In_reads_(num_devices) const OrtKeyValuePairs* const* ep_metadata_pairs,
                                     _In_ size_t num_devices,
                                     _In_ const OrtSessionOptions* session_options,
                                     _In_ const OrtLogger* logger, _Outptr_ OrtEp** ep);

  /** \brief Release the OrtEp instance.
   *
   * \param[in] this_ptr The OrtEpFactory instance.
   * \param[in] ep The OrtEp instance to release.
   *
   * \since Version 1.22.
   */
  void(ORT_API_CALL* ReleaseEp)(OrtEpFactory* this_ptr, struct OrtEp* ep);
};

#ifdef __cplusplus
}
#endif
