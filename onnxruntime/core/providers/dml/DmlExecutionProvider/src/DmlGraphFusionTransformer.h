// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once


#include "core/optimizer/graph_transformer.h"
#include "core/framework/execution_providers.h"

namespace Dml
{
	class ExecutionProviderImpl;

	class DmlGraphFusionTransformer : public onnxruntime::GraphTransformer
	{
	public:
		DmlGraphFusionTransformer(
			const std::string& name,
			const onnxruntime::IExecutionProvider* provider
		);

	private:
		onnxruntime::common::Status ApplyImpl(onnxruntime::Graph& graph, 
											  bool& modified, 
											  int graph_level, 
											  const onnxruntime::logging::Logger& logger) const final;
	private:
		const ExecutionProviderImpl* m_providerImpl = nullptr;
	};
}
