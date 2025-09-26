import { json } from '@sveltejs/kit';
import { dev } from '$app/environment';
import type { RequestHandler } from './$types';

// Azure AI Foundry API endpoint
const AZURE_AI_FOUNDRY_API_URL = 'https://ai.azure.com/api/eastus/ux/v1.0/entities/crossRegion';

interface ModelEntity {
	name?: string;
	displayName?: string;
	entityId?: string;
	[key: string]: unknown;
}

export const POST: RequestHandler = async ({ request, fetch }) => {
	try {
		const requestBody = await request.json();
		
		try {
			// Forward the request to the Azure AI Foundry API
			const response = await fetch(AZURE_AI_FOUNDRY_API_URL, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'Accept': 'application/json',
					'User-Agent': 'AzureAiStudio'
				},
				body: JSON.stringify(requestBody)
			});

			if (!response.ok) {
				const errorText = await response.text();
				
				// For deployment, return a proper error
				if (!dev) {
					return json(
						{ error: `Azure AI Foundry API error: ${response.status} - ${errorText}` },
						{ status: response.status }
					);
				} else {
					// In development, return empty data
					return json(getEmptyData());
				}
			}

			const data = await response.json();
			return json(data);
		} catch (networkError) {
			
			// For production, return error; for development, return mock data
			if (!dev) {
				return json(
					{ error: 'Unable to connect to Azure AI Foundry API' },
					{ status: 503 }
				);
			} else {
				return json(getEmptyData());
			}
		}
	} catch (error) {
		return json(
			{ error: 'Internal server error' },
			{ status: 500 }
		);
	}
};

export const GET: RequestHandler = async ({ url, fetch }) => {
	try {
		const id = url.searchParams.get('id');
		
		if (!id) {
			return json({ error: 'Model ID is required' }, { status: 400 });
		}

		// Build request for single model lookup
		const requestBody = {
			resourceIds: [
				{
					resourceId: "azureml",
					entityContainerType: "Registry"
				}
			],
			indexEntitiesRequest: {
				filters: [
					{
						field: "type",
						operator: "eq",
						values: ["models"]
					},
					{
						field: "kind",
						operator: "eq", 
						values: ["Versioned"]
					},
					{
						field: "name",
						operator: "eq",
						values: [id]
					}
				]
			}
		};

		try {
			const response = await fetch(AZURE_AI_FOUNDRY_API_URL, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'Accept': 'application/json',
					'User-Agent': 'AzureAiStudio',
				},
				body: JSON.stringify(requestBody)
			});

			if (!response.ok) {
				const errorText = await response.text();
				
				if (!dev) {
					return json(
						{ error: `Azure AI Foundry API error: ${response.status} - ${errorText}` },
						{ status: response.status }
					);
				} else {
					// Return empty data for development
					const emptyData = getEmptyData();
					const entities = emptyData?.indexEntitiesResponse?.value || [];
					const foundModel = entities.find((model: ModelEntity) => model.name?.toLowerCase().includes(id.toLowerCase()));
					
					if (foundModel) {
						return json(foundModel);
					} else {
						return json({ error: 'Model not found' }, { status: 404 });
					}
				}
			}

			const data = await response.json();
			
			// Check if model was found
			const entities = data?.indexEntitiesResponse?.value || data?.entities || [];
			if (entities.length === 0) {
				return json({ error: 'Model not found' }, { status: 404 });
			}

			// Return the first matching model
			return json(entities[0]);
		} catch (networkError) {
			
			if (!dev) {
				return json(
					{ error: 'Unable to connect to Azure AI Foundry API' },
					{ status: 503 }
				);
			} else {
				// Return empty data for development
				const emptyData = getEmptyData();
				const entities = emptyData?.indexEntitiesResponse?.value || [];
				const foundModel = entities.find((model: ModelEntity) => model.name?.toLowerCase().includes(id.toLowerCase()));
				
				if (foundModel) {
					return json(foundModel);
				} else {
					return json({ error: 'Model not found' }, { status: 404 });
				}
			}
		}
	} catch (error) {
		return json(
			{ error: 'Internal server error' },
			{ status: 500 }
		);
	}
};

// Empty data for development when external API is not accessible
function getEmptyData() {
	return {
		indexEntitiesResponse: {
			value: [] as ModelEntity[]
		}
	};
}