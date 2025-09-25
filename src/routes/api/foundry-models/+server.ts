// SvelteKit API route to proxy requests to Azure AI Foundry API
// This bypasses CORS restrictions since server-to-server requests aren't subject to CORS
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const AZURE_API_BASE = 'https://ai.azure.com/api/eastus/ux/v1.0/entities';

export const POST: RequestHandler = async ({ request }) => {
	try {
		// Get the request body from the frontend
		const requestBody = await request.json();
		
		console.log('Proxying request to Azure AI Foundry API:', JSON.stringify(requestBody, null, 2));

		// Make the request to Azure AI Foundry API from the server
		const response = await fetch(`${AZURE_API_BASE}/crossRegion`, {
			method: 'POST',
			headers: {
				'Content-Type': 'application/json',
				'User-Agent': 'AzureAiStudio',
				'Accept': 'application/json',
			},
			body: JSON.stringify(requestBody)
		});

		if (!response.ok) {
			const errorText = await response.text();
			console.error(`Azure API error ${response.status}:`, errorText);
			
			return json(
				{ error: `Azure API returned ${response.status}`, details: errorText },
				{ status: response.status }
			);
		}

		const data = await response.json();
		
		// Check different possible structures for entity count
		const entityCount = data?.indexEntitiesResponse?.value?.length || 
		                   data?.entities?.length || 
		                   data?.indexEntitiesResponse?.entities?.length || 0;
		
		console.log(`Successfully fetched ${entityCount} models from Azure API`);
		console.log('Full response structure check:');
		console.log('- data.entities exists:', !!data?.entities);
		console.log('- data.indexEntitiesResponse exists:', !!data?.indexEntitiesResponse);
		console.log('- data.indexEntitiesResponse.value exists:', !!data?.indexEntitiesResponse?.value);
		console.log('- data.indexEntitiesResponse.value length:', data?.indexEntitiesResponse?.value?.length || 0);
		
		// Return the data to the frontend
		return json(data);

	} catch (err) {
		console.error('Error in foundry models proxy:', err);
		
		return json(
			{ 
				error: 'Failed to fetch models from Azure API',
				details: err instanceof Error ? err.message : 'Unknown error'
			},
			{ status: 500 }
		);
	}
};

// Optional: Add GET handler for model details
export const GET: RequestHandler = async ({ url }) => {
	const modelId = url.searchParams.get('id');
	
	if (!modelId) {
		return json({ error: 'Model ID is required' }, { status: 400 });
	}

	try {
		const response = await fetch(`${AZURE_API_BASE}/model/${modelId}`, {
			method: 'GET',
			headers: {
				'Content-Type': 'application/json',
				'User-Agent': 'AzureAiStudio'
			}
		});

		if (!response.ok) {
			const errorText = await response.text();
			return json(
				{ error: `Azure API returned ${response.status}`, details: errorText },
				{ status: response.status }
			);
		}

		const data = await response.json();
		return json(data);

	} catch (err) {
		console.error('Error fetching model details:', err);
		return json(
			{
				error: 'Failed to fetch model details',
				details: err instanceof Error ? err.message : 'Unknown error'
			},
			{ status: 500 }
		);
	}
};