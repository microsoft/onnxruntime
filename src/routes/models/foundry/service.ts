// API service for Azure AI Foundry models
// Uses Azure Function as CORS proxy for static deployment
import type { FoundryModel, GroupedFoundryModel } from './types';

// Azure Function endpoint for CORS proxy
const FOUNDRY_API_ENDPOINT = 'https://onnxruntime-foundry-proxy-hpape7gzf2haesef.eastus-01.azurewebsites.net/api/foundryproxy';

export interface ApiFilters {
	device?: string;
	family?: string;
	acceleration?: string;
	searchTerm?: string;
}

export interface ApiSortOptions {
	sortBy: string;
	sortOrder: 'asc' | 'desc';
}

export class FoundryModelService {
	// Cache for all models - fetched once
	private allModelsCache: FoundryModel[] | null = null;
	private fetchPromise: Promise<FoundryModel[]> | null = null;

	// Detect acceleration from model name
	private detectAcceleration(modelName: string): string | undefined {
		const nameLower = modelName.toLowerCase();
		if (nameLower.includes('-qnn-') || nameLower.includes('qnn')) {
			return 'qnn';
		}
		if (nameLower.includes('-vitis-') || nameLower.includes('vitis')) {
			return 'vitis';
		}
		if (nameLower.includes('-openvino-') || nameLower.includes('openvino')) {
			return 'openvino';
		}
		if (nameLower.includes('-trt-') || nameLower.includes('tensorrt') || nameLower.includes('trt-rtx')) {
			return 'trt-rtx';
		}
		return undefined;
	}

	// Get acceleration display name
	getAccelerationDisplayName(acceleration: string): string {
		const accelerationNames: Record<string, string> = {
			'qnn': 'Qualcomm QNN',
			'vitis': 'AMD Vitis AI',
			'openvino': 'Intel OpenVINO',
			'trt-rtx': 'NVIDIA TensorRT RTX'
		};
		return accelerationNames[acceleration] || acceleration;
	}

	// Fetch all models from API once and cache them
	async fetchAllModels(): Promise<FoundryModel[]> {
		// Return cached models if available
		if (this.allModelsCache) {
			console.log('Returning cached models:', this.allModelsCache.length);
			return this.allModelsCache;
		}

		// If already fetching, return the existing promise
		if (this.fetchPromise) {
			console.log('Fetch already in progress, waiting...');
			return this.fetchPromise;
		}

		console.log('=== FETCHING ALL MODELS FROM API (ONE TIME) ===');
		
		// Create fetch promise
		this.fetchPromise = this.fetchModelsFromAPI();
		
		try {
			this.allModelsCache = await this.fetchPromise;
			console.log(`Cached ${this.allModelsCache.length} models`);
			return this.allModelsCache;
		} finally {
			this.fetchPromise = null;
		}
	}

	// Clear cache if needed (for refresh)
	clearCache(): void {
		console.log('Clearing model cache');
		this.allModelsCache = null;
		this.fetchPromise = null;
	}

	async fetchModels(filters: ApiFilters = {}, sortOptions: ApiSortOptions = { sortBy: 'name', sortOrder: 'asc' }): Promise<FoundryModel[]> {
		console.log('=== FOUNDRY SERVICE FETCH MODELS START ===');
		console.log('Input filters:', filters);
		console.log('Input sort options:', sortOptions);
		
		// Get all models from cache or API
		const allModels = await this.fetchAllModels();
		
		// Apply filters and sorting client-side
		const filteredModels = this.applyClientSideProcessing(allModels, filters, sortOptions);
		console.log(`Final result after filtering: ${filteredModels.length} models`);
		
		return filteredModels;
	}
	
	private async fetchModelsFromAPI(): Promise<FoundryModel[]> {
		const devices = ['npu', 'gpu', 'cpu'];
		const allModels: FoundryModel[] = [];
		const seenModelIds = new Set<string>();
		
		for (const device of devices) {
			try {
				console.log(`Fetching models for device: ${device}`);
				const deviceModels = await this.fetchModelsForDevice(device);
				console.log(`Got ${deviceModels.length} models for device ${device}`);
				
				// Add models that we haven't seen before (avoid duplicates)
				let newModelsCount = 0;
				for (const model of deviceModels) {
					if (!seenModelIds.has(model.id)) {
						seenModelIds.add(model.id);
						allModels.push(model);
						newModelsCount++;
					}
				}
				console.log(`Added ${newModelsCount} new unique models from device ${device}`);
			} catch (error) {
				console.error(`Failed to fetch models for device ${device}:`, error);
				// Continue with other devices if one fails
			}
		}
		
		console.log(`Total unique models collected: ${allModels.length}`);
		return allModels;
	}
	
	private async fetchModelsForDevice(device: string): Promise<FoundryModel[]> {
		const requestBody = this.buildRequestBody(device);
		
		// Debug logging - client side
		console.log('=== FOUNDRY CLIENT DEBUG ===');
		console.log('Device:', device);
		console.log('Request body built:', JSON.stringify(requestBody, null, 2));
		console.log('API endpoint:', FOUNDRY_API_ENDPOINT);
		
		try {
			console.log('Making fetch request...');
			const startTime = performance.now();
			
			const response = await fetch(FOUNDRY_API_ENDPOINT, {
				method: 'POST',
				headers: {
					'Content-Type': 'application/json',
					'Accept': 'application/json'
				},
				body: JSON.stringify(requestBody)
			});

			const endTime = performance.now();
			console.log(`Request completed in ${endTime - startTime}ms`);
			console.log('Response status:', response.status);
			console.log('Response status text:', response.statusText);
			console.log('Response headers:', Object.fromEntries(response.headers.entries()));

			if (!response.ok) {
				console.error('=== API REQUEST FAILED ===');
				console.error(`API request failed with status: ${response.status} ${response.statusText}`);
				
				// Try to get error details from response body
				try {
					const errorText = await response.text();
					console.error('Error response body:', errorText);
					
					// Try to parse as JSON for better error details
					try {
						const errorJson = JSON.parse(errorText);
						console.error('Parsed error JSON:', errorJson);
					} catch (e) {
						console.error('Could not parse error response as JSON');
					}
				} catch (e) {
					console.error('Could not read error response body:', e);
				}
				
				return [];
			}

			console.log('Response successful, parsing JSON...');
			const apiData = await response.json();
			console.log('Raw API response:', JSON.stringify(apiData, null, 2));
			console.log('API response keys:', apiData ? Object.keys(apiData) : 'null/undefined');
			
			const models = this.transformApiResponse(apiData);
			console.log(`Transformed ${models.length} models from API response`);
			console.log('Sample transformed model:', models.length > 0 ? models[0] : 'none');
			
			return models;
			
		} catch (error) {
			console.error('=== FETCH ERROR ===');
			console.error('Failed to fetch foundry models:', error);
			console.error('Error name:', error instanceof Error ? error.name : 'unknown');
			console.error('Error message:', error instanceof Error ? error.message : error);
			console.error('Error stack:', error instanceof Error ? error.stack : 'no stack');
			console.error('Request that caused error:', JSON.stringify(requestBody, null, 2));
			return [];
		}
	}

	private buildRequestBody(device: string) {
		console.log('=== BUILDING REQUEST BODY ===');
		console.log('Device for request body:', device);
		
		const baseFilters = [
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
				field: "properties/variantInfo/variantMetadata/device",
				operator: "eq",
				values: [device]
			}
		];

		console.log('Filters:', baseFilters);

		const requestBody = {
			resourceIds: [
				{
					resourceId: "azureml",
					entityContainerType: "Registry"
				}
			],
			indexEntitiesRequest: {
				filters: baseFilters
			}
		};

		console.log('Final request body:', JSON.stringify(requestBody, null, 2));
		return requestBody;
	}

	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	private transformApiResponse(apiData: any): FoundryModel[] {
		// Handle the Azure AI Foundry API response structure
		let entities = [];
		
		// Check for entities in the response structure
		if (apiData?.indexEntitiesResponse?.value && Array.isArray(apiData.indexEntitiesResponse.value)) {
			entities = apiData.indexEntitiesResponse.value;
		}
		// Fallback: check if entities are directly in the response
		else if (apiData?.entities && Array.isArray(apiData.entities)) {
			entities = apiData.entities;
		}
		
		if (entities.length === 0) {
			return [];
		}

		// eslint-disable-next-line @typescript-eslint/no-explicit-any
		return entities.map((entity: any) => this.transformSingleModel(entity));
	}

	// eslint-disable-next-line @typescript-eslint/no-explicit-any
	private transformSingleModel(entity: any): FoundryModel {
		
		// Extract unique values from devices array
		const deviceSupport: string[] = [];
		
		// Extract device support from various possible locations in the Azure API response
		if (entity.annotations?.tags?.device) {
			deviceSupport.push(entity.annotations.tags.device);
		}
		if (entity.properties?.variantInfo?.variantMetadata?.device) {
			deviceSupport.push(entity.properties.variantInfo.variantMetadata.device);
		}
		if (entity.properties?.supportedDevices) {
			deviceSupport.push(...entity.properties.supportedDevices);
		}
		
		// If no device support found, try to infer from model name or ID
		if (deviceSupport.length === 0) {
			const modelId = entity.entityId || '';
			const modelName = entity.name || entity.displayName || '';
			
			if (modelId.toLowerCase().includes('npu') || modelName.toLowerCase().includes('npu')) {
				deviceSupport.push('npu');
			} else if (modelId.toLowerCase().includes('gpu') || modelName.toLowerCase().includes('gpu')) {
				deviceSupport.push('gpu');
			} else if (modelId.toLowerCase().includes('cpu') || modelName.toLowerCase().includes('cpu')) {
				deviceSupport.push('cpu');
			} else {
				// Default to cpu if we can't determine
				deviceSupport.push('cpu');
			}
		}

		// Remove duplicates manually
		const uniqueDeviceSupport = deviceSupport.filter((device, index) => deviceSupport.indexOf(device) === index);

		// Extract name from entityId which looks like: "azureml://registries/.../deepseek-r1-distill-qwen-1.5b-qnn-npu/version/1"
		let modelName = entity.name || entity.displayName || 'Unknown Model';
		if (entity.entityId) {
			const idParts = entity.entityId.split('/');
			const objectIdIndex = idParts.findIndex((part: string) => part === 'objectId');
			if (objectIdIndex >= 0 && objectIdIndex + 1 < idParts.length) {
				modelName = idParts[objectIdIndex + 1];
			}
		}
		
		// Extract version from entityId
		let version = entity.version || '1.0.0';
		if (entity.entityId) {
			const versionMatch = entity.entityId.match(/\/version\/(\d+(?:\.\d+)*)/);
			if (versionMatch) {
				version = versionMatch[1];
			}
		}

		// Extract tags from annotations
		const tags: string[] = [];
		if (entity.annotations?.tags) {
			Object.entries(entity.annotations.tags).forEach(([key, value]) => {
				if (typeof value === 'string' && value !== 'true' && value !== 'false') {
					tags.push(`${key}:${value}`);
				} else if (key !== 'archived' && key !== 'invisible') {
					tags.push(key);
				}
			});
		}

		// Detect acceleration from model name
		const acceleration = this.detectAcceleration(modelName);

		return {
			id: entity.entityId || entity.id || modelName.toLowerCase().replace(/\s+/g, '-'),
			name: modelName,
			version: version,
			description: entity.description || entity.annotations?.description || `${modelName} model for NPU inference`,
			longDescription: entity.properties?.readme || entity.properties?.longDescription || entity.annotations?.longDescription,
			deviceSupport: uniqueDeviceSupport,
			tags: tags,
			publisher: entity.annotations?.publisher || entity.owner || entity.properties?.author || entity.entityResourceName || 'Azure ML',
			acceleration: acceleration,
			lastModified: entity.lastModifiedDateTime || entity.properties?.lastModified || new Date().toISOString(),
			createdDate: entity.createdDateTime || entity.properties?.created || new Date().toISOString(),
			downloadCount: entity.properties?.downloadCount || entity.stats?.downloadCount || 0,
			framework: entity.annotations?.tags?.framework || entity.properties?.framework || entity.properties?.modelFramework || 'ONNX',
			license: entity.properties?.license || entity.license || entity.annotations?.license || 'MIT',
			taskType: entity.annotations?.tags?.task || entity.properties?.taskType || entity.properties?.task || 'Text Generation',
			modelSize: this.formatModelSize(entity.properties?.modelSize || entity.properties?.size),
			inputFormat: entity.properties?.inputFormat || entity.properties?.inputs?.format,
			outputFormat: entity.properties?.outputFormat || entity.properties?.outputs?.format,
			sampleCode: entity.properties?.sampleCode,
			documentation: entity.properties?.documentation || entity.properties?.readme,
			githubUrl: entity.properties?.githubUrl || entity.properties?.sourceUrl,
			paperUrl: entity.properties?.paperUrl || entity.properties?.paper,
			demoUrl: entity.properties?.demoUrl || entity.properties?.demo,
			benchmarks: this.extractBenchmarks(entity.properties?.benchmarks),
			requirements: entity.properties?.requirements || entity.properties?.dependencies || [],
			compatibleVersions: entity.properties?.compatibleVersions || []
		};
	}

	private extractBenchmarks(benchmarkData: unknown): Array<{ metric: string; value: string; device: string }> {
		if (!benchmarkData) return [];
		
		if (Array.isArray(benchmarkData)) {
			return benchmarkData.map((b: Record<string, unknown>) => ({
				metric: (b.metric as string) || (b.name as string) || 'Unknown',
				value: (b.value as number | string)?.toString() || 'N/A',
				device: (b.device as string) || 'npu'
			}));
		}
		
		// Handle object format
		if (typeof benchmarkData === 'object') {
			const benchmarks: Array<{ metric: string; value: string; device: string }> = [];
			const data = benchmarkData as Record<string, unknown>;
			for (const metric in data) {
				if (Object.prototype.hasOwnProperty.call(data, metric)) {
					benchmarks.push({
						metric,
						value: (data[metric] as number | string)?.toString() || 'N/A',
						device: 'npu'
					});
				}
			}
			return benchmarks;
		}
		
		return [];
	}

	private formatModelSize(size: unknown): string {
		if (!size) return 'Unknown';
		
		if (typeof size === 'number') {
			// Convert bytes to human readable format
			const units = ['B', 'KB', 'MB', 'GB', 'TB'];
			let index = 0;
			let sizeNum = size;
			
			while (sizeNum >= 1024 && index < units.length - 1) {
				sizeNum /= 1024;
				index++;
			}
			
			return `${sizeNum.toFixed(1)} ${units[index]}`;
		}
		
		return size.toString();
	}

	private applyClientSideProcessing(models: FoundryModel[], filters: ApiFilters, sortOptions: ApiSortOptions): FoundryModel[] {
		let filteredModels = [...models];

		// Apply device filter
		if (filters.device) {
			const deviceFilter = filters.device;
			filteredModels = filteredModels.filter(model => 
				model.deviceSupport.includes(deviceFilter)
			);
		}

		// Apply family filter (searches in model name/alias)
		if (filters.family) {
			const familyLower = filters.family.toLowerCase();
			filteredModels = filteredModels.filter(model => {
				const nameMatch = model.name.toLowerCase().includes(familyLower);
				return nameMatch;
			});
		}

		// Apply search filter
		if (filters.searchTerm) {
			const searchLower = filters.searchTerm.toLowerCase();
			filteredModels = filteredModels.filter(model => {
				const nameMatch = model.name.toLowerCase().indexOf(searchLower) >= 0;
				const descMatch = model.description.toLowerCase().indexOf(searchLower) >= 0;
				const tagMatch = model.tags.some((tag: string) => tag.toLowerCase().indexOf(searchLower) >= 0);
				const publisherMatch = model.publisher.toLowerCase().indexOf(searchLower) >= 0;
				return nameMatch || descMatch || tagMatch || publisherMatch;
			});
		}

		// Apply acceleration filter
		if (filters.acceleration) {
			filteredModels = filteredModels.filter(model => model.acceleration === filters.acceleration);
		}

		// Sort models
		filteredModels.sort((a, b) => {
			const aVal: unknown = a[sortOptions.sortBy as keyof FoundryModel];
			const bVal: unknown = b[sortOptions.sortBy as keyof FoundryModel];

			// Handle special sorting cases
			if (sortOptions.sortBy === 'lastModified' || sortOptions.sortBy === 'createdDate') {
				const aDate = new Date(aVal as string | number | Date);
				const bDate = new Date(bVal as string | number | Date);
				if (sortOptions.sortOrder === 'asc') {
					return aDate.getTime() - bDate.getTime();
				} else {
					return bDate.getTime() - aDate.getTime();
				}
			} else if (sortOptions.sortBy === 'downloadCount') {
				const aNum = Number(aVal) || 0;
				const bNum = Number(bVal) || 0;
				if (sortOptions.sortOrder === 'asc') {
					return aNum - bNum;
				} else {
					return bNum - aNum;
				}
			} else {
				// String comparison
				const aStr = String(aVal || '').toLowerCase();
				const bStr = String(bVal || '').toLowerCase();
				
				if (sortOptions.sortOrder === 'asc') {
					return aStr.localeCompare(bStr);
				} else {
					return bStr.localeCompare(aStr);
				}
			}
		});

		return filteredModels;
	}

	// Helper function to extract alias from model name
	private extractAlias(modelName: string): string {
		// Remove device-specific suffixes like -cuda-gpu, -generic-cpu, -npu
		const deviceSuffixes = [
			'-cuda-gpu', '-cuda', '-gpu',
			'-generic-cpu', '-cpu', 
			'-npu',
			'-fpga',
			'-asic'
		];
		
		let alias = modelName.toLowerCase();
		
		// Remove device suffixes
		for (const suffix of deviceSuffixes) {
			if (alias.endsWith(suffix)) {
				alias = alias.slice(0, -suffix.length);
				break;
			}
		}
		
		return alias;
	}

	// Helper function to create display name from alias
	private createDisplayName(alias: string): string {
		// Convert kebab-case to more readable format
		return alias
			.split('-')
			.map(word => word.charAt(0).toUpperCase() + word.slice(1))
			.join(' ');
	}

	// Group models by alias and return grouped models
	async fetchGroupedModels(filters: ApiFilters = {}, sortOptions: ApiSortOptions = { sortBy: 'name', sortOrder: 'asc' }): Promise<GroupedFoundryModel[]> {
		console.log('=== FETCH GROUPED MODELS START ===');
		console.log('Filters received:', filters);
		console.log('Sort options received:', sortOptions);
		
		// First get all individual models
		console.log('Calling fetchModels...');
		const allModels = await this.fetchModels(filters, sortOptions);
		console.log(`fetchModels returned ${allModels.length} individual models`);
		
		// Group models by alias
		console.log('Grouping models by alias...');
		const modelGroups = new Map<string, FoundryModel[]>();
		
		for (const model of allModels) {
			const alias = this.extractAlias(model.name);
			console.log(`Model ${model.name} -> alias: ${alias}`);
			if (!modelGroups.has(alias)) {
				modelGroups.set(alias, []);
			}
			const group = modelGroups.get(alias);
			if (group) {
				group.push(model);
			}
		}
		
		console.log(`Created ${modelGroups.size} model groups`);

		// Convert groups to GroupedFoundryModel objects
		const groupedModels: GroupedFoundryModel[] = [];
		
		for (const [alias, variants] of modelGroups) {
			console.log(`Processing group ${alias} with ${variants.length} variants`);
			
			// Sort variants to get the primary one (usually the first alphabetically)
			variants.sort((a, b) => a.name.localeCompare(b.name));
			const primaryModel = variants[0];
			
			// Combine device support from all variants
			const deviceSupport = [...new Set(variants.flatMap(v => v.deviceSupport))].sort();
			
			// Combine tags from all variants
			const tags = [...new Set(variants.flatMap(v => v.tags))].sort();
			
			// Sum download counts
			const totalDownloads = variants.reduce((sum, v) => sum + (v.downloadCount || 0), 0);
			
			// Get latest modification date
			const latestModified = variants.reduce((latest, v) => {
				const vDate = new Date(v.lastModified);
				const latestDate = new Date(latest);
				return vDate > latestDate ? v.lastModified : latest;
			}, variants[0].lastModified);

			// Get earliest creation date
			const earliestCreated = variants.reduce((earliest, v) => {
				const vDate = new Date(v.createdDate);
				const earliestDate = new Date(earliest);
				return vDate < earliestDate ? v.createdDate : earliest;
			}, variants[0].createdDate);

			// Get latest version
			const latestVersion = variants.reduce((latest, v) => {
				// Simple version comparison - you might want to improve this
				return v.version > latest ? v.version : latest;
			}, variants[0].version);

			const groupedModel: GroupedFoundryModel = {
				alias,
				displayName: this.createDisplayName(alias),
				description: primaryModel.description,
				longDescription: primaryModel.longDescription,
				deviceSupport,
				tags,
				publisher: primaryModel.publisher,
				acceleration: primaryModel.acceleration,
				lastModified: latestModified,
				createdDate: earliestCreated,
				downloadCount: totalDownloads,
				framework: primaryModel.framework,
				license: primaryModel.license,
				taskType: primaryModel.taskType,
				modelSize: primaryModel.modelSize,
				variants,
				availableDevices: deviceSupport,
				totalDownloads,
				latestVersion,
				documentation: primaryModel.documentation,
				githubUrl: primaryModel.githubUrl,
				paperUrl: primaryModel.paperUrl,
				demoUrl: primaryModel.demoUrl,
				benchmarks: primaryModel.benchmarks,
				requirements: primaryModel.requirements,
				compatibleVersions: primaryModel.compatibleVersions
			};

			groupedModels.push(groupedModel);
			console.log(`Added grouped model: ${alias} with ${variants.length} variants`);
		}
		
		console.log(`Created ${groupedModels.length} grouped models before sorting`);

		// Apply sorting to grouped models
		groupedModels.sort((a, b) => {
			let aVal: string | number = '';
			let bVal: string | number = '';

			// Type-safe property access
			switch (sortOptions.sortBy) {
				case 'alias':
					aVal = a.alias;
					bVal = b.alias;
					break;
				case 'displayName':
					aVal = a.displayName;
					bVal = b.displayName;
					break;
				case 'description':
					aVal = a.description;
					bVal = b.description;
					break;
				case 'publisher':
					aVal = a.publisher;
					bVal = b.publisher;
					break;
				case 'totalDownloads':
				case 'downloadCount':
					aVal = a.totalDownloads || 0;
					bVal = b.totalDownloads || 0;
					break;
				case 'lastModified':
					aVal = a.lastModified;
					bVal = b.lastModified;
					break;
				case 'createdDate':
					aVal = a.createdDate;
					bVal = b.createdDate;
					break;
				default:
					aVal = a.displayName;
					bVal = b.displayName;
			}

			if (typeof aVal === 'number' && typeof bVal === 'number') {
				return sortOptions.sortOrder === 'asc' ? aVal - bVal : bVal - aVal;
			} else {
				const aStr = String(aVal).toLowerCase();
				const bStr = String(bVal).toLowerCase();
				return sortOptions.sortOrder === 'asc' ? aStr.localeCompare(bStr) : bStr.localeCompare(aStr);
			}
		});

		console.log(`Final result: ${groupedModels.length} grouped models after sorting`);
		console.log('Sample grouped model:', groupedModels.length > 0 ? {
			alias: groupedModels[0].alias,
			displayName: groupedModels[0].displayName,
			variantCount: groupedModels[0].variants.length,
			deviceSupport: groupedModels[0].deviceSupport
		} : 'none');

		return groupedModels;
	}

	async fetchModelById(modelId: string): Promise<FoundryModel | null> {
		try {
			const response = await fetch(`${FOUNDRY_API_ENDPOINT}?id=${encodeURIComponent(modelId)}`);
			
			if (!response.ok) {
				console.error(`Failed to fetch model ${modelId}: ${response.status}`);
				return null;
			}

			const apiData = await response.json();
			const models = this.transformApiResponse(apiData);
			
			return models.length > 0 ? models[0] : null;
		} catch (error) {
			console.error(`Error fetching model ${modelId}:`, error);
			return null;
		}
	}
}

export const foundryModelService = new FoundryModelService();