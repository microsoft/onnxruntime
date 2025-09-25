# Azure AI Foundry Local Models Page

This feature adds a new page to the ONNX Runtime website that allows users to browse and filter Azure AI Foundry local models optimized for various hardware devices.

## Features

### Main Page (`/models/foundry`)

- **Model Discovery**: Browse a catalog of AI models optimized for local deployment
- **Advanced Filtering**: Filter models by:
  - Device support (NPU, GPU, CPU, FPGA)
  - Framework (ONNX, PyTorch, TensorFlow, etc.)
  - Task type (Text Generation, Object Detection, Speech Recognition, etc.)
  - Publisher/Organization
  - Search by name, description, or tags
- **Sorting Options**: Sort models by name, last modified date, download count, or publisher
- **Pagination**: Clean pagination for large model catalogs
- **Responsive Design**: Mobile-friendly interface using DaisyUI components

### Model Detail Page (`/models/foundry/[id]`)

- **Comprehensive Model Information**:
  - Model metadata (version, publisher, license, size)
  - Device support and hardware requirements
  - Detailed descriptions and documentation
  - Performance benchmarks
  - Sample usage code
- **Tabbed Interface**: Organized information with tabs for:
  - Overview and general information
  - Usage instructions and sample code
  - Performance benchmarks (when available)
  - File downloads and resources
- **Direct Download Links**: Easy access to model files and resources

## API Integration

The page integrates with the Azure AI Foundry API using the query structure provided:

```json
{
	"resourceIds": [
		{
			"resourceId": "azureml",
			"entityContainerType": "Registry"
		}
	],
	"indexEntitiesRequest": {
		"filters": [
			{
				"field": "type",
				"operator": "eq",
				"values": ["models"]
			},
			{
				"field": "kind",
				"operator": "eq",
				"values": ["Versioned"]
			},
			{
				"field": "properties/variantInfo/variantMetadata/device",
				"operator": "eq",
				"values": ["npu"]
			}
		]
	}
}
```

### Fallback Handling

When the API is unavailable, the page falls back to displaying mock data with sample models to demonstrate the functionality.

## File Structure

```
src/routes/models/foundry/
├── +page.svelte           # Main models listing page
├── [id]/+page.svelte      # Individual model detail page
├── service.ts             # API service layer
├── types.ts               # TypeScript type definitions
└── README.md              # This documentation
```

## Components Used

- **Svelte/SvelteKit**: Main framework
- **DaisyUI**: UI component library for styling
- **TypeScript**: Type safety and better development experience

## Integration with Main Site

The foundry models page is integrated into the main models page (`/models`) as the first card in the model hubs grid, making it easily discoverable for users looking for local AI models.

## Development Notes

- The service layer (`service.ts`) handles all API communication and data transformation
- Mock data is provided for development and testing purposes
- The component uses responsive design principles for optimal viewing across devices
- Error handling is implemented for API failures and network issues
- The page follows the existing site's design patterns and color schemes
