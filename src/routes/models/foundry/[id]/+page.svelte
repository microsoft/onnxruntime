<script lang="ts">
	import { page } from '$app/stores';
	import { onMount } from 'svelte';
	import { foundryModelService } from '../service';
	import type { FoundryModel } from '../types';
	import { DEVICE_ICONS } from '../types';

	// State
	let model: FoundryModel | null = null;
	let loading = false;
	let error = '';

	// Get model ID from URL params
	$: modelId = $page.params.id;

	async function fetchModelDetail(id: string) {
		loading = true;
		error = '';

		try {
			model = await foundryModelService.fetchModelById(id);
			if (!model) {
				error = 'Model not found';
			}
		} catch (err: any) {
			error = `Failed to fetch model details: ${err.message}`;
		} finally {
			loading = false;
		}
	}

	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleDateString('en-US', {
			year: 'numeric',
			month: 'long',
			day: 'numeric',
			hour: '2-digit',
			minute: '2-digit'
		});
	}

	function getDeviceIcon(device: string): string {
		return DEVICE_ICONS[device.toLowerCase()] || '🔧';
	}

	function copyToClipboard(text: string) {
		navigator.clipboard.writeText(text);
	}

	onMount(() => {
		if (modelId) {
			fetchModelDetail(modelId);
		}
	});

	// Sample code template
	$: sampleCodeTemplate = model
		? `
# Install required packages
pip install onnxruntime

# Load and run ${model.name}
import onnxruntime as ort
import numpy as np

# Create inference session
session = ort.InferenceSession("${model.name.toLowerCase().replace(/\s+/g, '_')}.onnx")

# Prepare input data (adjust based on model requirements)
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = session.run(None, {"input": input_data})
print("Model output shape:", outputs[0].shape)
`
		: '';
</script>

<svelte:head>
	<title>{model?.name || 'Model Detail'} - Azure AI Foundry</title>
	<meta name="description" content={model?.description || 'Azure AI Foundry model details'} />
</svelte:head>

<div class="container mx-auto px-4 py-8">
	<!-- Back button -->
	<div class="mb-6">
		<a href="/models/foundry" class="btn btn-ghost btn-sm"> ← Back to Models </a>
	</div>

	{#if loading}
		<div class="flex justify-center items-center py-12">
			<div class="loading loading-spinner loading-lg" />
			<span class="ml-4 text-lg">Loading model details...</span>
		</div>
	{:else if error}
		<div class="alert alert-error">
			<svg
				xmlns="http://www.w3.org/2000/svg"
				class="stroke-current shrink-0 h-6 w-6"
				fill="none"
				viewBox="0 0 24 24"
			>
				<path
					stroke-linecap="round"
					stroke-linejoin="round"
					stroke-width="2"
					d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
				/>
			</svg>
			<span>{error}</span>
			<div>
				<button class="btn btn-sm btn-outline" on:click={() => fetchModelDetail(modelId)}>
					Retry
				</button>
			</div>
		</div>
	{:else if model}
		<!-- Model Header -->
		<div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 mb-8">
			<div class="flex flex-col lg:flex-row gap-6">
				<div class="flex-1">
					<h1 class="text-4xl font-bold mb-2">{model.name}</h1>
					<p class="text-2xl font-bold text-primary mb-4">Version {model.version}</p>
					<p class="text-lg mb-6">{model.description}</p>

					<div class="flex flex-wrap gap-3 mb-6">
						{#each model.deviceSupport as device}
							<span class="badge badge-primary badge-lg">
								{getDeviceIcon(device)}
								{device.toUpperCase()}
							</span>
						{/each}
					</div>

					<div class="flex gap-4">
						{#if model.githubUrl}
							<a href={model.githubUrl} target="_blank" class="btn btn-outline">
								<svg class="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 24 24">
									<path
										d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"
									/>
								</svg>
								View on GitHub
							</a>
						{/if}
						{#if model.demoUrl}
							<a href={model.demoUrl} target="_blank" class="btn btn-outline"> Try Demo </a>
						{/if}
					</div>
				</div>

				<div class="lg:w-80">
					<div class="bg-gray-50 dark:bg-gray-700 rounded-lg p-6">
						<h3 class="font-semibold mb-4">Model Information</h3>
						<div class="space-y-3 text-sm">
							<div class="flex justify-between">
								<span class="text-gray-600 dark:text-gray-400">Publisher:</span>
								<span class="font-medium">{model.publisher}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-600 dark:text-gray-400">License:</span>
								<span class="font-medium">{model.license}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-600 dark:text-gray-400">Model Size:</span>
								<span class="font-medium">{model.modelSize}</span>
							</div>
							{#if model.downloadCount}
								<div class="flex justify-between">
									<span class="text-gray-600 dark:text-gray-400">Downloads:</span>
									<span class="font-medium">{model.downloadCount.toLocaleString()}</span>
								</div>
							{/if}
							<div class="flex justify-between">
								<span class="text-gray-600 dark:text-gray-400">Created:</span>
								<span class="font-medium">{formatDate(model.createdDate)}</span>
							</div>
							<div class="flex justify-between">
								<span class="text-gray-600 dark:text-gray-400">Updated:</span>
								<span class="font-medium">{formatDate(model.lastModified)}</span>
							</div>
						</div>
					</div>
				</div>
			</div>
		</div>

		<!-- Tabs Section -->
		<div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg">
			<div class="tabs tabs-lifted">
				<input type="radio" name="model_tabs" class="tab" aria-label="Overview" checked />
				<div class="tab-content bg-base-100 border-base-300 rounded-box p-6">
					<h2 class="text-2xl font-bold mb-4">Overview</h2>

					{#if model.longDescription}
						<div class="prose max-w-none mb-6">
							<p>{model.longDescription}</p>
						</div>
					{/if}

					{#if model.tags.length > 0}
						<div class="mb-6">
							<h3 class="text-lg font-semibold mb-3">Tags</h3>
							<div class="flex flex-wrap gap-2">
								{#each model.tags as tag}
									<span class="badge badge-outline">{tag}</span>
								{/each}
							</div>
						</div>
					{/if}

					{#if model.requirements && model.requirements.length > 0}
						<div class="mb-6">
							<h3 class="text-lg font-semibold mb-3">Requirements</h3>
							<ul class="list-disc list-inside space-y-1">
								{#each model.requirements as requirement}
									<li>{requirement}</li>
								{/each}
							</ul>
						</div>
					{/if}
				</div>

				<input type="radio" name="model_tabs" class="tab" aria-label="Usage" />
				<div class="tab-content bg-base-100 border-base-300 rounded-box p-6">
					<h2 class="text-2xl font-bold mb-4">Usage</h2>

					<div class="mb-6">
						<div class="flex items-center justify-between mb-3">
							<h3 class="text-lg font-semibold">Sample Code</h3>
							<button
								class="btn btn-sm btn-outline"
								on:click={() => copyToClipboard(sampleCodeTemplate)}
							>
								Copy Code
							</button>
						</div>
						<div class="mockup-code">
							<pre><code>{sampleCodeTemplate}</code></pre>
						</div>
					</div>

					<div class="grid grid-cols-1 md:grid-cols-2 gap-6">
						<div>
							<h3 class="text-lg font-semibold mb-3">Input Format</h3>
							<p class="text-gray-600 dark:text-gray-300">{model.inputFormat || 'Not specified'}</p>
						</div>
						<div>
							<h3 class="text-lg font-semibold mb-3">Output Format</h3>
							<p class="text-gray-600 dark:text-gray-300">
								{model.outputFormat || 'Not specified'}
							</p>
						</div>
					</div>
				</div>

				{#if model.benchmarks && model.benchmarks.length > 0}
					<input type="radio" name="model_tabs" class="tab" aria-label="Benchmarks" />
					<div class="tab-content bg-base-100 border-base-300 rounded-box p-6">
						<h2 class="text-2xl font-bold mb-4">Benchmarks</h2>

						<div class="overflow-x-auto">
							<table class="table table-zebra">
								<thead>
									<tr>
										<th>Metric</th>
										<th>Value</th>
										<th>Device</th>
									</tr>
								</thead>
								<tbody>
									{#each model.benchmarks as benchmark}
										<tr>
											<td>{benchmark.metric}</td>
											<td class="font-mono">{benchmark.value}</td>
											<td>
												<span class="badge badge-outline">
													{getDeviceIcon(benchmark.device)}
													{benchmark.device.toUpperCase()}
												</span>
											</td>
										</tr>
									{/each}
								</tbody>
							</table>
						</div>
					</div>
				{/if}

				<input type="radio" name="model_tabs" class="tab" aria-label="Files" />
				<div class="tab-content bg-base-100 border-base-300 rounded-box p-6">
					<h2 class="text-2xl font-bold mb-4">Model Files</h2>

					<div class="alert alert-info mb-4">
						<svg
							xmlns="http://www.w3.org/2000/svg"
							fill="none"
							viewBox="0 0 24 24"
							class="stroke-current shrink-0 w-6 h-6"
						>
							<path
								stroke-linecap="round"
								stroke-linejoin="round"
								stroke-width="2"
								d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
							/>
						</svg>
						<span>This section would display the model files available for download.</span>
					</div>

					<div class="space-y-3">
						<div
							class="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-600 rounded-lg"
						>
							<div>
								<div class="font-medium">{model.name.toLowerCase().replace(/\s+/g, '_')}.onnx</div>
								<div class="text-sm text-gray-500">Main model file</div>
							</div>
						</div>

						<div
							class="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-600 rounded-lg"
						>
							<div>
								<div class="font-medium">config.json</div>
								<div class="text-sm text-gray-500">Model configuration</div>
							</div>
						</div>

						<div
							class="flex items-center justify-between p-4 border border-gray-200 dark:border-gray-600 rounded-lg"
						>
							<div>
								<div class="font-medium">README.md</div>
								<div class="text-sm text-gray-500">Documentation</div>
							</div>
							<button class="btn btn-sm btn-outline">View</button>
						</div>
					</div>
				</div>
			</div>
		</div>
	{:else}
		<div class="text-center py-12">
			<h2 class="text-2xl font-bold mb-2">Model not found</h2>
			<p class="text-gray-600 dark:text-gray-300 mb-4">The requested model could not be found.</p>
			<a href="/models/foundry" class="btn btn-primary">Browse Models</a>
		</div>
	{/if}
</div>
