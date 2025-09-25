<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { foundryModelService } from './service';
	import type { GroupedFoundryModel } from './types';
	import { DEVICE_ICONS, SORT_OPTIONS, TASK_TYPE_COLORS, SUPPORTED_DEVICES } from './types';

	// Debounce utility for search
	let searchDebounceTimer: ReturnType<typeof setTimeout> | null = null;
	let debouncedSearchTerm = ''; // Initialize empty for no search filter initially

	// State
	let models: GroupedFoundryModel[] = [];
	let filteredModels: GroupedFoundryModel[] = [];
	let loading = false;
	let error = '';

	// Filter state
	let searchTerm = '';
	let selectedDevice = '';
	let selectedFramework = '';
	let selectedTaskType = '';
	let selectedPublisher = '';

	// Available filter options
	let availableDevices: string[] = [];
	let availableFrameworks: string[] = [];
	let availableTaskTypes: string[] = [];
	let availablePublishers: string[] = [];

	// Sort options
	let sortBy = 'displayName';
	let sortOrder: 'asc' | 'desc' = 'asc';

	// Pagination
	let currentPage = 1;
	let itemsPerPage = 12;
	$: totalPages = Math.ceil(filteredModels.length / itemsPerPage);
	$: paginatedModels = filteredModels.slice(
		(currentPage - 1) * itemsPerPage,
		currentPage * itemsPerPage
	);

	async function fetchFoundryModels() {
		loading = true;
		error = '';

		try {
			console.log('Starting to fetch foundry models...');

			const filters = {
				device: selectedDevice,
				framework: selectedFramework,
				taskType: selectedTaskType,
				publisher: selectedPublisher,
				searchTerm: debouncedSearchTerm // Use debounced search term
			};

			const sortOptions = { sortBy, sortOrder };
			const fetchedModels = await foundryModelService.fetchGroupedModels(filters, sortOptions);

			console.log('Received grouped models from service:', fetchedModels);
			console.log('Grouped models count:', fetchedModels.length);

			// Set models array - this should trigger reactive statements
			models = fetchedModels;
			console.log('Models array set, current length:', models.length);

			// Print first model with all its properties for debugging
			if (models.length > 0) {
				console.log('=== SAMPLE MODEL (First Model) ===');
				console.log('Full model object:', JSON.stringify(models[0], null, 2));
				console.log('=== END SAMPLE MODEL ===');
			}

			// Don't manually set filteredModels - let reactive statements handle it
			// The reactive statements will automatically apply filters and update filteredModels

			// Extract unique values for filter options
			updateFilterOptions();
		} catch (err: any) {
			error = `Failed to fetch models: ${err.message}`;
			console.error('Error fetching foundry models:', err);
		} finally {
			loading = false;
			console.log('Finished fetching, loading state:', loading);
			console.log('Models array after fetch:', models.length, 'items');
			console.log('Filtered models after fetch:', filteredModels.length, 'items');
		}
	}

	function updateFilterOptions() {
		availableDevices = [...new Set(models.flatMap((m) => m.deviceSupport))].sort();
		availableFrameworks = [
			...new Set(models.map((m) => m.framework).filter((f) => f))
		].sort() as string[];
		availableTaskTypes = [
			...new Set(models.map((m) => m.taskType).filter((t) => t))
		].sort() as string[];
		availablePublishers = [...new Set(models.map((m) => m.publisher))].sort();
	}

	function applyFilters() {
		console.log('applyFilters called with', models.length, 'models');
		console.log('Filter values:', {
			debouncedSearchTerm,
			selectedDevice,
			selectedFramework,
			selectedTaskType,
			selectedPublisher
		});

		filteredModels = models.filter((model) => {
			const matchesSearch =
				!debouncedSearchTerm ||
				model.displayName.toLowerCase().includes(debouncedSearchTerm.toLowerCase()) ||
				model.alias.toLowerCase().includes(debouncedSearchTerm.toLowerCase()) ||
				model.description.toLowerCase().includes(debouncedSearchTerm.toLowerCase()) ||
				model.tags.some((tag) => tag.toLowerCase().includes(debouncedSearchTerm.toLowerCase()));

			const matchesDevice = !selectedDevice || model.deviceSupport.includes(selectedDevice);
			const matchesFramework = !selectedFramework || model.framework === selectedFramework;
			const matchesTaskType = !selectedTaskType || model.taskType === selectedTaskType;
			const matchesPublisher = !selectedPublisher || model.publisher === selectedPublisher;

			return (
				matchesSearch && matchesDevice && matchesFramework && matchesTaskType && matchesPublisher
			);
		});

		console.log('After filtering:', filteredModels.length, 'models remain');

		// Apply sorting
		filteredModels.sort((a, b) => {
			let aVal: any;
			let bVal: any;

			// Map sort field to grouped model properties
			switch (sortBy) {
				case 'displayName':
				case 'name':
					aVal = a.displayName;
					bVal = b.displayName;
					break;
				case 'totalDownloads':
				case 'downloadCount':
					aVal = a.totalDownloads || 0;
					bVal = b.totalDownloads || 0;
					break;
				default:
					aVal = (a as any)[sortBy];
					bVal = (b as any)[sortBy];
			}

			// Handle special sorting cases
			if (sortBy === 'lastModified') {
				aVal = new Date(aVal);
				bVal = new Date(bVal);
			} else if (sortBy === 'downloadCount') {
				aVal = aVal || 0;
				bVal = bVal || 0;
			} else if (typeof aVal === 'string') {
				aVal = aVal.toLowerCase();
				bVal = bVal.toLowerCase();
			}

			if (sortOrder === 'asc') {
				return aVal > bVal ? 1 : -1;
			} else {
				return aVal < bVal ? 1 : -1;
			}
		});

		currentPage = 1; // Reset to first page when filters change
	}

	function clearFilters() {
		searchTerm = '';
		debouncedSearchTerm = '';
		selectedDevice = '';
		selectedFramework = '';
		selectedTaskType = '';
		selectedPublisher = '';
		sortBy = 'displayName';
		sortOrder = 'asc';
		currentPage = 1; // Reset pagination to first page
		// Note: filtering will happen automatically via reactive statements
	}

	function formatDate(dateString: string): string {
		return new Date(dateString).toLocaleDateString('en-US', {
			year: 'numeric',
			month: 'short',
			day: 'numeric'
		});
	}

	function getDeviceIcon(device: string): string {
		const icons: Record<string, string> = {
			npu: '🧠',
			gpu: '🎮',
			cpu: '💻',
			fpga: '⚡'
		};
		return icons[device.toLowerCase()] || '🔧';
	}

	function getTaskTypeColor(taskType: string): string {
		return TASK_TYPE_COLORS[taskType] || 'badge-neutral';
	}

	// Reactive statements for automatic filtering

	// Debounce search term updates
	$: {
		if (searchDebounceTimer) {
			clearTimeout(searchDebounceTimer);
		}
		searchDebounceTimer = setTimeout(() => {
			debouncedSearchTerm = searchTerm;
		}, 300); // 300ms debounce
	}

	// Reactive filtering - automatically applies whenever models or filters change
	$: {
		console.log('Reactive statement triggered. Models length:', models.length);
		console.log('Current filter values:', {
			debouncedSearchTerm,
			selectedDevice,
			selectedFramework,
			selectedTaskType,
			selectedPublisher,
			sortBy,
			sortOrder
		});

		if (models.length > 0) {
			console.log('Applying filters to', models.length, 'models');
			applyFilters();
			console.log('After reactive filtering: filteredModels has', filteredModels.length, 'items');
		} else {
			console.log('No models available, clearing filteredModels');
			filteredModels = [];
		}
	}

	onMount(() => {
		console.log('Component mounted, fetching models...');
		fetchFoundryModels();
	});

	onDestroy(() => {
		if (searchDebounceTimer) {
			clearTimeout(searchDebounceTimer);
		}
	});

	let description =
		'Discover and explore Azure AI Foundry local models optimized for various hardware devices including NPUs, GPUs, CPUs, FPGAs and other specialized compute platforms.';
	let keywords =
		'azure ai foundry, local models, npu models, gpu models, cpu models, onnx runtime, machine learning models, ai models, hardware optimization';
</script>

<svelte:head>
	<title>Azure AI Foundry Local Models - ONNX Runtime</title>
	<meta name="description" content={description} />
	<meta name="keywords" content={keywords} />
	<meta property="og:title" content="Azure AI Foundry Local Models - ONNX Runtime" />
	<meta property="og:description" content={description} />
	<meta property="twitter:title" content="Azure AI Foundry Local Models - ONNX Runtime" />
	<meta property="twitter:description" content={description} />
</svelte:head>

<div class="container mx-auto px-4 py-8">
	<div class="text-center mb-8">
		<h1 class="text-4xl font-bold mb-4">Azure AI Foundry Local Models</h1>
		<p class="text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
			Explore a curated collection of AI models optimized for local deployment across various
			hardware platforms. Find models specifically designed for NPUs, GPUs, CPUs, FPGAs and other
			specialized compute devices to maximize performance for your use case.
		</p>
	</div>

	<!-- Search and Filters Section -->
	<div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-8">
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4 mb-4">
			<!-- Search -->
			<div class="xl:col-span-2">
				<label class="block text-sm font-medium mb-2" for="search-models">Search Models</label>
				<div class="relative">
					<input
						id="search-models"
						type="text"
						bind:value={searchTerm}
						placeholder="Search by name, description, or tags... (searches as you type)"
						class="input input-bordered w-full"
					/>
					{#if searchTerm !== debouncedSearchTerm}
						<div class="absolute right-3 top-1/2 -translate-y-1/2">
							<span class="loading loading-xs loading-spinner" title="Searching..." />
						</div>
					{/if}
				</div>
			</div>

			<!-- Device Filter -->
			<div>
				<label class="block text-sm font-medium mb-2" for="device-filter">Device</label>
				<select
					id="device-filter"
					bind:value={selectedDevice}
					class="select select-bordered w-full"
				>
					<option value="">All Devices</option>
					{#each availableDevices as device}
						<option value={device}>{device.toUpperCase()}</option>
					{/each}
				</select>
			</div>

			<!-- Framework Filter -->
			<div>
				<label class="block text-sm font-medium mb-2" for="framework-filter">Framework</label>
				<select
					id="framework-filter"
					bind:value={selectedFramework}
					class="select select-bordered w-full"
				>
					<option value="">All Frameworks</option>
					{#each availableFrameworks as framework}
						<option value={framework}>{framework}</option>
					{/each}
				</select>
			</div>

			<!-- Task Type Filter -->
			<div>
				<label class="block text-sm font-medium mb-2" for="task-filter">Task Type</label>
				<select
					id="task-filter"
					bind:value={selectedTaskType}
					class="select select-bordered w-full"
				>
					<option value="">All Tasks</option>
					{#each availableTaskTypes as taskType}
						<option value={taskType}>{taskType}</option>
					{/each}
				</select>
			</div>

			<!-- Publisher Filter -->
			<div>
				<label class="block text-sm font-medium mb-2" for="publisher-filter">Publisher</label>
				<select
					id="publisher-filter"
					bind:value={selectedPublisher}
					class="select select-bordered w-full"
				>
					<option value="">All Publishers</option>
					{#each availablePublishers as publisher}
						<option value={publisher}>{publisher}</option>
					{/each}
				</select>
			</div>
		</div>

		<!-- Sort and Actions -->
		<div class="flex flex-wrap items-center justify-between gap-4">
			<div class="flex items-center gap-4">
				<div>
					<label class="block text-sm font-medium mb-1" for="sort-by">Sort by</label>
					<select id="sort-by" bind:value={sortBy} class="select select-bordered select-sm">
						<option value="name">Name</option>
						<option value="lastModified">Last Modified</option>
						<option value="downloadCount">Downloads</option>
						<option value="publisher">Publisher</option>
					</select>
				</div>
				<div>
					<label class="block text-sm font-medium mb-1" for="sort-order">Order</label>
					<select id="sort-order" bind:value={sortOrder} class="select select-bordered select-sm">
						<option value="asc">Ascending</option>
						<option value="desc">Descending</option>
					</select>
				</div>
			</div>

			<div class="flex items-center gap-4">
				<span class="text-sm text-gray-600 dark:text-gray-300">
					{filteredModels.length} model{filteredModels.length !== 1 ? 's' : ''} found
					{#if searchTerm !== debouncedSearchTerm}
						<span class="loading loading-xs loading-spinner ml-2" title="Searching..." />
					{/if}
				</span>
				<button on:click={clearFilters} class="btn btn-outline btn-sm"> Clear Filters </button>
				<button on:click={fetchFoundryModels} class="btn btn-primary btn-sm" disabled={loading}>
					{loading ? 'Loading...' : 'Reload Models'}
				</button>
			</div>
		</div>
	</div>

	<!-- Loading State -->
	{#if loading}
		<div class="flex justify-center items-center py-12">
			<div class="loading loading-spinner loading-lg" />
			<span class="ml-4 text-lg">Loading foundry models...</span>
		</div>
	{/if}

	<!-- Error State -->
	{#if error}
		<div class="alert alert-error mb-8">
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
				<button class="btn btn-sm btn-outline" on:click={fetchFoundryModels}> Retry </button>
			</div>
		</div>
	{/if}

	<!-- Models Grid -->
	{#if !loading && !error}
		{#if paginatedModels.length > 0}
			<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
				{#each paginatedModels as model}
					<div class="card bg-base-100 shadow-xl hover:shadow-2xl transition-shadow duration-200">
						<div class="card-body">
							<div class="flex items-start justify-between mb-3">
								<div class="flex-1">
									<h3 class="card-title text-lg font-semibold mb-1">{model.displayName}</h3>
									<p class="text-sm text-gray-500">{model.publisher}</p>
								</div>
								<div class="flex flex-col items-end gap-1">
									{#each model.deviceSupport as device}
										<span class="badge badge-primary badge-sm" title="{device} support">
											{getDeviceIcon(device)}
											{device.toUpperCase()}
										</span>
									{/each}
								</div>
							</div>

							<p class="text-sm text-gray-600 dark:text-gray-300 mb-4 line-clamp-3">
								{model.description}
							</p>

							<div class="flex flex-wrap gap-1 mb-4">
								{#if model.framework}
									<span class="badge badge-outline badge-sm">{model.framework}</span>
								{/if}
								{#if model.taskType}
									<span class="badge {getTaskTypeColor(model.taskType)} badge-sm"
										>{model.taskType === 'chat-completion' ? 'Chat-completion' : model.taskType}</span
									>
								{/if}
								{#if model.license}
									<span class="badge badge-outline badge-sm">
										<svg class="w-3 h-3 mr-1" fill="currentColor" viewBox="0 0 20 20">
											<path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z"/>
											<path fill-rule="evenodd" d="M4 5a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 3a1 1 0 000 2h6a1 1 0 100-2H7zm0 3a1 1 0 000 2h6a1 1 0 100-2H7z" clip-rule="evenodd"/>
										</svg>
										{model.license}
									</span>
								{/if}
								{#if model.modelSize}
									<span class="badge badge-ghost badge-sm">{model.modelSize}</span>
								{/if}
							</div>

							<div class="flex items-center justify-between text-xs text-gray-500 mb-4">
								<span>Updated {formatDate(model.lastModified)}</span>
								<div class="flex items-center gap-2">
									<span>v{model.latestVersion}</span>
									{#if model.totalDownloads && model.totalDownloads > 0}
										<span>•</span>
										<span>{model.totalDownloads.toLocaleString()} downloads</span>
									{/if}
								</div>
							</div>

							<div class="card-actions justify-end">
								<button class="btn btn-outline btn-sm" title="Download model">
									<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
										<path
											d="M3 17a1 1 0 011-1h12a1 1 0 011 1v2a1 1 0 01-1 1H4a1 1 0 01-1-1v-2zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z"
										/>
									</svg>
								</button>
							</div>
						</div>
					</div>
				{/each}
			</div>

			<!-- Pagination -->
			{#if totalPages > 1}
				<div class="flex justify-center">
					<div class="join">
						<button
							class="join-item btn"
							class:btn-disabled={currentPage === 1}
							on:click={() => (currentPage = Math.max(1, currentPage - 1))}
						>
							«
						</button>

						{#each Array(totalPages).fill(0) as _, i}
							{@const page = i + 1}
							{#if page === 1 || page === totalPages || (page >= currentPage - 2 && page <= currentPage + 2)}
								<button
									class="join-item btn"
									class:btn-active={currentPage === page}
									on:click={() => (currentPage = page)}
								>
									{page}
								</button>
							{:else if page === currentPage - 3 || page === currentPage + 3}
								<button class="join-item btn btn-disabled">...</button>
							{/if}
						{/each}

						<button
							class="join-item btn"
							class:btn-disabled={currentPage === totalPages}
							on:click={() => (currentPage = Math.min(totalPages, currentPage + 1))}
						>
							»
						</button>
					</div>
				</div>
			{/if}
		{:else}
			<div class="text-center py-12">
				<div class="text-6xl mb-4">🔍</div>
				<h3 class="text-xl font-semibold mb-2">No models found</h3>
				<p class="text-gray-600 dark:text-gray-300 mb-4">
					Try adjusting your search criteria or clearing the filters.
				</p>
				<button class="btn btn-primary" on:click={clearFilters}> Clear All Filters </button>
			</div>
		{/if}
	{/if}
</div>

<style>
	.line-clamp-3 {
		display: -webkit-box;
		-webkit-line-clamp: 3;
		line-clamp: 3;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}
</style>
