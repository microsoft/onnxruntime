<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import { foundryModelService } from './service';
	import type { GroupedFoundryModel } from './types';

	// Debounce utility for search
	let searchDebounceTimer: ReturnType<typeof setTimeout> | null = null;
	let debouncedSearchTerm = ''; // Initialize empty for no search filter initially

	// State
	let allModels: GroupedFoundryModel[] = []; // All models from API (fetched once)
	let filteredModels: GroupedFoundryModel[] = [];
	let loading = false;
	let error = '';
	let showCopyToast = false;
	let copyToastTimer: ReturnType<typeof setTimeout> | null = null;

	// Filter state
	let searchTerm = '';
	let selectedDevices: string[] = []; // Changed to array for multi-select
	let selectedFamily = '';
	let selectedAcceleration = '';

	// Available filter options
	let availableDevices: string[] = [];
	let availableFamilies: string[] = ['deepseek', 'mistral', 'qwen', 'phi'];
	let availableAccelerations: string[] = [];

	// Toggle device selection
	function toggleDevice(device: string) {
		if (selectedDevices.includes(device)) {
			selectedDevices = selectedDevices.filter((d) => d !== device);
		} else {
			selectedDevices = [...selectedDevices, device];
		}
	}

	// Sort options
	let sortBy = 'name';
	let sortOrder: 'asc' | 'desc' = 'asc';

	// Pagination
	let currentPage = 1;
	let itemsPerPage = 12;
	$: totalPages = Math.ceil(filteredModels.length / itemsPerPage);
	$: paginatedModels = filteredModels.slice(
		(currentPage - 1) * itemsPerPage,
		currentPage * itemsPerPage
	);

	// Fetch all models from API once (no filters applied at API level)
	async function fetchAllModels() {
		console.log('=== FETCHING ALL MODELS FROM API (ONE TIME) ===');

		loading = true;
		error = '';

		try {
			// Fetch all models without any filters
			// The service will cache the results
			console.log('Calling foundryModelService.fetchGroupedModels...');
			const fetchedModels = await foundryModelService.fetchGroupedModels(
				{},
				{ sortBy: 'name', sortOrder: 'asc' }
			);
			console.log('Service returned models:', fetchedModels.length);

			allModels = fetchedModels;
			updateFilterOptions();
			console.log('Filter options updated, available devices:', availableDevices);
			console.log('Available accelerations:', availableAccelerations);
		} catch (err: any) {
			console.error('=== SVELTE COMPONENT FETCH ERROR ===');
			console.error('Error caught:', err);
			console.error('Error name:', err?.name);
			console.error('Error message:', err?.message);
			console.error('Error stack:', err?.stack);
			error = `Failed to fetch models: ${err.message}`;
		} finally {
			loading = false;
		}
	}

	// Refresh models (clears cache and refetches)
	async function refreshModels() {
		foundryModelService.clearCache();
		await fetchAllModels();
	}

	function updateFilterOptions() {
		availableDevices = [...new Set(allModels.flatMap((m) => m.deviceSupport))].sort();
		availableAccelerations = [
			...new Set(allModels.map((m) => m.acceleration).filter((h): h is string => !!h))
		].sort();
	}

	function applyFilters() {
		// Filter from allModels, not models
		filteredModels = allModels.filter((model) => {
			const matchesSearch =
				!debouncedSearchTerm ||
				model.displayName.toLowerCase().includes(debouncedSearchTerm.toLowerCase()) ||
				model.alias.toLowerCase().includes(debouncedSearchTerm.toLowerCase()) ||
				model.description.toLowerCase().includes(debouncedSearchTerm.toLowerCase()) ||
				model.tags.some((tag) => tag.toLowerCase().includes(debouncedSearchTerm.toLowerCase()));

			const matchesDevice =
				selectedDevices.length === 0 ||
				selectedDevices.some((device) => model.deviceSupport.includes(device));
			const matchesFamily =
				!selectedFamily ||
				model.displayName.toLowerCase().includes(selectedFamily.toLowerCase()) ||
				model.alias.toLowerCase().includes(selectedFamily.toLowerCase());
			const matchesAcceleration =
				!selectedAcceleration || model.acceleration === selectedAcceleration;

			return matchesSearch && matchesDevice && matchesFamily && matchesAcceleration;
		});

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
		selectedDevices = [];
		selectedFamily = '';
		selectedAcceleration = '';
		sortBy = 'name';
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

	function copyModelId(modelId: string) {
		navigator.clipboard.writeText(modelId);
		showCopyToast = true;
		
		if (copyToastTimer) {
			clearTimeout(copyToastTimer);
		}
		
		copyToastTimer = setTimeout(() => {
			showCopyToast = false;
		}, 3000);
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

	// Reactive filtering - automatically applies whenever allModels or filters change
	$: {
		if (allModels.length > 0) {
			applyFilters();
		} else {
			filteredModels = [];
		}
		// Watch all filter dependencies
		selectedDevices;
		selectedFamily;
		selectedAcceleration;
		debouncedSearchTerm;
		sortBy;
		sortOrder;
	}

	onMount(() => {
		fetchAllModels();
	});

	onDestroy(() => {
		if (searchDebounceTimer) {
			clearTimeout(searchDebounceTimer);
		}
		if (copyToastTimer) {
			clearTimeout(copyToastTimer);
		}
	});

	let description =
		'Discover and explore Foundry local models optimized for various hardware devices including NPUs, GPUs, CPUs, FPGAs and other specialized compute platforms.';
	let keywords =
		'foundry, local models, npu models, gpu models, cpu models, onnx runtime, machine learning models, ai models, hardware optimization';
</script>

<svelte:head>
	<title>Foundry Local Models - ONNX Runtime</title>
	<meta name="description" content={description} />
	<meta name="keywords" content={keywords} />
	<meta property="og:title" content="Foundry Local Models - ONNX Runtime" />
	<meta property="og:description" content={description} />
	<meta property="twitter:title" content="Foundry Local Models - ONNX Runtime" />
	<meta property="twitter:description" content={description} />
</svelte:head>

<div class="container mx-auto px-4 py-8">
	<div class="text-center mb-8">
		<h1 class="text-4xl font-bold mb-4">Foundry Local Models</h1>
		<p class="text-lg text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
			Explore a curated collection of AI models optimized for local deployment across various
			hardware platforms. Use the search and filters below to find models specifically designed for
			your NPUs, GPUs, and CPUs.
		</p>
	</div>

	<!-- Search and Filters Section -->
	<div class="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-8">
		<div class="mb-4">
			<h2 class="text-lg font-semibold mb-2">Filter & Search Models</h2>
			<p class="text-sm text-gray-600 dark:text-gray-400">
				Results update automatically as you type or change filters
			</p>
		</div>
		<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4 mb-4">
			<!-- Search -->
			<div class="lg:col-span-2">
				<label class="block text-sm font-medium mb-2" for="search-models">Search Models</label>
				<div class="relative">
					<input
						id="search-models"
						type="text"
						bind:value={searchTerm}
						placeholder="Search by name, description, or tags..."
						class="input input-bordered w-full"
					/>
					{#if searchTerm !== debouncedSearchTerm}
						<div class="absolute right-3 top-1/2 -translate-y-1/2">
							<span class="loading loading-xs loading-spinner" title="Searching..." />
						</div>
					{/if}
				</div>
			</div>

			<!-- Execution Device Filter (Multi-select buttons) -->
			<div class="flex flex-col">
				<div class="block text-sm font-medium mb-2">Filter by Execution Device</div>
				<div class="flex gap-2 flex-1" role="group" aria-label="Filter by Execution Device">
					{#each availableDevices as device}
						<button
							type="button"
							class="btn flex-1 h-full min-h-[3rem]"
							class:btn-primary={selectedDevices.includes(device)}
							class:btn-outline={!selectedDevices.includes(device)}
							on:click={() => toggleDevice(device)}
							aria-pressed={selectedDevices.includes(device)}
						>
							{device.toUpperCase()}
						</button>
					{/each}
				</div>
			</div>

			<!-- Acceleration Filter -->
			<div>
				<label class="block text-sm font-medium mb-2" for="acceleration-filter"
					>Filter by Acceleration</label
				>
				<select
					id="acceleration-filter"
					bind:value={selectedAcceleration}
					class="select select-bordered w-full"
				>
					<option value="">All Accelerations</option>
					{#each availableAccelerations as acceleration}
						<option value={acceleration}
							>{foundryModelService.getAccelerationDisplayName(acceleration)}</option
						>
					{/each}
				</select>
			</div>

			<!-- Model Family Filter -->
			<div>
				<label class="block text-sm font-medium mb-2" for="family-filter"
					>Filter by Model Family</label
				>
				<select
					id="family-filter"
					bind:value={selectedFamily}
					class="select select-bordered w-full"
				>
					<option value="">All Model Families</option>
					{#each availableFamilies as family}
						<option value={family}>{family.charAt(0).toUpperCase() + family.slice(1)}</option>
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
					{#if searchTerm !== debouncedSearchTerm}
						<span class="loading loading-xs loading-spinner mr-2" title="Applying filters..." />
						Filtering...
					{:else}
						{filteredModels.length} model{filteredModels.length !== 1 ? 's' : ''} found
					{/if}
				</span>
				{#if searchTerm || selectedDevices.length > 0 || selectedFamily || selectedAcceleration}
					<button on:click={clearFilters} class="btn btn-outline btn-sm"> Clear Filters </button>
				{/if}
				<!-- Reload button is now less prominent, only for refreshing data from server -->
				<div class="dropdown dropdown-end">
					<div tabindex="0" role="button" class="btn btn-ghost btn-xs">⋯</div>
					<ul class="dropdown-content menu bg-base-100 rounded-box z-[1] w-44 p-2 shadow">
						<li>
							<button on:click={refreshModels} disabled={loading}>
								{loading ? 'Loading...' : 'Refresh from server'}
							</button>
						</li>
					</ul>
				</div>
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
				<button class="btn btn-sm btn-outline" on:click={refreshModels}> Retry </button>
			</div>
		</div>
	{/if}

	<!-- Models Grid -->
	{#if !loading && !error}
		{#if paginatedModels.length > 0}
			<div
				class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8 transition-opacity duration-300"
				class:opacity-75={searchTerm !== debouncedSearchTerm}
			>
				{#each paginatedModels as model (model.alias)}
					<div
						class="card bg-base-100 shadow-xl hover:shadow-2xl transition-shadow duration-200 border border-gray-200 dark:border-gray-700 flex flex-col h-full"
					>
						<div class="card-body p-5 flex flex-col">
							<!-- Header with title and publisher -->
							<div class="mb-3">
								<h3 class="card-title text-lg font-semibold mb-1">{model.displayName}</h3>
								<p class="text-sm text-gray-500">
									{model.publisher.charAt(0).toUpperCase() + model.publisher.slice(1)}
								</p>
							</div>

							<!-- Description -->
							<p class="text-sm text-gray-600 dark:text-gray-300 mb-4 line-clamp-3 flex-grow">
								{model.description}
							</p>

							<!-- All badges in one row -->
							<div class="flex flex-wrap items-center gap-2 mb-4">
								<!-- Device support badges -->
								{#each model.deviceSupport as device, index}
									<span class="badge badge-primary badge-sm gap-1" title="{device} support">
										{getDeviceIcon(device)}
										{device.toUpperCase()}
									</span>
								{/each}
								
								{#if model.deviceSupport.length > 0 && (model.taskType || model.license)}
									<span class="text-gray-400">•</span>
								{/if}
								
								<!-- Task type badge -->
								{#if model.taskType}
									<span class="badge badge-secondary badge-sm"
										>{model.taskType.charAt(0).toUpperCase() + model.taskType.slice(1)}</span
									>
								{/if}
								
								{#if model.taskType && model.license}
									<span class="text-gray-400">•</span>
								{/if}
								
								<!-- License badge -->
								{#if model.license}
									<span class="badge badge-outline badge-sm gap-1">
										<svg class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
											<path d="M9 2a1 1 0 000 2h2a1 1 0 100-2H9z" />
											<path
												fill-rule="evenodd"
												d="M4 5a2 2 0 012-2h8a2 2 0 012 2v8a2 2 0 01-2 2H6a2 2 0 01-2-2V5zm3 3a1 1 0 000 2h6a1 1 0 100-2H7zm0 3a1 1 0 000 2h6a1 1 0 100-2H7z"
												clip-rule="evenodd"
											/>
										</svg>
										{model.license}
									</span>
								{/if}
							</div>

							<!-- Footer with metadata -->
							<div class="flex flex-col gap-2 text-xs text-gray-500 pt-3 border-t border-gray-200 dark:border-gray-700">
								<div class="flex items-center justify-between">
									<div class="flex items-center gap-2">
										<span class="badge badge-ghost badge-sm">v{model.latestVersion}</span>
										<span>•</span>
										<span>Updated {formatDate(model.lastModified)}</span>
									</div>
									<button
										class="btn btn-ghost btn-xs gap-1"
										on:click={() => copyModelId(model.alias)}
										aria-label="Copy model ID"
									>
										<svg class="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
											<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
										</svg>
										<span class="hidden sm:inline">Copy ID</span>
									</button>
								</div>
								{#if model.totalDownloads && model.totalDownloads > 0}
									<div class="flex items-center gap-1">
										<svg class="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
											<path fill-rule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clip-rule="evenodd" />
										</svg>
										<span>{model.totalDownloads.toLocaleString()} downloads</span>
									</div>
								{/if}
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

	<!-- Toast notification for copy action -->
	{#if showCopyToast}
		<div class="toast toast-top toast-center z-50">
			<div class="alert alert-success">
				<svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
					<path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7" />
				</svg>
				<span>Model ID copied to clipboard!</span>
			</div>
		</div>
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
