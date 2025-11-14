<script>
	let description =
		'ONNX Runtime Release Roadmap - find the latest release information for ONNX Runtime.';
	let keywords =
		'onnx runtime, onnx runtime roadmap, onnx runtime release, onnx runtime 1.23, onnx runtime 1.23.2, onnx runtime 1.24.0, onnx runtime 1.24';
	// Added release metadata variables
	const previousRelease = { version: '1.23.2', date: 'Oct 2025' };
	const inProgressRelease = { version: 'None', date: '' };
	const nextRelease = { version: '1.24', date: 'Jan 2026' };
</script>

<svelte:head>
	<!-- Dynamic meta tags -->
	<meta name="description" content={description} />
	<meta name="keywords" content={keywords} />
	<!-- Open Graph / Facebook -->
	<meta property="og:description" content={description} />

	<!-- Twitter -->
	<meta property="twitter:description" content={description} />
</svelte:head>

<div class="container mx-auto px-8">
	<h1 class="text-3xl">ONNX Runtime Release Roadmap</h1>
	<p>
		ONNX Runtime is released on a quarterly basis. Patch releases are published between major
		releases as necessary.
	</p>
	<div class="flex justify-center my-4">
		<div class="stats stats-vertical md:stats-horizontal shadow rounded-sm">
			<div class="stat">
				<div class="stat-figure">
					<svg
						class="stroke-success"
						xmlns="https://www.w3.org/2000/svg"
						width="32"
						height="32"
						viewBox="0 0 24 24"
						fill="none"
						stroke="#000000"
						stroke-width="2"
						stroke-linecap="round"
						stroke-linejoin="round"><polyline points="20 6 9 17 4 12" /></svg
					>
				</div>
				<div class="stat-title">Previous release</div>
				<div class="stat-value text-success">{previousRelease.version}</div>
				<div class="stat-desc">Release date: {previousRelease.date}</div>
			</div>

			{#if inProgressRelease.version && inProgressRelease.version !== 'None'}
			<div class="stat">
				<div class="stat-figure text-secondary">
					<svg
						xmlns="https://www.w3.org/2000/svg"
						width="24"
						height="24"
						class="stroke-warning"
						viewBox="0 0 24 24"
						fill="none"
						stroke-width="2"
						stroke-linecap="round"
						stroke-linejoin="round"
						><circle cx="12" cy="12" r="3" /><path
							d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"
						/></svg
					>
				</div>
				<div class="font-bold underline">In-Progress Release</div>
				<div class="stat-value text-warning">{inProgressRelease.version}</div>
				<div class="stat-desc">Release date: {inProgressRelease.date}</div>
			</div>
			{/if}

			<div class="stat">
				<div class="stat-figure text-primary">
					<svg
						xmlns="https://www.w3.org/2000/svg"
						fill="none"
						viewBox="0 0 24 24"
						class="stroke-primary"
						width="28"
						height="28"
					>
						<path
							stroke-linecap="round"
							stroke-linejoin="round"
							stroke-width="2"
							d="M6 2h12M6 22h12M8 2v6l4 4-4 4v6M16 2v6l-4 4 4 4v6"
						/>
					</svg>
				</div>
				<div class="stat-title">Next release</div>
				<div class="stat-value text-primary">{nextRelease.version}</div>
				<div class="stat-desc">Release date: {nextRelease.date}</div>
			</div>
		</div>
	</div>

	<h2 class="text-xl font-bold mt-2">Announcements</h2>
	<ul class="list-disc ml-8">
		<li>
			<strong>ONNX Runtime {previousRelease.version}  now officially released.</strong> For release notes, assets,
			and more, visit our
			<a
				href="https://github.com/microsoft/onnxruntime/releases/tag/v{previousRelease.version}"
				class="text-blue-600 underline">GitHub Releases page</a
			>.
		</li>
	</ul>
{#if false}
	<h2 class="text-xl font-bold mt-2">Versioning Updates</h2>
	<p class="font-thin">
		We are planning to upgrade ONNX Runtime support for the following (where the first value is the
		highest version previously supported and the second value is the version support that will be
		added in ORT 1.20.1):
	</p>
	<ul class="list-disc ml-8">
		<li>QNN SDK 2.27 --> 2.28</li>
		<li>DirectML 1.15.2 --> 1.16</li>
		<li>ONNX 1.17 support will be included in a future release.</li>
	</ul>

	<h2 class="text-xl font-bold mt-2">Major Updates</h2>
	<p class="font-thin">
		In addition to various bug fixes and performance improvements, ORT 1.20.1 will include the
		following updates:
	</p>
	<ul class="list-disc ml-8">
		<li>
			CPU FP16 implementation fixes for the following kernels: LayerNormalization,
			SimplifiedLayerNormalization, SkipLayerNormalization, SkipSimplifiedLayerNormalization.
		</li>
		<li>Python quantization tool updates.</li>
		<li>New QNN SDK version support.</li>
	</ul>
{/if}
	<h2 class="text-xl font-bold mt-2">Feature Requests</h2>
	<p class="font-thin">
		To request new ONNX Runtime features to be included in a future release, please submit a feature
		request through <a
			href="https://github.com/microsoft/onnxruntime/issues/new?assignees=&labels=feature+request&projects=&template=07-feature_request.yml&title=%5BFeature+Request%5D+"
			class="text-blue-600 underline">GitHub Issues</a
		>
		or through
		<a
			href="https://github.com/microsoft/onnxruntime/discussions/new?category=ideas-feature-requests"
			class="text-blue-600 underline">GitHub Discussions</a
		>.
	</p>
	<p class="font-thin">To ensure that your request is addressed as quickly as possible, please:</p>
	<ul class="list-disc ml-8">
		<li>Include a detailed title.</li>
		<li>
			Provide as much detail as possible in the body of your request (e.g., use case for the
			feature, the platform(s) or EP(s) this feature is needed for, etc.).
		</li>
		<li>
			Apply a label corresponding to the appropriate ONNX Runtime area (e.g., "platform:mobile",
			"platform:web", "ep:CUDA", etc.) if you know it.
		</li>
	</ul>
	<p class="font-thin">
		<em>Note: All timelines and features listed on this page are subject to change.</em>
	</p>
	<div class="divider" />
	
	{#if nextRelease.version && nextRelease.version !== 'None'}
	<h2 class="text-xl font-bold mt-2">ONNX Runtime {nextRelease.version}</h2>
	<p class="font-thin">
		<strong>Tentative release date:</strong> {nextRelease.date}
	</p>

	<div class="join join-vertical w-full p-2">
		<!-- Key Stories Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="key-stories-124" />
			<div class="collapse-title text-xl font-bold">Key Stories</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li>Feature complete Plugin EP</li>
					<li>Memory Optimization</li>
					<li>Low-bit Precision</li>
				</ul>
			</div>
		</div>

		<!-- Core Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="core-124" />
			<div class="collapse-title text-xl font-bold">Core</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li>DirectX Inter-operability</li>
					<li>ARM Improvements</li>
					<li>2-bit precision</li>
					<li>Weightless EP Context</li>
					<li>Graph partitioning improvements</li>
				</ul>
			</div>
		</div>

		<!-- Plugin EP Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="plugin-ep-124" />
			<div class="collapse-title text-xl font-bold">Plugin EP</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li>Offline Compilation</li>
					<li>Kernel Registry EP</li>
					<li>Language Bindings</li>
					<li>Testsuite and tooling</li>
				</ul>
			</div>
		</div>

		<!-- EP Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="ep-124" />
			<div class="collapse-title text-xl font-bold">EP</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li>WebGPU Improvements</li>
				</ul>
			</div>
		</div>

		<!-- TRT EP Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="trt-ep-124" />
			<div class="collapse-title text-xl font-bold">TRT EP</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li>Update TRT version used by the TRT EP</li>
				</ul>
			</div>
		</div>

		<!-- ONNX Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="onnx-124" />
			<div class="collapse-title text-xl font-bold">ONNX</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li>Update ONNXruntime support for ONNX Opset/IR (TBD)</li>
				</ul>
			</div>
		</div>
	</div>
	{/if}
</div>
