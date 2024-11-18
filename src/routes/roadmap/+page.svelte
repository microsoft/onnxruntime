<script>
	let description = 'ONNX Runtime Release Roadmap - find the latest release information for ONNX Runtime.';
	let keywords = 'onnx runtime, onnx runtime roadmap, onnx runtime release, onnx runtime 1.20, onnx runtime 1.21, onnx runtime 1.20.1';
</script>
<svelte:head>
	<!-- Dynamic meta tags -->
	<meta name="description" content={description} />
	<meta name="keywords" content={keywords} />
	<!-- Open Graph / Facebook -->
	<meta property="og:description" content={description}/>
	
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
						xmlns="http://www.w3.org/2000/svg"
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
				<div class="stat-value text-success">1.20.0</div>
				<div class="stat-desc">Release date: 11/1/2024</div>
			</div>

			<div class="stat">
				<div class="stat-figure text-secondary">
					<svg
						xmlns="http://www.w3.org/2000/svg"
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
				<div class="stat-value text-warning">1.20.1</div>
				<div class="stat-desc">Release date: 11/20/2024</div>
			</div>

			<div class="stat">
				<div class="stat-figure text-primary">
					<svg
						xmlns="http://www.w3.org/2000/svg"
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
				<div class="stat-value text-primary">1.21</div>
				<div class="stat-desc">Release date: Feb. 2025</div>
			</div>
		</div>
	</div>

	<h2 class="text-xl font-bold mt-2">Announcements</h2>
	<ul class="list-disc ml-8">
		<li><strong>The onnxruntime-gpu v1.10.0 will be removed from PyPI.</strong> We have hit our PyPI project size limit for onnxruntime-gpu,
		so we will be removing our oldest package version to free up the necessary space.</li>
		<li><strong>ONNX Runtime v1.20.0 is now officially released.</strong> For release notes, assets, and more, visit our <a
			href="https://github.com/microsoft/onnxruntime/releases/tag/v1.20.0"
			class="text-blue-600 underline">GitHub Releases page</a
		>.</li>
	</ul>

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
		<li>CPU FP16 implementation fixes for the following kernels: LayerNormalization, SimplifiedLayerNormalization, SkipLayerNormalization, SkipSimplifiedLayerNormalization.</li>
		<li>Python quantization tool updates.</li>
		<li>New QNN SDK version support.</li>
	</ul>

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
	<div class="divider"></div>
	<h2 class="text-xl font-bold mt-2">ONNX Runtime 1.20.1</h2>
	<p class="font-thin">
		<strong>Tentative release date:</strong> 11/20/2024
	</p>

	<div class="join join-vertical w-full p-2">
		<!-- Announcements Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="announcements" />
			<div class="collapse-title text-xl font-bold">Announcements</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li><strong>The onnxruntime-gpu v1.10.0 will be removed from PyPI.</strong> We have hit our PyPI project size limit for onnxruntime-gpu,
						so we will be removing our oldest package version to free up the necessary space.
					</li>
				</ul>
			</div>
		</div>

		<!-- Build System & Packages Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="build" />
			<div class="collapse-title text-xl font-bold">Build System & Packages</div>
			<div class="collapse-content">
				<p class="font-thin">
					No features planned for 1.20.1. Stay tuned for 1.21 features.
				</p>
			</div>
		</div>

		<!-- Core Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="core" />
			<div class="collapse-title text-xl font-bold">Core</div>
			<div class="collapse-content">
				<p class="font-thin">
					No features planned for 1.20.1. Stay tuned for 1.21 features.
				</p>
			</div>
		</div>

		<!-- Performance Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="performance" />
			<div class="collapse-title text-xl font-bold">Performance</div>
			<div class="collapse-content">
				<p class="font-thin">
					No features planned for 1.20.1. Stay tuned for 1.21 features.
				</p>
			</div>
		</div>

		<!-- Quantization Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="quantization" />
			<div class="collapse-title text-xl font-bold">Quantization</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li>Introduce get_int_qdq_config() helper to get QDQ configurations (<a
						href="https://github.com/microsoft/onnxruntime/pull/22677"
						class="text-blue-600 underline">#22677</a
					>).</li>
					<li>Update QDQ Pad, Slice, Softmax (<a
						href="https://github.com/microsoft/onnxruntime/pull/22676"
						class="text-blue-600 underline">#22676</a
					>).</li>
					<li>Handle input models with pre-quantized weights (<a
						href="https://github.com/microsoft/onnxruntime/pull/22633"
						class="text-blue-600 underline">#22633</a
					>).</li>
					<li>Prevent int32 quantized bias from clipping by adjusting the weight's scale (<a
						href="https://github.com/microsoft/onnxruntime/pull/22020"
						class="text-blue-600 underline">#22020</a
					>).</li>
				</ul>
			</div>
		</div>

		<!-- EPs Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="eps" />
			<div class="collapse-title text-xl font-bold">EPs</div>
			<div class="collapse-content">
				<h3 class="text-lg font-semibold">CPU</h3>
				<ul class="list-disc ml-8">
					<li>Fix CPU FP16 implementations for the following kernels: LayerNormalization, SimplifiedLayerNormalization, SkipLayerNormalization, SkipSimplifiedLayerNormalization.</li>
				</ul>
				<h3 class="text-lg font-semibold">QNN</h3>
				<ul class="list-disc ml-8">
					<li>QNN SDK 2.28.x support.</li>
				</ul>
				<h3 class="text-lg font-semibold">DirectML</h3>
				<ul class="list-disc ml-8">
					<li>DirectML 1.16 support.</li>
				</ul>
			</div>
		</div>

		<!-- Mobile Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="mobile" />
			<div class="collapse-title text-xl font-bold">Mobile</div>
			<div class="collapse-content">
				<p class="font-thin">
					No features planned for 1.20.1. Stay tuned for 1.21 features.
				</p>
			</div>
		</div>

		<!-- Web Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="web" />
			<div class="collapse-title text-xl font-bold">Web</div>
			<div class="collapse-content">
				<p class="font-thin">
					No features planned for 1.20.1. Stay tuned for 1.21 features.
				</p>
			</div>
		</div>

		<!-- generate() API Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="generate" />
			<div class="collapse-title text-xl font-bold">generate() API</div>
			<div class="collapse-content">
				<p class="font-thin">
					No features planned for 1.20.1. Stay tuned for 1.21 features.
				</p>
			</div>
		</div>

		<!-- Extensions Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="extensions" />
			<div class="collapse-title text-xl font-bold">Extensions</div>
			<div class="collapse-content">
				<p class="font-thin">
					No features planned for 1.20.1. Stay tuned for 1.21 features.
				</p>
			</div>
		</div>

		<!-- Olive Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="olive" />
			<div class="collapse-title text-xl font-bold">Olive</div>
			<div class="collapse-content">
				<p class="font-thin">
					No features planned for 1.20.1. Stay tuned for 1.21 features.
				</p>
			</div>
		</div>
	</div>
</div>
