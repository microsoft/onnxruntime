<script>
	let description =
		'ONNX Runtime Release Roadmap - find the latest release information for ONNX Runtime.';
	let keywords =
		'onnx runtime, onnx runtime roadmap, onnx runtime release, onnx runtime 1.21, onnx runtime 1.22, onnx runtime 1.23';
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
				<div class="stat-value text-success">1.21.0</div>
				<div class="stat-desc">Release date: Feb 2025</div>
			</div>
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
				<div class="font-bold underline">Current Release</div>
				<div class="stat-value text-warning">1.22.0</div>
				<div class="stat-desc">Release date: May 2025</div>
			</div>
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
				<div class="stat-value text-primary">1.23</div>
				<div class="stat-desc">Release date: Aug 2025</div>
			</div>
		</div>
	</div>

	<h2 class="text-xl font-bold mt-2">Announcements</h2>
	<ul class="list-disc ml-8">
		<li>
			There will be <strong>breaking</strong> API changes in this release. We'll keep you posted!
		</li>
	</ul>

	<h2 class="text-xl font-bold mt-2">Major Updates in 1.22</h2>
	<p class="font-thin">
		The current release (1.22.0) includes the following key features and updates:
	</p>
	<ul class="list-disc ml-8">
		<li><strong>ORT API changes</strong> - Improved API interface for better usability</li>
		<li><strong>New WebGPU Execution Provider</strong> - Enhanced support for web-based ML workloads</li>
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
	<div class="divider" />
	<h2 class="text-xl font-bold mt-2">ONNX Runtime 1.21</h2>
	<p class="font-thin">
		<strong>Release date:</strong> February 2025
	</p>

	<div class="join join-vertical w-full p-2">
		<!-- Announcements Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="announcements-1.21" />
			<div class="collapse-title text-xl font-bold">Announcements</div>
			<div class="collapse-content">
				<p class="font-thin">No large announcements of note this release! We've made a lot of small refinements to streamline your ONNX Runtime experience.</p>
			</div>
		</div>

		<!-- GenAI & Advanced Model Features -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="genai-1.21" />
			<div class="collapse-title text-xl font-bold">GenAI & Advanced Model Features</div>
			<div class="collapse-content">
				<h3 class="text-lg font-semibold">Enhanced Decoding & Pipeline Support</h3>
				<ul class="list-disc ml-8">
					<li>Added "chat mode" support for CPU, GPU, and WebGPU.</li>
					<li>Provided support for decoder model pipelines.</li>
					<li>Added support for Java API for MultiLoRA.</li>
				</ul>
				<h3 class="text-lg font-semibold">API & Compatibility Updates</h3>
				<ul class="list-disc ml-8">
					<li>Chat mode introduced breaking changes in the API (see migration guide).</li>
				</ul>
				<h3 class="text-lg font-semibold">Bug Fixes for Model Output</h3>
				<ul class="list-disc ml-8">
					<li>Fixed Phi series garbage output issues with long prompts.</li>
					<li>Resolved gibberish issues with top_k on CPU.</li>
				</ul>
			</div>
		</div>

		<!-- Core & Execution Optimizations -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="core-1.21" />
			<div class="collapse-title text-xl font-bold">Core Refinements</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li>Reduced default logger usage for improved efficiency (<a href="https://github.com/microsoft/onnxruntime/pull/23030" class="text-blue-600 underline">#23030</a>).</li>
					<li>Fixed a visibility issue in threadpool (<a href="https://github.com/microsoft/onnxruntime/pull/23098" class="text-blue-600 underline">#23098</a>).</li>
				</ul>
			</div>
		</div>

		<!-- Execution Providers Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="eps-1.21" />
			<div class="collapse-title text-xl font-bold">Execution Provider (EP) Updates</div>
			<div class="collapse-content">
				<h3 class="text-lg font-semibold">General</h3>
				<ul class="list-disc ml-8">
					<li>Removed TVM EP from the source tree (<a href="https://github.com/microsoft/onnxruntime/pull/22827" class="text-blue-600 underline">#22827</a>).</li>
					<li>Marked NNAPI EP for deprecation (following Google's deprecation of NNAPI).</li>
					<li>Fixed a DLL delay loading issue that impacts WebGPU EP and DirectML EP's usability on Windows (<a href="https://github.com/microsoft/onnxruntime/pull/23111" class="text-blue-600 underline">#23111</a>, <a href="https://github.com/microsoft/onnxruntime/pull/23227" class="text-blue-600 underline">#23227</a>)</li>
				</ul>
				
				<h3 class="text-lg font-semibold">TensorRT EP</h3>
				<ul class="list-disc ml-8">
					<li>Added support for TensorRT 10.8.</li>
					<li>onnx-tensorrt open-source parser user: please check documentation for requirements.</li>
					<li>Assigned DDS ops (NMS, RoiAlign, NonZero) to TensorRT by default.</li>
					<li>Introduced option trt_op_types_to_exclude to exclude specific ops from TensorRT assignment.</li>
				</ul>
				
				<h3 class="text-lg font-semibold">CUDA EP</h3>
				<ul class="list-disc ml-8">
					<li>Added a python API preload_dlls to coexist with PyTorch.</li>
					<li>Miscellaneous enhancements for Flux model inference.</li>
				</ul>
				
				<h3 class="text-lg font-semibold">QNN EP</h3>
				<ul class="list-disc ml-8">
					<li>Introduced QNN shared memory support.</li>
					<li>Improved performance for AI Hub models.</li>
					<li>Added support for QAIRT/QNN SDK 2.31.</li>
					<li>Added Python 3.13 package.</li>
					<li>QNN EP is now built as a shared library/DLL by default. To retain previous build behavior, use build option --use_qnn static_lib.</li>
				</ul>
				
				<h3 class="text-lg font-semibold">DirectML EP</h3>
				<ul class="list-disc ml-8">
					<li>Updated DirectML version from 1.15.2 to 1.15.4 (<a href="https://github.com/microsoft/onnxruntime/pull/22635" class="text-blue-600 underline">#22635</a>).</li>
				</ul>
				
				<h3 class="text-lg font-semibold">OpenVINO EP</h3>
				<ul class="list-disc ml-8">
					<li>Introduced OpenVINO EP Weights Sharing feature.</li>
					<li>Added support for various contrib Ops in OVEP:
						<ul class="list-disc ml-8">
							<li>SkipLayerNormalization, MatMulNBits, FusedGemm, FusedConv, EmbedLayerNormalization, BiasGelu, Attention, DynamicQuantizeMatMul, FusedMatMul, QuickGelu, SkipSimplifiedLayerNormalization</li>
						</ul>
					</li>
				</ul>
				
				<h3 class="text-lg font-semibold">VitisAI EP</h3>
				<ul class="list-disc ml-8">
					<li>Miscellaneous bug fixes and improvements.</li>
				</ul>
			</div>
		</div>

		<!-- Mobile Platform Enhancements -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="mobile-1.21" />
			<div class="collapse-title text-xl font-bold">Mobile Platform Enhancements</div>
			<div class="collapse-content">
				<h3 class="text-lg font-semibold">CoreML Updates</h3>
				<ul class="list-disc ml-8">
					<li>Added support for caching generated CoreML models.</li>
				</ul>
			</div>
		</div>

		<!-- Extensions & Tokenizer -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="extensions-1.21" />
			<div class="collapse-title text-xl font-bold">Extensions & Tokenizer Improvements</div>
			<div class="collapse-content">
				<h3 class="text-lg font-semibold">Expanded Tokenizer Support</h3>
				<ul class="list-disc ml-8">
					<li>Now supports more tokenizer models, including ChatGLM, Baichuan2, Phi-4, etc.</li>
					<li>Added full Phi-4 pre/post-processing support for text, vision, and audio.</li>
					<li>Introduced RegEx pattern loading from tokenizer.json.</li>
				</ul>
				
				<h3 class="text-lg font-semibold">Image Codec Enhancements</h3>
				<ul class="list-disc ml-8">
					<li>ImageCodec now links to native APIs if available; otherwise, falls back to built-in libraries.</li>
				</ul>
				
				<h3 class="text-lg font-semibold">Unified Tokenizer API</h3>
				<ul class="list-disc ml-8">
					<li>Introduced a new tokenizer op schema to unify the tokenizer codebase.</li>
					<li>Added support for loading tokenizer data from a memory blob in the C API.</li>
				</ul>
			</div>
		</div>

		<!-- Infrastructure & Build -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="build-1.21" />
			<div class="collapse-title text-xl font-bold">Infrastructure & Build Improvements</div>
			<div class="collapse-content">
				<h3 class="text-lg font-semibold">CMake File Changes</h3>
				<ul class="list-disc ml-8">
					<li>CMake Version: Increased the minimum required CMake version from 3.26 to 3.28. Added support for CMake 4.0.</li>
					<li>Python Version: Increased the minimum required Python version from 3.8 to 3.10 for building ONNX Runtime from source.</li>
					<li>Improved VCPKG support</li>
					<li>Added options for WebGPU EP:
						<ul class="list-disc ml-8">
							<li>onnxruntime_USE_EXTERNAL_DAWN</li>
							<li>onnxruntime_CUSTOM_DAWN_SRC_PATH</li>
							<li>onnxruntime_BUILD_DAWN_MONOLITHIC_LIBRARY</li>
							<li>onnxruntime_ENABLE_PIX_FOR_WEBGPU_EP</li>
							<li>onnxruntime_ENABLE_DAWN_BACKEND_VULKAN</li>
							<li>onnxruntime_ENABLE_DAWN_BACKEND_D3D12</li>
						</ul>
					</li>
					<li>Added cmake option onnxruntime_BUILD_QNN_EP_STATIC_LIB for building with QNN EP as a static library.</li>
					<li>Removed cmake option onnxruntime_USE_PREINSTALLED_EIGEN.</li>
					<li>Fixed a build issue with Visual Studio 2022 17.3 (<a href="https://github.com/microsoft/onnxruntime/pull/23911" class="text-blue-600 underline">#23911</a>)</li>
				</ul>
				
				<h3 class="text-lg font-semibold">Modernized Build Tools</h3>
				<ul class="list-disc ml-8">
					<li>Now using VCPKG for most package builds.</li>
					<li>Upgraded Gradle from 7.x to 8.x.</li>
					<li>Updated JDK from 11 to 17.</li>
					<li>Enabled onnxruntime_USE_CUDA_NHWC_OPS by default for CUDA builds.</li>
					<li>Added support for WASM64 (build from source; no package published).</li>
				</ul>
				
				<h3 class="text-lg font-semibold">Dependency Cleanup</h3>
				<ul class="list-disc ml-8">
					<li>Removed Google's nsync from dependencies.</li>
				</ul>
				
				<h3 class="text-lg font-semibold">Others</h3>
				<ul class="list-disc ml-8">
					<li>Updated Node.js installation script to support network proxy usage (<a href="https://github.com/microsoft/onnxruntime/pull/23231" class="text-blue-600 underline">#23231</a>)</li>
				</ul>
			</div>
		</div>

		<!-- Web -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="web-1.21" />
			<div class="collapse-title text-xl font-bold">Web</div>
			<div class="collapse-content">
				<p class="font-thin">No updates of note.</p>
			</div>
		</div>
	</div>
</div> <!-- Closing the main container div -->
