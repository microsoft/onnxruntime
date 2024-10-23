<script>
	let description = 'ONNX Runtime Release Roadmap - find the latest release information for ONNX Runtime.';
	let keywords = 'onnx runtime, onnx runtime roadmap, onnx runtime release, onnx runtime 1.20';
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
				<div class="stat-value text-success">1.19.2</div>
				<div class="stat-desc">Release date: 9/4/2024</div>
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
				<div class="stat-value text-warning">1.20</div>
				<div class="stat-desc">Release date: 10/30/2024</div>
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
		<li><strong>All ONNX Runtime Training packages have been deprecated.</strong> ORT 1.19.2 was the last
		release for which onnxruntime-training (PyPI), onnxruntime-training-cpu (PyPI), Microsoft.ML.OnnxRuntime.Training
		(Nuget), onnxruntime-training-c (CocoaPods), onnxruntime-training-objc (CocoaPods), and onnxruntime-training-android
		(Maven Central) were published.</li>
		<li><strong>ONNX Runtime packages will stop supporting Python 3.8 and Python 3.9.</strong> This decision aligns with 
		NumPy Python version support. To continue using ORT with Python 3.8 and Python 3.9, you can use ORT 1.19.2 and earlier.</li>
		<li><strong>ONNX Runtime 1.20 CUDA packages will include new dependencies that were not required in 1.19 packages.</strong>
		The following dependencies are new: libcudnn_adv.so.9, libcudnn_cnn.so.9, libcudnn_engines_precompiled.so.9,
		libcudnn_engines_runtime_compiled.so.9, libcudnn_graph.so.9, libcudnn_heuristic.so.9, libcudnn_ops.so.9, libnvrtc.so.12, 
		and libz.so.1.</li>
	</ul>
	<h2 class="text-xl font-bold mt-2">New Packages</h2>
	<p class="font-thin">We are planning to start releasing the following packages:</p>
	<ul class="list-disc ml-8">
		<li>Maven package with Android support for QNN EP</li>
		<li>CocoaPods package with Mac / iOS support for ORT GenAI</li>
	</ul>

	<h2 class="text-xl font-bold mt-2">Versioning Updates</h2>
	<p class="font-thin">
		We are planning to upgrade ONNX Runtime support for the following (where the first value is the
		highest version previously supported and the second value is the version support that will be
		added in ORT 1.20):
	</p>
	<ul class="list-disc ml-8">
		<li>TensorRT 10.2 --> 10.4</li>
		<li>DirectML 1.15.1 --> 1.15.2</li>
		<li>Python 3.13 support will also be added.</li>
		<li>ONNX 1.17 support will be included in a future release.</li>
	</ul>

	<h2 class="text-xl font-bold mt-2">Major Updates</h2>
	<p class="font-thin">
		In addition to various bug fixes and performance improvements, ORT 1.20 will include the
		following major updates:
	</p>
	<ul class="list-disc ml-8">
		<li>Add MultiLoRA support.</li>
		<li>Improve CPU FP16 and INT4 performance.</li>
		<li>
			Increase GenAI API model support, including Whisper, Phi-3.5-vision multi-frame, and more.
		</li>
		<li>Publish Phi-3.5 ONNX model variants to Hugging Face.</li>
		<li>
			Expand mobile support to include GPU EP and FP16 support for CoreML EP and XNNPACK kernels.
		</li>
		<li>Add Apple support for AI Toolkit for VS Code.</li>
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
	<h2 class="text-xl font-bold mt-2">ONNX Runtime 1.20</h2>
	<p class="font-thin">
		<strong>Tentative release date:</strong> 10/30/2024
	</p>

	<div class="join join-vertical w-full p-2">
		<!-- Announcements Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="announcements" />
			<div class="collapse-title text-xl font-bold">Announcements</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li><strong>All ONNX Runtime Training packages have been deprecated.</strong> ORT 1.19.2 was the last
					release for which onnxruntime-training (PyPI), onnxruntime-training-cpu (PyPI), Microsoft.ML.OnnxRuntime.Training
					(Nuget), onnxruntime-training-c (CocoaPods), onnxruntime-training-objc (CocoaPods), and onnxruntime-training-android
					(Maven Central) were published.</li>
					<li><strong>ONNX Runtime packages will stop supporting Python 3.8 and Python 3.9.</strong> This decision aligns with 
					NumPy Python version support. To continue using ORT with Python 3.8 and Python 3.9, you can use ORT 1.19.2 and earlier.</li>
					<li><strong>ONNX Runtime 1.20 CUDA packages will include new dependencies that were not required in 1.19 packages.</strong>
					The following dependencies are new: libcudnn_adv.so.9, libcudnn_cnn.so.9, libcudnn_engines_precompiled.so.9,
					libcudnn_engines_runtime_compiled.so.9, libcudnn_graph.so.9, libcudnn_heuristic.so.9, libcudnn_ops.so.9, libnvrtc.so.12, 
					and libz.so.1.</li>
				</ul>
			</div>
		</div>

		<!-- Build System & Packages Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="build" />
			<div class="collapse-title text-xl font-bold">Build System & Packages</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li>Python 3.13 support is included in PyPI packages.</li>
					<li>ONNX 1.17 support will be delayed until a future release, but the ONNX version used by ONNX Runtime has been patched to include a shape inference change to the Einsum op.</li>
					<li>DLLs in the Maven build are now digitally signed.</li>
					<li>(Experimental) vcpkg support added for the CPU EP. The DML EP does not yet support vcpkg, and other EPs have not been tested.</li>
				</ul>
			</div>
		</div>

		<!-- Core Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="core" />
			<div class="collapse-title text-xl font-bold">Core</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li>MultiLoRA support.</li>
					<li>Memory utilization (specifically related to external weights) and partitioning improvements.</li>
				</ul>
			</div>
		</div>

		<!-- Performance Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="performance" />
			<div class="collapse-title text-xl font-bold">Performance</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li>FP16 SLM model support on CPU EP.</li>
					<li>INT4 quantized embedding support on CPU and CUDA EPs.</li>
				</ul>
			</div>
		</div>

		<!-- EPs Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="eps" />
			<div class="collapse-title text-xl font-bold">EPs</div>
			<div class="collapse-content">
				<h3 class="text-lg font-semibold">TensorRT</h3>
				<ul class="list-disc ml-8">
					<li>TensorRT 10.4 support.</li>
					<li>DDS enablement and performance improvements for NMS.</li>
				</ul>
				<h3 class="text-lg font-semibold">QNN</h3>
				<ul class="list-disc ml-8">
					<li>HTP shared weights context binary (offline tool).</li>
					<li>Runtime support for QNN HTP shared weights in multiple ORT sessions.</li>
					<li>Efficient mode support.</li>
				</ul>
				<h3 class="text-lg font-semibold">OpenVINO</h3>
				<ul class="list-disc ml-8">
					<li>Context generation memory optimizations.</li>
					<li>Efficient mode support.</li>
				</ul>
				<h3 class="text-lg font-semibold">DirectML</h3>
				<ul class="list-disc ml-8">
					<li>DirectML 1.15.2 support.</li>
				</ul>
			</div>
		</div>

		<!-- Mobile Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="mobile" />
			<div class="collapse-title text-xl font-bold">Mobile</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li>Android QNN support, including a pre-built Maven package, performance improvements, and Phi-3 model support.</li>
					<li>Mobile GPU EP for support.</li>
					<li>FP16 support for CoreML EP and XNNPACK kernels.</li>
				</ul>
			</div>
		</div>

		<!-- Web Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="web" />
			<div class="collapse-title text-xl font-bold">Web</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li>Quantized embedding support.</li>
					<li>On-demand weight loading support (offloads Wasm32 heap and enables 8B-parameter LLMs).</li>
					<li>wasm64 support (available in custom builds but not included in released packages).</li>
					<li>GQA support.</li>
					<li>Integrated Intel GPU performance improvements.</li>
					<li>Opset-21 support (Reshape, Shape, Gelu).</li>
				</ul>
			</div>
		</div>

		<!-- GenAI Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="genai" />
			<div class="collapse-title text-xl font-bold">GenAI</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li>Continuous decoding support, including chat mode and system prompt caching.</li>
					<li>MultiLoRA API.</li>
					<li>Additional model support, including Whisper, Phi-3.5 Vision Multi-Frame, and Qualcomm NPU support for Phi-3.5 and Llama-3.1.</li>
					<li>Mac/iOS support available in pre-built packages.</li>
				</ul>
			</div>
		</div>

		<!-- Extensions Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="extensions" />
			<div class="collapse-title text-xl font-bold">Extensions</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li>Tokenization performance improvements.</li>
					<li>Additional multi-modal model support (CLIP and Mllama), including more kernel attributes.</li>
					<li>Unigram tokenization model support.</li>
					<li>OpenCV dependency removed from C API build.</li>
				</ul>
				<p class="font-thin">
					Full release notes for ONNX Runtime Extensions v0.13 will be found <a
					href="https://github.com/microsoft/onnxruntime-extensions/releases"
					class="text-blue-600 underline">here</a> once they are available (10/30 target).
				</p>
			</div>
		</div>

		<!-- Olive Section -->
		<div class="collapse collapse-arrow join-item border-base-300 border">
			<input type="checkbox" name="olive" />
			<div class="collapse-title text-xl font-bold">Olive</div>
			<div class="collapse-content">
				<ul class="list-disc ml-8">
					<li>Olive command line interface (CLI) now available with support to execute well-defined, concrete workflows without the need to create or edit configs manually.</li>
					<li>Additional improvements, including support for YAML-based workflow configs, streamlined DataConfig management, simplified workflow configuration, and more.</li>
					<li>Llama and Phi-3 model updates, including an updated MultiLoRA example using the ORT generate() API.</li>
				</ul>
				<p class="font-thin">
					Full release notes for Olive v0.7.0 can be found <a
					href="https://github.com/microsoft/Olive/releases/"
					class="text-blue-600 underline">here</a>.
				</p>
			</div>
		</div>
	</div>
</div>
