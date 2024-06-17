<script>
	import windowsdevkit from '../../images/windowsdevkit.png';
	let description = 'Windows Dev Kit 2023, aka Project Volterra, enables developers to build apps that unlock the power of the NPU hardware to accelerate AI/ML workloads.'
	let image = 'https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png'
	let imageSquare = 'https://i.ibb.co/0YBy62j/ORT-icon-for-light-bg.png'
	let authors = ['']
	let keywords = 'onnxruntime, onnx runtime windows, onnx runtime windows models, onnx runtime windows deployment, onnx runtime windows performance, onnx runtime windows time to market, onnx runtime windows deploy anywhere, onnx runtime windows boost performance, onnx runtime windows improve time to market, onnx runtime windows production ready, onnx runtime windows lower latency, onnx runtime windows higher throughput, onnx runtime windows get innovations into production faster, onnx runtime windows testimonials, onnx runtime windows performance enhancements, onnx runtime windows production ready, onnx runtime windows lower latency, onnx runtime windows higher throughput, onnx runtime windows get innovations into production faster, onnx runtime windows performance enhancements'
</script>
<svelte:head>
	<!-- Dynamic meta tags -->
	<meta name="description" content={description} />
	<meta name="image" content={image} />
	<meta name="author" content={authors.join(', ')} />
	<meta name="keywords" content={keywords} />
	<!-- Open Graph / Facebook -->
	<meta property="og:description" content={description}/>
	<meta property="og:image" content={image} />
	
	<!-- Twitter -->
	<meta property="twitter:description" content={description} />
	<meta property="twitter:image" content={image} />
	<meta property="twitter:card" content={imageSquare} />
</svelte:head>

<div class="container mx-auto px-10">
	<h1 class="text-4xl mb-4">ONNX Runtime + Windows Dev Kit 2023 = NPU powered AI</h1>
	<h2 class="text-2xl mb-2">Delivering NPU powered AI capabilities in your apps</h2>
	<p>
		Windows Dev Kit 2023, aka Project Volterra, enables developers to build apps that unlock the
		power of the NPU hardware to accelerate AI/ML workloads delivering AI-enhanced features &
		experiences without compromising app performance. You can get started now and access the power
		of the NPU through the open source and cross-platform ONNX Runtime inference engine making it
		easy to run AI/ML models from popular machine learning frameworks like PyTorch and TensorFlow.
	</p>
	<div class="divider" />
	<div class="grid grid-cols-3 gap-4">
		<div class="md:col-span-2 col-span-3">
			<h2 class="text-xl text-blue-700">Get started on your Windows Dev Kit 2023 today</h2>
			Follow these steps to setup your device to use ONNX Runtime (ORT) with the built in NPU:
			<ol class="list-decimal ml-10">
				<li>
					<a
						class="text-blue-700"
						href="https://qpm.qualcomm.com/main/tools/details/qualcomm_ai_engine_direct">Download</a
					> the Qualcomm AI Engine Direct SDK (QNN SDK)
				</li>
				<li>
					<a
						class="text-blue-700"
						href="https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.QNN">Download</a
					> and install the ONNX Runtime with QNN package
				</li>
				<li>Start using the ONNX Runtime API in your application.</li>
			</ol>
			<br /><br />
			<p class="text-xl text-blue-700">Optimizing models for the NPU</p>
			<a class="text-blue-700" href="https://onnx.ai/">ONNX</a> is a standard format for
			representing ML models authored in frameworks like PyTorch, TensorFlow, and others. ONNX
			Runtime can run any ONNX model, however to make use of the NPU, you currently need to quantize
			the ONNX model to QDQ model.
			<br />
			See our
			<a
				class="text-blue-700"
				href="https://github.com/microsoft/onnxruntime-inference-examples/tree/main/c_cxx/QNN_EP/mobilenetv2_classification"
				>C# tutorial</a
			>
			for an example of how this is done.
			<br />
			Many models can be optimized for the NPU using this process. Even if a model cannot be optimized
			for the NPU, it can still be run by ONNX Runtime on the CPU.
			<br /><br />
			<p class="text-xl text-blue-700">Getting Help</p>
			For help with ONNX Runtime, you can<a
				class="text-blue-700"
				href="https://github.com/microsoft/onnxruntime/discussions">start a discussion</a
			>
			on GitHub or
			<a class="text-blue-700" href="https://github.com/microsoft/onnxruntime/issues"
				>file an issue</a
			>.
		</div>
		<div class="m-auto">
			<img src={windowsdevkit} alt="Windows Dev kit" />
		</div>
	</div>
</div>
