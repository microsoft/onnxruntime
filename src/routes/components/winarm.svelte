<script>
	import windowsdevkit from '../../images/windowsdevkit.png';
</script>

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
	<div class="grid grid-cols-3">
		<div class="md:col-span-2 col-span-3">
			<h2 class="text-2xl mb-2">Get started on your Windows Dev Kit 2023 today</h2>
			<p class="text-xl text-blue-500">
				Follow these steps to setup your device to use ONNX Runtime (ORT) with the built in NPU:
			</p>
			<ol class="list-decimal ml-10">
				<li>
					Request access to the Neural Processing SDK for Windows on Snapdragon. Qualcomm may reach
					out to you via email with further registration instructions for approval.
				</li>
				<li>Once approved, you will receive an email with links to download SNPE.</li>
				<li>Select the SNPE link which takes you to a Qualcomm login and download page.</li>
				<li>Select the Snapdragon_NPE_SDK.WIN.1.0 Installer link, download and install.</li>
				<li>Download and install the ONNX Runtime with SNPE package.</li>
				<li>Start using the ONNX Runtime API in your application.</li>
			</ol>
			<p class="text-xl text-blue-500">Optimizing models for the NPU</p>

			ONNX is a standard format for representing ML models authored in frameworks like PyTorch,
			TensorFlow, and others. ONNX Runtime can run any ONNX model, however to make use of the NPU,
			you currently need to use the following steps:
			<ol class="list-disc ml-10">
				<li>Run the tools provided in the SNPE SDK on your model to generate a binary file.</li>
				<li>Include the contents of the binary file as a node in the ONNX graph.</li>
				See our C# tutorial for an example of how this is done.
			</ol>
			<br />
			Many models can be optimized for the NPU using this process. Even if a model cannot be optimized
			for NPU by the SNPE SDK, it can still be run by ONNX Runtime on the CPU.
			<p class="text-xl text-blue-500">Tutorials</p>
			<ol class="list-disc ml-10">
				<li>C# Image classification with VGG16 using ONNX Runtime with SNPE</li>
				<li>C++ image classification with Inception v3 using ONNX Runtime with SNPE</li>
			</ol>
			<p class="text-xl text-blue-500">Getting Help</p>
			For help with ONNX Runtime, you can start a discussion on GitHub or file an issue.
		</div>
		<div class="m-auto">
			<img src={windowsdevkit} alt="Windows Dev kit" />
		</div>
	</div>
</div>
