<script lang="ts">
	import Highlight from 'svelte-highlight';
	import python from 'svelte-highlight/languages/python';
	import csharp from 'svelte-highlight/languages/csharp';
	import javascript from 'svelte-highlight/languages/javascript';
	import java from 'svelte-highlight/languages/java';
	import cpp from 'svelte-highlight/languages/cpp';
	import FaLink from 'svelte-icons/fa/FaLink.svelte';
	import { blur, fade } from 'svelte/transition';

	let pythonCode =
		'import onnxruntime as ort\n# Load the model and create InferenceSession\nmodel_path = "path/to/your/onnx/model"\nsession = ort.InferenceSession(model_path)\n# Load and preprocess the input image inputTensor\n...\n# Run inference\noutputs = session.run(None {"input": inputTensor})\nprint(outputs)';
	let csharpCode =
		'using Microsoft.ML.OnnxRuntime;\n// Load the model and create InferenceSession\nstring model_path = "path/to/your/onnx/model";\nvar session = new InferenceSession(model_path);\n// Load and preprocess the input image to inputTensor\n...\n// Run inference\nvar outputs = session.Run(inputTensor).ToList();\nConsole.WriteLine(outputs[0].AsTensor()[0]);';
	let javascriptCode =
		'import * as ort from "onnxruntime-web";\n// Load the model and create InferenceSession\nconst modelPath = "path/to/your/onnx/model";\nconst session = await ort.InferenceSession.create(modelPath);\n// Load and preprocess the input image to inputTensor\n...\n// Run inference\nconst outputs = await session.run({ input: inputTensor });\nconsole.log(outputs);';
	let javaCode =
		'import ai.onnxruntime.*;\n// Load the model and create InferenceSession\nString modelPath = "path/to/your/onnx/model";\nOrtEnvironment env = OrtEnvironment.getEnvironment();\nOrtSession session = env.createSession(modelPath);\n// Load and preprocess the input image to inputTensor\n...\n// Run inference\nOrtSession.Result outputs = session.run(inputTensor);\nSystem.out.println(outputs.get(0).getTensor().getFloatBuffer().get(0));\n\n';
	let cppCode =
		'#include "onnxruntime_cxx_api.h"\n// Load the model and create InferenceSession\nOrt::Env env;\nstd::string model_path = "path/to/your/onnx/model";\nOrt::Session session(env, model_path, Ort::SessionOptions{ nullptr });\n// Load and preprocess the input image to inputTensor\n...\n// Run inference\nstd::vector outputTensors =\nsession.Run(Ort::RunOptions{nullptr}, inputNames.data(), &inputTensor, \n  inputNames.size(), outputNames.data(), outputNames.size());\nconst float* outputDataPtr = outputTensors[0].GetTensorMutableData();\nstd::cout << outputDataPtr[0] << std::endl;';
	// a svelte function to conditionally render different "Highlight" components based on what tab was clicked
	let activeTab = 'Python'; // set the initial active tab to Python

	let tabs = ['Python', 'C#', 'JavaScript', 'Java', 'C++'];
	let interacted = false;
	let currentTab = 0;
	let cycleCode = () => {
		currentTab = (currentTab + 1) % 5;
		activeTab = tabs[currentTab];
		activeTab = activeTab;
		if (!interacted) {
			setTimeout(cycleCode, 3000);
		}
	};
	if (!interacted) {
		setTimeout(cycleCode, 10000);
	}

	let handleClick = (e: any) => {
		interacted = true;
		const tabText = e.target.textContent.trim();
		if (tabText === 'More..') {
			window.location.href = './docs/get-started';
		}
		activeTab = tabText;
		activeTab = activeTab;
	};
</script>

<div class="container mx-auto px-4">
	<h1 class="text-xl mb-4 text-center">
		Use ONNX Runtime with your favorite language and get started with the tutorials:
	</h1>
	<div class="grid-cols-1 lg:grid-cols-3 gap-4 grid">
		<div class="col-span-1 mx-auto mt-6 mx-4 lg:mx-0 lg:ml-10">
			<div class="join join-vertical gap-4 w-full">
				<a href="./getting-started" class="btn btn-primary rounded-sm btn-block">Quickstart</a>
				<a rel="external" href="./docs/tutorials" class="btn btn-primary rounded-sm btn-block"
					>Tutorials</a
				>
				<a rel="external" href="./docs/install" class="btn btn-primary rounded-sm btn-block"
					>Install ONNX Runtime</a
				>
				<a
					rel="external"
					href="./docs/execution-providers"
					class="btn btn-primary rounded-sm btn-block">Hardware acceleration</a
				>
			</div>
		</div>
		<div class="hidden lg:block col-span-2 mx-auto tab-container">
			<div class="tabs">
				<p
					on:mouseenter={handleClick}
					class="tab tab-lg tab-bordered {activeTab === 'Python' ? 'tab-active' : ''}"
				>
					Python
				</p>
				<p
					on:mouseenter={handleClick}
					class="tab tab-lg tab-bordered {activeTab === 'C#' ? 'tab-active' : ''}"
				>
					C#
				</p>
				<p
					on:mouseenter={handleClick}
					class="tab tab-lg tab-bordered {activeTab === 'JavaScript' ? 'tab-active' : ''}"
				>
					JavaScript
				</p>
				<p
					on:mouseenter={handleClick}
					class="tab tab-lg tab-bordered {activeTab === 'Java' ? 'tab-active' : ''}"
				>
					Java
				</p>
				<p
					on:mouseenter={handleClick}
					class="tab tab-lg tab-bordered {activeTab === 'C++' ? 'tab-active' : ''}"
				>
					C++
				</p>
				<button
					on:click={handleClick}
					class="tab tab-lg tab-bordered {activeTab === 'More..' ? 'tab-active' : ''}"
					>More..</button
				>
			</div>
			{#if activeTab === 'Python'}
				<Highlight language={python} code={pythonCode} />
				<div class="div" in:fade={{ duration: 500 }}>
					<a
						href="https://onnxruntime.ai/docs/get-started/with-python"
						class="btn btn-sm float-right -mt-8 z-10 rounded-none"
						>Python Docs<span class="w-5 h-5"><FaLink /></span></a
					>
				</div>
			{:else if activeTab === 'C#'}
				<div class="div" in:fade={{ duration: 500 }}>
					<Highlight language={csharp} code={csharpCode} />
					<a
						href="https://onnxruntime.ai/docs/get-started/with-csharp"
						class="btn btn-sm float-right -mt-8 z-10 rounded-none"
						>C# Docs<span class="w-5 h-5"><FaLink /></span></a
					>
				</div>
			{:else if activeTab === 'JavaScript'}
				<div class="div" in:fade={{ duration: 500 }}>
					<Highlight language={javascript} code={javascriptCode} />
					<a
						href="https://onnxruntime.ai/docs/get-started/with-javascript"
						class="btn btn-sm float-right -mt-8 z-10 rounded-none"
						>JavaScript Docs<span class="w-5 h-5"><FaLink /></span></a
					>
				</div>
			{:else if activeTab === 'Java'}
				<div class="div" in:fade={{ duration: 500 }}>
					<Highlight language={java} code={javaCode} />
					<a
						href="https://onnxruntime.ai/docs/get-started/with-java"
						class="btn btn-sm float-right -mt-8 z-10 rounded-none"
						>Java Docs<span class="w-5 h-5"><FaLink /></span></a
					>
				</div>
			{:else if activeTab === 'C++'}
				<div class="div" in:fade={{ duration: 500 }}>
					<Highlight language={cpp} code={cppCode} />
					<a
						href="https://onnxruntime.ai/docs/get-started/with-cpp"
						class="btn btn-sm float-right -mt-8 z-10 rounded-none"
						>C++ Docs<span class="w-5 h-5"><FaLink /></span></a
					>
				</div>
			{:else if activeTab === 'More..'}
				Link to docs
			{:else}
				Copy code
			{/if}
		</div>
	</div>
</div>

<style>
	.tab-container {
		min-width: 675px;
		min-height: 525px;
	}
</style>
