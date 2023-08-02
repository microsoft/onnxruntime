<script>
	import Highlight from 'svelte-highlight';
	import python from 'svelte-highlight/languages/python';
	import csharp from 'svelte-highlight/languages/csharp';
	import javascript from 'svelte-highlight/languages/javascript';
	import java from 'svelte-highlight/languages/java';
	import cpp from 'svelte-highlight/languages/cpp';
	import FaRegClipboard from 'svelte-icons/fa/FaRegClipboard.svelte';
	import { blur, fade } from 'svelte/transition';

	let pythonCode =
		'import onnxruntime as ort\n# Load the model and create InferenceSession\nmodel_path = "path/to/your/onnx/model"\nsession = ort.InferenceSession(model_path)\n# Load and preprocess the input image inputTensor\n...\n# Run inference\noutputs = session.run(None {"input": inputTensor})\nprint(outputs)';
	let csharpCode =
		'using Microsoft.ML.OnnxRuntime;\n// Load the model and create InferenceSession\nstring model_path = "path/to/your/onnx/model";\nvar session = new InferenceSession(model_path);\n// Load and preprocess the input image to inputTensor\n...\n// Run inference\nvar outputs = session.Run(inputTensor).ToList();\nConsole.WriteLine(outputs[0].AsTensor()[0]);';
	let javascriptCode =
		'import * as ort from "onnxruntime-web";\n// Load the model and create InferenceSession\nconst modelPath = "path/to/your/onnx/model";\nconst session = await ort.InferenceSession.create(modelPath);\n// Load and preprocess the input image to inputTensor\n...\n// Run inference\nconst outputs = await session.run({ input: inputTensor });\nconsole.log(outputs);';
	let javaCode =
		'import ai.onnxruntime.*;\n// Load the model and create InferenceSession\nString modelPath = "path/to/your/onnx/model";\nOrtEnvironment env = OrtEnvironment.getEnvironment();\nOrtSession session = env.createSession(modelPath);\n// Load and preprocess the input image inputTensor\n...\n// Run inference\nOrtSession.Result outputs = session.run(inputTensor);\nSystem.out.println(outputs.get(0).getTensor().getFloatBuffer().get(0));';
	let cppCode =
		'#include "onnxruntime_cxx_api.h"\n// Load the model and create InferenceSession\nOrt::Env env;\nstd::string model_path = "path/to/your/onnx/model";\nOrt::Session session(env, model_path, Ort::SessionOptions{ nullptr });\n// Load and preprocess the input image to \n// inputTensor, inputNames, and outputNames\n...\n// Run inference\nstd::vector outputTensors =\nsession.Run(Ort::RunOptions{nullptr}, \ninputNames.data(), \n&inputTensor, \ninputNames.size(), \noutputNames.data(), \noutputNames.size());\nconst float* outputDataPtr = outputTensors[0].GetTensorMutableData();\nstd::cout << outputDataPtr[0] << std::endl;';
	// a svelte function to conditionally render different "Highlight" components based on what tab was clicked
	let activeTab = 'Python'; // set the initial active tab to Python

	// TODO: ensure work, change timeout
	let tabs = ['Python', 'C#', 'JavaScript', 'Java', 'C++'];
	let interacted = false;
	let currentTab = 0;
	let copied = '';
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

	let handleClick = (event) => {
		interacted = true;
		const tabText = event.target.textContent.trim();
		if (tabText === 'More..') {
			window.location.href = '/docs/get-started';
		}
		activeTab = tabText;
		activeTab = activeTab;
	};
	let copy = async () => {
		let copy;
		switch (activeTab) {
			case 'Python':
				copy = pythonCode;
				break;
			case 'C#':
				copy = csharpCode;
				break;
			case 'JavaScript':
				copy = javascriptCode;
				break;
			case 'Java':
				copy = javaCode;
				break;
			case 'C++':
				copy = cppCode;
				break;
			default:
				copy = ''; // Set a default value if needed
				break;
		}
		try {
			await navigator.clipboard.writeText(copy);
			copied = activeTab;
			setTimeout(() => {
				copied = '';
			}, 3000);
		} catch (err) {
			console.error('Failed to copy:', err);
		}
	};
	// TODO: get data theme from html tag
</script>

{#if copied != ''}
	<div class="toast toast-top">
		<div class="alert alert-info">
			<span>{copied} code successfully copied!</span>
		</div>
	</div>
{/if}
<div class="container mx-auto">
	<div class="grid-cols-3 gap-10 grid">
		<div class="col-span-1 mx-auto ">
			<h1 class="text-xl">Use ONNX Runtime with your favorite language</h1>
		</div>
		<div class="hidden md:block col-span-2 mx-auto tab-container">
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
				<div class="div" in:fade={{ duration: 500 }}>
					<button on:click={copy} class="btn btn-sm float-right -ml-20 z-10 rounded-none"
						><span class="icon"><FaRegClipboard /></span></button
					>
					<Highlight language={python} code={pythonCode} />
				</div>
			{:else if activeTab === 'C#'}
				<div class="div" in:fade={{ duration: 500 }}>
					<button on:click={copy} class="btn btn-sm float-right -ml-20 z-10 rounded-none"
						><span class="icon"><FaRegClipboard /></span></button
					>
					<Highlight language={csharp} code={csharpCode} />
				</div>
			{:else if activeTab === 'JavaScript'}
				<div class="div" in:fade={{ duration: 500 }}>
					<button on:click={copy} class="btn btn-sm float-right -ml-20 z-10 rounded-none"
						><span class="icon"><FaRegClipboard /></span></button
					>
					<Highlight language={javascript} code={javascriptCode} />
				</div>
			{:else if activeTab === 'Java'}
				<div class="div" in:fade={{ duration: 500 }}>
					<button on:click={copy} class="btn btn-sm float-right -ml-20 z-10 rounded-none"
						><span class="icon"><FaRegClipboard /></span></button
					>
					<Highlight language={java} code={javaCode} />
				</div>
			{:else if activeTab === 'C++'}
				<div class="div" in:fade={{ duration: 500 }}>
					<button on:click={copy} class="btn btn-sm float-right -ml-20 z-10 rounded-none"
						><span class="icon"><FaRegClipboard /></span></button
					>
					<Highlight language={cpp} code={cppCode} />
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
	.icon {
		width: 24px;
		height: 24px;
	}
</style>
