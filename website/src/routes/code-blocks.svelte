<script>
	import { base } from '$app/paths';
	import Highlight from 'svelte-highlight';
	import python from 'svelte-highlight/languages/python';
	import csharp from 'svelte-highlight/languages/csharp';
	import javascript from 'svelte-highlight/languages/javascript';
	import java from 'svelte-highlight/languages/java';
	import cpp from 'svelte-highlight/languages/cpp';
	import typescript from 'svelte-highlight/languages/typescript';
	import oneLight from 'svelte-highlight/styles/one-light';
	import oneDark from 'svelte-highlight/styles/onedark';

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

	let handleClick = (e) => {
		// get the text content of the clicked tab
		const tabText = e.target.textContent.trim();
		// if tabtext === 'c++' {
		//     tabtext = 'cpp';
		// }
		// update the active tab state
		if (tabText === 'More..') {
			window.location.href = '/docs/get-started';
		}
		activeTab = tabText;
		activeTab = activeTab;
		console.log(activeTab);
	};
	// get data theme from html tag
	// let html = document.querySelector('html');
	// let currentTheme = html!=null?html.getAttribute('data-theme'):'corporate';
</script>

<svelte:head >
	<!-- {#if currentTheme == 'corporate'} -->
	{@html oneLight}
	<!-- {:else}
		{@html oneDark}
	{/if} -->
</svelte:head>
<div class="container mx-auto">
	<div class="tabs">
		<a
			on:click={handleClick}
			class="tab tab-lg tab-lifted {activeTab === 'Python' ? 'tab-active' : ''}">Python</a
		>
		<a on:click={handleClick} class="tab tab-lg tab-lifted {activeTab === 'C#' ? 'tab-active' : ''}"
			>C#</a
		>
		<a
			on:click={handleClick}
			class="tab tab-lg tab-lifted {activeTab === 'JavaScript' ? 'tab-active' : ''}">JavaScript</a
		>
		<a
			on:click={handleClick}
			class="tab tab-lg tab-lifted {activeTab === 'Java' ? 'tab-active' : ''}">Java</a
		>
		<a
			on:click={handleClick}
			class="tab tab-lg tab-lifted {activeTab === 'C++' ? 'tab-active' : ''}">C++</a
		>
		<a
			on:click={handleClick}
			class="tab tab-lg tab-lifted {activeTab === 'More..' ? 'tab-active' : ''}">More..</a
		>
	</div>

	{#if activeTab === 'Python'}
		<Highlight language={python} code={pythonCode} />
	{:else if activeTab === 'C#'}
		<Highlight language={csharp} code={csharpCode} />
	{:else if activeTab === 'JavaScript'}
		<Highlight language={javascript} code={javascriptCode} />
	{:else if activeTab === 'Java'}
		<Highlight language={java} code={javaCode} />
	{:else if activeTab === 'C++'}
		<Highlight language={cpp} code={cppCode} />
	{:else if activeTab === 'More..'}
		Link to docs
	{/if}
</div>
