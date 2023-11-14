<script>
	import { onMount } from 'svelte';
	import jq from 'jquery';
	let platforms = ['Windows', 'Linux', 'Mac', 'Android', 'iOS', 'Web Browser'];
	let platformIDs = ['windows', 'linux', 'mac', 'android', 'ios', 'web'];
	let apis = ['Python', 'C++', 'C#', 'C', 'Java', 'JS', 'Obj-C', 'WinRT'];
	let apiIDs = ['Python', 'C++', 'C#', 'C-API', 'Java', 'JS', 'objectivec', 'WinRT'];
	let architectures = ['X64', 'X86', 'ARM64', 'ARM32', 'IBM Power'];
	let architecturesIDs = ['X64', 'X86', 'ARM64', 'ARM32', 'Power'];
	let hardwareAccelerations = [
		'Default CPU',
		'CoreML',
		'CUDA',
		'DirectML',
		'MIGraphX',
		'NNAPI',
		'oneDNN',
		'OpenVINO',
		'ROCm',
		'QNN',
		'Tensor RT',
		'ACL (Preview)',
		'ArmNN (Preview)',
		'Azure (Preview)',
		'CANN (Preview)',
		'Rockchip NPU (Preview)',
		'TVM (Preview)',
		'Vitis AI (Preview)',
		'XNNPACK (Preview)'
	];
	let hardwareAccelerationIDs = [
		'DefaultCPU',
		'CoreML',
		'CUDA',
		'DirectML',
		'MIGraphX',
		'NNAPI',
		'DNNL',
		'OpenVINO',
		'ROCm',
		'QNN',
		'TensorRT',
		'ACL',
		'ArmNN',
		'Azure',
		'CANN',
		'RockchipNPU',
		'TVM',
		'VitisAI',
		'XNNPACK'
	];
	// Training
	const TrainingScenarios = ['Large Model Training', 'On-Device Training'];
	const TrainingScenarioIds = ['ot_large_model', 'ot_on_device'];
	const TrainingPlatforms = ['Linux', 'Windows', 'Mac', 'Android', 'iOS'];
	const TrainingPlatformIds = ['ot_linux', 'ot_windows', 'ot_mac', 'ot_android', 'ot_ios'];
	const TrainingAPIs = ['Python', 'C', 'C++', 'C#', 'Java', 'Obj-C'];
	const TrainingAPIIds = ['ot_python', 'ot_c', 'ot_cplusplus', 'ot_csharp', 'ot_java', 'ot_objc'];
	const TrainingVersions = ['CUDA 11.8', 'ROCm', 'CPU'];
	const TrainingVersionIds = ['ot_CUDA118', 'ot_ROCm', 'ot_CPU'];
	const TrainingBuilds = ['Stable', 'Preview (Nightly)'];
	const TrainingBuildIds = ['ot_stable', 'ot_nightly'];
	const validCombos = {
		'windows,C-API,X64,CUDA':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a> <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'windows,C++,X64,CUDA':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a> <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'windows,C#,X64,CUDA':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a> <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'windows,Python,X64,CUDA':
			"pip install onnxruntime-gpu <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'linux,Python,ARM64,CUDA':
			"For Jetpack 4.4+, follow installation instructions from <a class='text-blue-500' href='https://elinux.org/Jetson_Zoo#ONNX_Runtime' target='_blank'>here</a>",

		'linux,C-API,X64,CUDA':
			"Download .tgz file from&nbsp;<a class='text-blue-500' href='https://github.com/microsoft/onnxruntime/releases' target='_blank'>Github</a> <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'linux,C++,X64,CUDA':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a> <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'linux,C#,X64,CUDA':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a> <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'linux,Python,X64,CUDA':
			"pip install onnxruntime-gpu <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'linux,C-API,ARM32,DefaultCPU':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-arm' target='_blank'>here</a>",

		'linux,C++,ARM32,DefaultCPU':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-arm' target='_blank'>here</a>",

		'linux,Python,ARM32,DefaultCPU':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-arm' target='_blank'>here</a>",

		'windows,C-API,X64,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

		'windows,C-API,X86,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

		'windows,C-API,ARM32,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

		'windows,C++,ARM32,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

		'windows,C#,ARM32,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

		'windows,C-API,ARM64,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

		'windows,C++,ARM64,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

		'windows,C#,ARM64,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

		'windows,C++,X64,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

		'windows,C++,X86,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

		'windows,C#,X64,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

		'windows,C#,X86,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

		'linux,C-API,X64,DefaultCPU':
			"Download .tgz file from&nbsp;<a class='text-blue-500' href='https://github.com/microsoft/onnxruntime/releases' target='_blank'>Github</a>",

		'linux,C++,X64,DefaultCPU':
			"Download .tgz file from&nbsp;<a class='text-blue-500' href='https://github.com/microsoft/onnxruntime/releases' target='_blank'>Github</a>",

		'linux,C#,X64,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

		'mac,C-API,X64,DefaultCPU':
			"Download .tgz file from&nbsp;<a class='text-blue-500' href='https://github.com/microsoft/onnxruntime/releases' target='_blank'>Github</a>",

		'mac,C++,X64,DefaultCPU':
			"Download .tgz file from&nbsp;<a class='text-blue-500' href='https://github.com/microsoft/onnxruntime/releases' target='_blank'>Github</a>",

		'mac,C#,X64,DefaultCPU':
			"Download .tgz file from&nbsp;<a class='text-blue-500' href='https://github.com/microsoft/onnxruntime/releases' target='_blank'>Github</a>",

		'mac,C#,X64,CoreML':
			"Download .tgz file from&nbsp;<a class='text-blue-500' href='https://github.com/microsoft/onnxruntime/releases' target='_blank'>Github</a>",

		'windows,Python,X64,DefaultCPU': 'pip install onnxruntime',

		'mac,Python,X64,DefaultCPU': 'pip install onnxruntime',

		'linux,Python,X64,DefaultCPU': 'pip install onnxruntime',

		'linux,Python,ARM64,DefaultCPU': 'pip install onnxruntime',

		'windows,C-API,X64,DNNL':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

		'windows,C++,X64,DNNL':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

		'windows,C#,X64,DNNL':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

		'windows,Python,X64,DNNL':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

		'linux,C-API,X64,DNNL':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

		'linux,C++,X64,DNNL':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

		'linux,C#,X64,DNNL':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

		'linux,Python,X64,DNNL':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-mkldnn' target='_blank'>here</a>",

		'linux,Python,X64,TVM':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-stvm' target='_blank'>here</a>",

		'linux,Python,X86,TVM':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-stvm' target='_blank'>here</a>",

		'linux,Python,ARM32,TVM':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-stvm' target='_blank'>here</a>",

		'linux,Python,ARM64,TVM':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-stvm' target='_blank'>here</a>",

		'windows,Python,X64,TVM':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-stvm' target='_blank'>here</a>",

		'windows,Python,X86,TVM':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-stvm' target='_blank'>here</a>",

		'windows,Python,ARM32,TVM':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-stvm' target='_blank'>here</a>",

		'windows,Python,ARM64,TVM':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-stvm' target='_blank'>here</a>",

		'linux,C-API,X64,OpenVINO':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

		'linux,C++,X64,OpenVINO':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

		'linux,C#,X64,OpenVINO':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

		'linux,Python,X64,OpenVINO':
			"pip install onnxruntime-openvino <br/>Docker image also <a class='text-blue-500' href='https://hub.docker.com/r/openvino/onnxruntime_ep_ubuntu18' target='_blank'>available</a>.",

		'windows,C-API,X64,OpenVINO':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

		'windows,C++,X64,OpenVINO':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

		'windows,C#,X64,OpenVINO':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://aka.ms/build-ort-openvino' target='_blank'>here</a>",

		'windows,Python,X64,OpenVINO': 'pip install onnxruntime-openvino',

		'windows,C-API,X64,TensorRT':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a> <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html' target='_blank'>docs</a> for usage details.",

		'windows,C++,X64,TensorRT':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a> <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html' target='_blank'>docs</a> for usage details.",

		'windows,C#,X64,TensorRT':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a> <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html' target='_blank'>docs</a> for usage details.",

		'windows,Python,X64,TensorRT':
			"pip install onnxruntime-gpu <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'linux,C-API,X64,TensorRT':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a> <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html' target='_blank'>docs</a> for usage details.",

		'linux,C++,X64,TensorRT':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a> <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html' target='_blank'>docs</a> for usage details.",

		'linux,C#,X64,TensorRT':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Gpu' target='_blank'>Microsoft.ML.OnnxRuntime.Gpu</a> <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html' target='_blank'>docs</a> for usage details.",

		'linux,Python,X64,TensorRT':
			"pip install onnxruntime-gpu <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'linux,C#,ARM64,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime",

		'linux,Python,ARM64,TensorRT':
			"pip install onnxruntime-gpu <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'windows,C-API,X86,DirectML':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML' target='_blank'>Microsoft.ML.OnnxRuntime.DirectML</a> <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'windows,C++,X86,DirectML':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML' target='_blank'>Microsoft.ML.OnnxRuntime.DirectML</a>",

		'windows,C#,X86,DirectML':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML' target='_blank'>Microsoft.ML.OnnxRuntime.DirectML</a>",

		'windows,Python,X86,DirectML':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

		'windows,C-API,X64,DirectML':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML' target='_blank'>Microsoft.ML.OnnxRuntime.DirectML</a>",

		'windows,C++,X64,DirectML':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML' target='_blank'>Microsoft.ML.OnnxRuntime.DirectML</a>",

		'windows,C#,X64,DirectML':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML' target='_blank'>Microsoft.ML.OnnxRuntime.DirectML</a>",

		'windows,Python,X64,DirectML': 'pip install onnxruntime-directml',

		'windows,C-API,ARM64,DirectML':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML' target='_blank'>Microsoft.ML.OnnxRuntime.DirectML</a>",

		'windows,C++,ARM64,DirectML':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML' target='_blank'>Microsoft.ML.OnnxRuntime.DirectML</a>",

		'windows,C#,ARM64,DirectML':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML' target='_blank'>Microsoft.ML.OnnxRuntime.DirectML</a>",

		'windows,Python,ARM64,DirectML':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-directml' target='_blank'>here</a>",

		'linux,Java,X64,DefaultCPU':
			"Add a dependency on <a class='text-blue-500' href='https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime' target='_blank'>com.microsoft.onnxruntime:onnxruntime</a> using Maven/Gradle",

		'linux,Java,X64,CUDA':
			"Add a dependency on <a class='text-blue-500' href='https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime_gpu' target='_blank'>com.microsoft.onnxruntime:onnxruntime_gpu</a> using Maven/Gradle. <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'mac,Java,X64,DefaultCPU':
			"Add a dependency on <a class='text-blue-500' href='https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime' target='_blank'>com.microsoft.onnxruntime:onnxruntime</a> using Maven/Gradle",

		//javascript
		'linux,JS,X64,DefaultCPU': 'npm install onnxruntime-node',

		'mac,JS,X64,DefaultCPU': 'npm install onnxruntime-node',

		'windows,JS,X64,DefaultCPU': 'npm install onnxruntime-node',

		'web,JS,,': 'npm install onnxruntime-web',

		'android,JS,ARM64,DefaultCPU': 'npm install onnxruntime-react-native',

		'android,JS,X64,DefaultCPU': 'npm install onnxruntime-react-native',

		'android,JS,X86,DefaultCPU': 'npm install onnxruntime-react-native',

		'ios,JS,ARM64,DefaultCPU': 'npm install onnxruntime-react-native',

		'windows,WinRT,X86,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.AI.MachineLearning' target='_blank'>Microsoft.AI.MachineLearning</a>",

		'windows,WinRT,X64,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.AI.MachineLearning' target='_blank'>Microsoft.AI.MachineLearning</a>",

		'windows,WinRT,ARM64,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.AI.MachineLearning' target='_blank'>Microsoft.AI.MachineLearning</a>",

		'windows,WinRT,ARM32,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.AI.MachineLearning' target='_blank'>Microsoft.AI.MachineLearning</a>",

		'windows,WinRT,X86,DirectML':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.AI.MachineLearning' target='_blank'>Microsoft.AI.MachineLearning</a>",

		'windows,WinRT,X64,DirectML':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.AI.MachineLearning' target='_blank'>Microsoft.AI.MachineLearning</a>",

		'windows,WinRT,ARM64,DirectML':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.AI.MachineLearning' target='_blank'>Microsoft.AI.MachineLearning</a>",

		'windows,Java,X64,DefaultCPU':
			"Add a dependency on <a class='text-blue-500' href='https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime' target='_blank'>com.microsoft.onnxruntime:onnxruntime</a> using Maven/Gradle",

		'windows,Java,X64,CUDA':
			"Add a dependency on <a class='text-blue-500' href='https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime_gpu' target='_blank'>com.microsoft.onnxruntime:onnxruntime_gpu</a> using Maven/Gradle. <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'windows,Java,X64,TensorRT':
			"Add a dependency on <a class='text-blue-500' href='https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime_gpu' target='_blank'>com.microsoft.onnxruntime:onnxruntime_gpu</a> using Maven/Gradle. <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html' target='_blank'>docs</a> for usage details.",

		'windows,Java,X64,DNNL':
			"Follow <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/build/inferencing.html#common-build-instructions' target='_blank'>build</a> and <a class='text-blue-500' href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

		'windows,Java,X64,OpenVINO':
			"Follow <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/build/inferencing.html#common-build-instructions' target='_blank'>build</a> and <a class='text-blue-500' href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

		'linux,Java,X64,TensorRT':
			"Add a dependency on <a class='text-blue-500' href='https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime_gpu' target='_blank'>com.microsoft.onnxruntime:onnxruntime_gpu</a> using Maven/Gradle. <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html' target='_blank'>docs</a> for usage details.",

		'linux,Java,X64,DNNL':
			"Follow <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/build/inferencing.html#common-build-instructions' target='_blank'>build</a> and <a class='text-blue-500' href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

		'linux,Java,X64,OpenVINO':
			"Follow <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/build/inferencing.html#common-build-instructions' target='_blank'>build</a> and <a class='text-blue-500' href='https://aka.ms/onnxruntime-java' target='_blank'>API instructions</a>",

		'android,C-API,ARM64,NNAPI':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-nnapi' target='_blank'>here</a>",

		'android,C++,ARM64,NNAPI':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-nnapi' target='_blank'>here</a>",

		'android,Java,ARM64,NNAPI':
			"Add a dependency on <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android' target='_blank'>com.microsoft.onnxruntime:onnxruntime-android</a> or <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-mobile' target='_blank'>com.microsoft.onnxruntime:onnxruntime-mobile</a> using Maven/Gradle and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'android,C-API,X86,NNAPI':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-nnapi' target='_blank'>here</a>",

		'android,C++,X86,NNAPI':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-nnapi' target='_blank'>here</a>",

		'android,C#,X86,NNAPI':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>.",

		'android,Java,X64,NNAPI':
			"Add a dependency on <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android' target='_blank'>com.microsoft.onnxruntime:onnxruntime-android</a> or <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-mobile' target='_blank'>com.microsoft.onnxruntime:onnxruntime-mobile</a> using Maven/Gradle and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'android,C-API,X64,NNAPI':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-nnapi' target='_blank'>here</a>",

		'android,C++,X64,NNAPI':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-nnapi' target='_blank'>here</a>",

		'android,C#,X64,NNAPI':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>.",

		'android,Java,X86,NNAPI':
			"Add a dependency on <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android' target='_blank'>com.microsoft.onnxruntime:onnxruntime-android</a> or <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-mobile' target='_blank'>com.microsoft.onnxruntime:onnxruntime-mobile</a> using Maven/Gradle and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'android,C-API,ARM32,NNAPI':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-nnapi' target='_blank'>here</a>",

		'android,C++,ARM32,NNAPI':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-nnapi' target='_blank'>here</a>",

		'android,C#,ARM32,NNAPI':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>.",

		'android,Java,ARM32,NNAPI':
			"Add a dependency on <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android' target='_blank'>com.microsoft.onnxruntime:onnxruntime-android</a> or <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-mobile' target='_blank'>com.microsoft.onnxruntime:onnxruntime-mobile</a> using Maven/Gradle and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'android,C-API,ARM64,DefaultCPU':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-android' target='_blank'>here</a>",

		'android,C++,ARM64,DefaultCPU':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-android' target='_blank'>here</a>",

		'android,Java,ARM64,DefaultCPU':
			"Add a dependency on <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android' target='_blank'>com.microsoft.onnxruntime:onnxruntime-android</a> or <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-mobile' target='_blank'>com.microsoft.onnxruntime:onnxruntime-mobile</a> using Maven/Gradle and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'android,C-API,ARM32,DefaultCPU':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-android' target='_blank'>here</a>",

		'android,C++,ARM32,DefaultCPU':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-android' target='_blank'>here</a>",

		'android,C#,ARM32,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>.",

		'android,Java,ARM32,DefaultCPU':
			"Add a dependency on <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android' target='_blank'>com.microsoft.onnxruntime:onnxruntime-android</a> or <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-mobile' target='_blank'>com.microsoft.onnxruntime:onnxruntime-mobile</a> using Maven/Gradle and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'android,C-API,X86,DefaultCPU':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-android' target='_blank'>here</a>",

		'android,C++,X86,DefaultCPU':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-android' target='_blank'>here</a>",

		'android,C#,X86,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>.",

		'android,Java,X86,DefaultCPU':
			"Add a dependency on <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android' target='_blank'>com.microsoft.onnxruntime:onnxruntime-android</a> or <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-mobile' target='_blank'>com.microsoft.onnxruntime:onnxruntime-mobile</a> using Maven/Gradle and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'android,C-API,X64,DefaultCPU':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-android' target='_blank'>here</a>",

		'android,C++,X64,DefaultCPU':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-android' target='_blank'>here</a>",

		'android,C#,X64,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>.",

		'android,Java,X64,DefaultCPU':
			"Add a dependency on <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android' target='_blank'>com.microsoft.onnxruntime:onnxruntime-android</a> or <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-mobile' target='_blank'>com.microsoft.onnxruntime:onnxruntime-mobile</a> using Maven/Gradle and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'android,C#,ARM64,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>.",

		'android,C#,ARM64,NNAPI':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>.",

		'ios,C-API,ARM64,DefaultCPU':
			"Add 'onnxruntime-c' or 'onnxruntime-mobile-c' using CocoaPods and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'ios,C++,ARM64,DefaultCPU':
			"Add 'onnxruntime-c' or 'onnxruntime-mobile-c' using CocoaPods and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'ios,C-API,ARM64,CoreML':
			"Add 'onnxruntime-c' or 'onnxruntime-mobile-c' using CocoaPods and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'ios,C++,ARM64,CoreML':
			"Add 'onnxruntime-c' or 'onnxruntime-mobile-c' using CocoaPods and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'ios,objectivec,ARM64,DefaultCPU':
			"Add 'onnxruntime-objc' or 'onnxruntime-mobile-objc' using CocoaPods and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'ios,objectivec,ARM64,CoreML':
			"Add 'onnxruntime-objc' or 'onnxruntime-mobile-objc' using CocoaPods and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'ios,C#,ARM64,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>.",

		'ios,C#,ARM64,CoreML':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>.",

		'windows,Python,X64,VitisAI':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-vitisai' target='_blank'>here</a>",

		'windows,C++,X64,VitisAI':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-vitisai' target='_blank'>here</a>",

		'linux,C++,ARM64,VitisAI':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-vitisai' target='_blank'>here</a>",

		'linux,Python,ARM64,VitisAI':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-vitisai' target='_blank'>here</a>",

		'linux,Python,X64,MIGraphX':
			"pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training-rocm<br/>Build from source by following build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-migraphx' target='_blank'>here</a>",

		'linux,C-API,X64,MIGraphX':
			"pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training-rocm<br/>Build from source by following build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-migraphx' target='_blank'>here</a>",

		'linux,C++,X64,MIGraphX':
			"pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training-rocm<br/>Build from source by following build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-migraphx' target='_blank'>here</a>",

		'linux,Python,X64,ROCm':
			"pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training-rocm<br/>Build from source by following build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-rocm' target='_blank'>here</a>",

		'linux,C-API,X64,ROCm':
			"pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training-rocm<br/>Build from source by following build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-rocm' target='_blank'>here</a>",

		'linux,C++,X64,ROCm':
			"pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training-rocm<br/>Build from source by following build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-rocm' target='_blank'>here</a>",

		'linux,Python,ARM64,ACL':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-acl' target='_blank'>here</a>",

		'linux,C-API,ARM64,ACL':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-acl' target='_blank'>here</a>",

		'linux,C++,ARM64,ACL':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-acl' target='_blank'>here</a>",

		'linux,Python,ARM32,ACL':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-acl' target='_blank'>here</a>",

		'linux,C-API,ARM32,ACL':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-acl' target='_blank'>here</a>",

		'linux,C++,ARM32,ACL':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-acl' target='_blank'>here</a>",

		'linux,Python,ARM64,ArmNN':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-armnn' target='_blank'>here</a>",

		'linux,C-API,ARM64,ArmNN':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-armnn' target='_blank'>here</a>",

		'linux,C++,ARM64,ArmNN':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-armnn' target='_blank'>here</a>",

		'linux,Python,ARM32,ArmNN':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-armnn' target='_blank'>here</a>",

		'linux,C-API,ARM32,ArmNN':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-armnn' target='_blank'>here</a>",

		'linux,C++,ARM32,ArmNN':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-armnn' target='_blank'>here</a>",

		'linux,Python,ARM64,RockchipNPU':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-rknpu' target='_blank'>here</a>",

		'linux,C-API,ARM64,RockchipNPU':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-rknpu' target='_blank'>here</a>",

		'linux,C++,ARM64,RockchipNPU':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-rknpu' target='_blank'>here</a>",

		//mac m1
		'mac,C-API,ARM64,CoreML':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

		'mac,C#,ARM64,CoreML':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a> <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'mac,C++,ARM64,CoreML':
			"Download .tgz file from&nbsp;<a class='text-blue-500' href='https://github.com/microsoft/onnxruntime/releases' target='_blank'>Github</a>",

		'mac,Java,ARM64,CoreML':
			"Add a dependency on <a class='text-blue-500' href='https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime' target='_blank'>com.microsoft.onnxruntime:onnxruntime</a> using Maven/Gradle",

		'mac,Python,ARM64,DefaultCPU': 'pip install onnxruntime',

		'mac,Java,ARM64,DefaultCPU':
			"Add a dependency on <a class='text-blue-500' href='https://search.maven.org/artifact/com.microsoft.onnxruntime/onnxruntime' target='_blank'>com.microsoft.onnxruntime:onnxruntime</a> using Maven/Gradle",

		'mac,C#,ARM64,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

		'mac,C-API,ARM64,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

		'mac,C++,ARM64,DefaultCPU':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime' target='_blank'>Microsoft.ML.OnnxRuntime</a>",

		//power
		'linux,C-API,Power,DefaultCPU':
			"Follow build instructions from <a class='text-blue-500' href='https://onnxruntime.ai/docs/build/inferencing.html#build-instructions' target='_blank'>here</a>",

		'linux,C++,Power,DefaultCPU':
			"Follow build instructions from <a class='text-blue-500' href='https://onnxruntime.ai/docs/build/inferencing.html#build-instructions' target='_blank'>here</a>",

		'linux,Python,Power,DefaultCPU': 'pip install onnxruntime-powerpc64le',

		//QNN
		'windows,C-API,ARM64,QNN':
			"View installation instructions <a class='text-blue-500' href='https://aka.ms/build-ort-qnn' target='_blank'>here</a>",

		'windows,C++,ARM64,QNN':
			"View installation instructions <a class='text-blue-500' href='https://aka.ms/build-ort-qnn' target='_blank'>here</a>",

		'windows,C#,ARM64,QNN':
			"View installation instructions <a class='text-blue-500' href='https://aka.ms/build-ort-qnn' target='_blank'>here</a>",

		'linux,C-API,ARM64,QNN':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-qnn' target='_blank'>here</a>",

		'linux,C++,ARM64,QNN':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-qnn' target='_blank'>here</a>",

		'android,C-API,ARM64,QNN':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-qnn' target='_blank'>here</a>",

		'android,C++,ARM64,QNN':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-qnn' target='_blank'>here</a>",

		//Xnnpack
		'ios,C-API,ARM64,XNNPACK':
			"Add 'onnxruntime-c' using CocoaPods and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a> or Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-xnnpack' target='_blank'>here</a>",

		'ios,objectivec,ARM64,XNNPACK':
			"Add 'onnxruntime-objc' using CocoaPods and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'android,C-API,ARM64,XNNPACK':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-xnnpack' target='_blank'>here</a>",

		'android,C++,ARM64,XNNPACK':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-xnnpack' target='_blank'>here</a>",

		'android,Java,ARM64,XNNPACK':
			"Add a dependency on <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android' target='_blank'>com.microsoft.onnxruntime:onnxruntime-android</a> using Maven/Gradle and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'android,C-API,ARM32,XNNPACK':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-xnnpack' target='_blank'>here</a>",

		'android,C++,ARM32,XNNPACK':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-xnnpack' target='_blank'>here</a>",

		'android,Java,ARM32,XNNPACK':
			"Add a dependency on <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-android' target='_blank'>com.microsoft.onnxruntime:onnxruntime-android</a> using Maven/Gradle and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'windows,C-API,X86,XNNPACK':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-xnnpack' target='_blank'>here</a>",

		'windows,C++,X86,XNNPACK':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-xnnpack' target='_blank'>here</a>",

		'linux,C-API,X86,XNNPACK':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-xnnpack' target='_blank'>here</a>",

		'linux,C++,X86,XNNPACK':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-xnnpack' target='_blank'>here</a>",

		'linux,Python,ARM64,CANN':
			"pip install onnxruntime-cann <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'linux,C-API,ARM64,CANN':
			"Follow build instructions from <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html#build' target='_blank'>here</a>.",

		'linux,C++,ARM64,CANN':
			"Follow build instructions from <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html#build' target='_blank'>here</a>.",

		'linux,Python,X64,CANN':
			"pip install onnxruntime-cann <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/community-maintained/community-maintained/CANN-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'linux,C-API,X64,CANN':
			"Follow build instructions from <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html#build' target='_blank'>here</a>.",

		'linux,C++,X64,CANN':
			"Follow build instructions from <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/community-maintained/CANN-ExecutionProvider.html#build' target='_blank'>here</a>.",

		'windows,Python,X64,Azure':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-azure' target='_blank'>here</a>",

		'linux,Python,X64,Azure':
			"Follow build instructions from <a class='text-blue-500' href='https://aka.ms/build-ort-azure' target='_blank'>here</a>"
	};
	const ot_validCombos = {
		'ot_linux,ot_large_model,ot_python,ot_X64,ot_CUDA118,ot_stable':
			'python -m pip install cerberus flatbuffers h5py numpy>=1.16.6 onnx packaging protobuf sympy setuptools>=41.4.0<br/>pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training<br/>pip install torch-ort<br/>python -m torch_ort.configure',

		'ot_linux,ot_large_model,ot_python,ot_X64,ot_CUDA118,ot_nightly':
			'python -m pip install cerberus flatbuffers h5py numpy>=1.16.6 onnx packaging protobuf sympy setuptools>=41.4.0<br/>pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-training<br/>pip install torch-ort<br/>python -m torch_ort.configure',

		'ot_linux,ot_large_model,ot_python,ot_X64,ot_ROCm,ot_stable':
			"pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training-rocm<br/>pip install torch-ort<br/>python -m torch_ort.configure<br/><br/>*<a class='text-blue-500' href='https://download.onnxruntime.ai/' target='blank'>Available versions</a>",

		'ot_linux,ot_large_model,ot_python,ot_X64,ot_ROCm,ot_nightly':
			"pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-training-rocm<br/>pip install torch-ort<br/>python -m torch_ort.configure<br/><br/>*<a class='text-blue-500' href='https://download.onnxruntime.ai/' target='blank'>Available versions</a>",

		'ot_linux,ot_on_device,ot_python,ot_X64,ot_CPU,ot_stable':
			'python -m pip install cerberus flatbuffers h5py numpy>=1.16.6 onnx packaging protobuf sympy setuptools>=41.4.0<br/>pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training-cpu',

		'ot_linux,ot_on_device,ot_python,ot_X64,ot_CPU,ot_nightly':
			'python -m pip install cerberus flatbuffers h5py numpy>=1.16.6 onnx packaging protobuf sympy setuptools>=41.4.0<br/>pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-training-cpu',

		'ot_linux,ot_on_device,ot_python,ot_X64,ot_CUDA118,ot_stable':
			'python -m pip install cerberus flatbuffers h5py numpy>=1.16.6 onnx packaging protobuf sympy setuptools>=41.4.0<br/>pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training',

		'ot_linux,ot_on_device,ot_python,ot_X64,ot_CUDA118,ot_nightly':
			'python -m pip install cerberus flatbuffers h5py numpy>=1.16.6 onnx packaging protobuf sympy setuptools>=41.4.0<br/>pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-training',

		'ot_linux,ot_on_device,ot_cplusplus,ot_X64,ot_CPU,ot_stable':
			"Download .tgz file from&nbsp;<a class='text-blue-500' href='https://github.com/microsoft/onnxruntime/releases' target='_blank'>Github</a> <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'ot_linux,ot_on_device,ot_csharp,ot_X64,ot_CPU,ot_stable':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Training' target='_blank'>Microsoft.ML.OnnxRuntime.Training</a>",

		'ot_linux,ot_on_device,ot_c,ot_X64,ot_CUDA118,ot_stable':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://onnxruntime.ai/docs/build/training.html' target='_blank'>here</a>",

		'ot_linux,ot_on_device,ot_cplusplus,ot_X64,ot_CUDA118,ot_stable':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://onnxruntime.ai/docs/build/training.html' target='_blank'>here</a>",

		'ot_linux,ot_on_device,ot_csharp,ot_X64,ot_CUDA118,ot_stable':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://onnxruntime.ai/docs/build/training.html' target='_blank'>here</a>",

		'ot_linux,ot_on_device,ot_c,ot_X64,ot_CPU,ot_stable':
			"Download .tgz file from&nbsp;<a class='text-blue-500' href='https://github.com/microsoft/onnxruntime/releases' target='_blank'>Github</a> <br/>Refer to <a class='text-blue-500' href='http://www.onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements' target='_blank'>docs</a> for requirements.",

		'ot_windows,ot_on_device,ot_python,ot_X64,ot_CPU,ot_stable':
			'python -m pip install cerberus flatbuffers h5py numpy>=1.16.6 onnx packaging protobuf sympy setuptools>=41.4.0<br/>pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training-cpu',

		'ot_windows,ot_on_device,ot_python,ot_X64,ot_CPU,ot_nightly':
			'python -m pip install cerberus flatbuffers h5py numpy>=1.16.6 onnx packaging protobuf sympy setuptools>=41.4.0<br/>pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-training-cpu',

		'ot_windows,ot_on_device,ot_python,ot_X64,ot_CUDA118,ot_stable':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://onnxruntime.ai/docs/build/training.html' target='_blank'>here</a>",

		'ot_windows,ot_on_device,ot_c,ot_X64,ot_CPU,ot_stable':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Training' target='_blank'>Microsoft.ML.OnnxRuntime.Training</a>",

		'ot_windows,ot_on_device,ot_cplusplus,ot_X64,ot_CPU,ot_stable':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Training' target='_blank'>Microsoft.ML.OnnxRuntime.Training</a>",

		'ot_windows,ot_on_device,ot_csharp,ot_X64,ot_CPU,ot_stable':
			"Install Nuget package&nbsp;<a class='text-blue-500' href='https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.Training' target='_blank'>Microsoft.ML.OnnxRuntime.Training</a>",

		'ot_windows,ot_on_device,ot_c,ot_X64,ot_CUDA118,ot_stable':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://onnxruntime.ai/docs/build/training.html' target='_blank'>here</a>",

		'ot_windows,ot_on_device,ot_cplusplus,ot_X64,ot_CUDA118,ot_stable':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://onnxruntime.ai/docs/build/training.html' target='_blank'>here</a>",

		'ot_windows,ot_on_device,ot_csharp,ot_X64,ot_CUDA118,ot_stable':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://onnxruntime.ai/docs/build/training.html' target='_blank'>here</a>",

		'ot_android,ot_on_device,ot_c,ot_X64,ot_CPU,ot_stable':
			"Follow installation instructions from&nbsp;<a class='text-blue-500' href='https://onnxruntime.ai/docs/install/#install-for-on-device-training' target='_blank'>here</a>",

		'ot_android,ot_on_device,ot_cplusplus,ot_X64,ot_CPU,ot_stable':
			"Follow installation instructions from&nbsp;<a class='text-blue-500' href='https://onnxruntime.ai/docs/install/#install-for-on-device-training' target='_blank'>here</a>",

		'ot_android,ot_on_device,ot_java,ot_X64,ot_CPU,ot_stable':
			"Add a dependency on <a class='text-blue-500' href='https://mvnrepository.com/artifact/com.microsoft.onnxruntime/onnxruntime-training-android' target='_blank'>com.microsoft.onnxruntime:onnxruntime-training-android</a> using Maven/Gradle and refer to the instructions <a class='text-blue-500' href='https://onnxruntime.ai/docs/install/#install-for-on-device-training' target='_blank'>here.</a>",

		'ot_android,ot_on_device,ot_c,ot_X64,ot_CPU,ot_nightly':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://onnxruntime.ai/docs/build/android.html' target='_blank'>here</a>",

		'ot_android,ot_on_device,ot_cplusplus,ot_X64,ot_CPU,ot_nightly':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://onnxruntime.ai/docs/build/android.html' target='_blank'>here</a>",

		'ot_android,ot_on_device,ot_java,ot_X64,ot_CPU,ot_nightly':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://onnxruntime.ai/docs/build/android.html' target='_blank'>here</a>",

		'ot_mac,ot_on_device,ot_python,ot_X64,ot_CPU,ot_stable':
			'python -m pip install cerberus flatbuffers h5py numpy>=1.16.6 onnx packaging protobuf sympy setuptools>=41.4.0<br/>pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT/pypi/simple/ onnxruntime-training-cpu',

		'ot_mac,ot_on_device,ot_python,ot_X64,ot_CPU,ot_nightly':
			'python -m pip install cerberus flatbuffers h5py numpy>=1.16.6 onnx packaging protobuf sympy setuptools>=41.4.0<br/>pip install -i https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/pypi/simple/ onnxruntime-training-cpu',

		'ot_ios,ot_on_device,ot_objc,ot_X64,ot_CPU,ot_stable':
			"Add 'onnxruntime-training-objc' using CocoaPods and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'ot_ios,ot_on_device,ot_c,ot_X64,ot_CPU,ot_stable':
			"Add 'onnxruntime-training-c' using CocoaPods and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'ot_ios,ot_on_device,ot_cplusplus,ot_X64,ot_CPU,ot_stable':
			"Add 'onnxruntime-training-c' using CocoaPods and refer to the <a class='text-blue-500' href='https://onnxruntime.ai/docs/tutorials/mobile/' target='_blank'>mobile deployment guide</a>",

		'ot_ios,ot_on_device,ot_objc,ot_X64,ot_CPU,ot_nightly':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://onnxruntime.ai/docs/build/ios.html' target='_blank'>here</a>",

		'ot_ios,ot_on_device,ot_c,ot_X64,ot_CPU,ot_nightly':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://onnxruntime.ai/docs/build/ios.html' target='_blank'>here</a>",

		'ot_ios,ot_on_device,ot_cplusplus,ot_X64,ot_CPU,ot_nightly':
			"Follow build instructions from&nbsp;<a class='text-blue-500' href='https://onnxruntime.ai/docs/build/ios.html' target='_blank'>here</a>"
	};
	onMount(() => {
		var supportedOperatingSystemsNew = [
			{ key: 'linux', value: 'linux' },
			{ key: 'mac', value: 'macos' },
			{ key: 'win', value: 'windows' },
			{ key: 'web', value: 'web' }
		];

		var opts = {
			os: '',
			architecture: '',
			language: '',
			hardwareAcceleration: ''
		};
		var ot_opts = {
			ot_scenario: '',
			ot_os: '',
			ot_architecture: 'ot_X64',
			ot_language: '',
			ot_hardwareAcceleration: '',
			ot_build: ''
		};

		var os = jq('.os > .r-option');

		var architecture = jq('.architecture > .r-option');
		var language = jq('.language > .r-option');
		var hardwareAcceleration = jq('.hardwareAcceleration > .r-option');

		var ot_os = jq('.ot_os > .r-option');
		var ot_tab = jq('#OT_tab');

		var ot_scenario = jq('.ot_scenario > .r-option');
		var ot_architecture = jq('.ot_architecture > .r-option');
		var ot_language = jq('.ot_language > .r-option');
		var ot_hardwareAcceleration = jq('.ot_hardwareAcceleration > .r-option');
		var ot_build = jq('.ot_build > .r-option');

		var supported = true;
		var ot_defaultSelection = true;

		function checkKeyPress(event) {
			var keycode = event.keyCode ? event.keyCode : event.which;
			if (keycode == '13' || keycode == '32' || (keycode >= '37' && keycode <= '40')) {
				return true;
			} else {
				return false;
			}
		}

		os.on('click', function () {
			selectedOption(os, this, 'os');
		});
		os.on('keypress keyup', function (event) {
			if (checkKeyPress(event)) {
				selectedOption(os, this, 'os');
			}
		});
		ot_os.on('click', function () {
			ot_selectedOption(ot_os, this, 'ot_os');
		});
		ot_os.on('keypress keyup', function (event) {
			if (checkKeyPress(event)) {
				ot_selectedOption(ot_os, this, 'ot_os');
			}
		});
		ot_tab.on('click', function () {
			ot_commandMessage(ot_buildMatcher());
			ot_checkValidity();
		});
		ot_scenario.on('click', function () {
			ot_selectedOption(ot_scenario, this, 'ot_scenario');
		});
		ot_scenario.on('keypress keyup', function (event) {
			if (checkKeyPress(event)) {
				ot_selectedOption(ot_scenario, this, 'ot_scenario');
			}
		});
		ot_build.on('click', function () {
			ot_selectedOption(ot_build, this, 'ot_build');
		});
		ot_build.on('keypress keyup', function (event) {
			if (checkKeyPress(event)) {
				ot_selectedOption(ot_build, this, 'ot_build');
			}
		});
		architecture.on('click', function () {
			selectedOption(architecture, this, 'architecture');
		});
		architecture.on('keypress keyup', function (event) {
			if (checkKeyPress(event)) {
				selectedOption(architecture, this, 'architecture');
			}
		});
		ot_architecture.on('click', function () {
			ot_selectedOption(ot_architecture, this, 'ot_architecture');
		});
		ot_architecture.on('keypress keyup', function (event) {
			if (checkKeyPress(event)) {
				ot_selectedOption(ot_architecture, this, 'ot_architecture');
			}
		});
		language.on('click', function () {
			selectedOption(language, this, 'language');
		});
		language.on('keypress keyup', function (event) {
			if (checkKeyPress(event)) {
				selectedOption(language, this, 'language');
			}
		});
		ot_language.on('click', function () {
			ot_selectedOption(ot_language, this, 'ot_language');
		});
		ot_language.on('keypress keyup', function (event) {
			if (checkKeyPress(event)) {
				ot_selectedOption(ot_language, this, 'ot_language');
			}
		});
		hardwareAcceleration.on('click', function () {
			selectedOption(hardwareAcceleration, this, 'hardwareAcceleration');
		});
		hardwareAcceleration.on('keypress keyup', function (event) {
			if (checkKeyPress(event)) {
				selectedOption(hardwareAcceleration, this, 'hardwareAcceleration');
			}
		});
		ot_hardwareAcceleration.on('click', function () {
			ot_selectedOption(ot_hardwareAcceleration, this, 'ot_hardwareAcceleration');
		});
		ot_hardwareAcceleration.on('keypress keyup', function (event) {
			if (checkKeyPress(event)) {
				ot_selectedOption(ot_hardwareAcceleration, this, 'ot_hardwareAcceleration');
			}
		});

		function checkValidity() {
			var current_os = opts['os'];
			var current_lang = opts['language'];
			var current_arch = opts['architecture'];
			var current_hw = opts['hardwareAcceleration'];

			var valid = Object.getOwnPropertyNames(validCombos);

			//os section
			for (var i = 0; i < os.length; i++) {
				//disable other selections once item in category selected
				// if(os[i].id!=current_os && current_os!=''){
				//     jq(os[i]).addClass("btn-disabled");
				//     continue;
				// }
				var isvalidcombo = false;
				for (var k = 0; k < valid.length; k++) {
					if (
						valid[k].indexOf(os[i].id) != -1 &&
						valid[k].indexOf(current_arch) != -1 &&
						valid[k].indexOf(current_lang) != -1 &&
						valid[k].indexOf(current_hw) != -1
					) {
						isvalidcombo = true;
						break;
					}
				}
				if (isvalidcombo == false && os[i].id != current_os) {
					jq(os[i]).addClass('btn-disabled');
				}
			}

			//language section
			for (var i = 0; i < language.length; i++) {
				//disable other selections once item in category selected
				//  if(language[i].id!=current_lang && current_lang!=''){
				//     jq(language[i]).addClass("btn-disabled");
				//      continue;
				//   }
				var isvalidcombo = false;
				for (var k = 0; k < valid.length; k++) {
					if (
						valid[k].indexOf(current_os) != -1 &&
						valid[k].indexOf(current_arch) != -1 &&
						valid[k].indexOf(language[i].id) != -1 &&
						valid[k].indexOf(current_hw) != -1
					) {
						isvalidcombo = true;
						break;
					}
				}
				if (isvalidcombo == false && language[i].id != current_lang) {
					jq(language[i]).addClass('btn-disabled');
				}
			}

			//architecture section
			for (var i = 0; i < architecture.length; i++) {
				//disable other selections once item in category selected
				//     if(architecture[i].id!=current_arch && current_arch!=''){
				//         jq(architecture[i]).addClass("btn-disabled");
				//         continue;
				//     }
				var isvalidcombo = false;
				for (var k = 0; k < valid.length; k++) {
					if (
						valid[k].indexOf(current_os) != -1 &&
						valid[k].indexOf(architecture[i].id) != -1 &&
						valid[k].indexOf(current_lang) != -1 &&
						valid[k].indexOf(current_hw) != -1
					) {
						isvalidcombo = true;
						break;
					}
				}
				if (isvalidcombo == false && architecture[i].id != current_arch) {
					jq(architecture[i]).addClass('btn-disabled');
				}
			}

			//accelerator section
			for (var i = 0; i < hardwareAcceleration.length; i++) {
				//disable other selections once item in category selected
				//      if(hardwareAcceleration[i].id!=current_hw && current_hw!=''){
				//       jq(hardwareAcceleration[i]).addClass("btn-disabled");
				//       continue;
				// }
				var isvalidcombo = false;

				//go thru all valid options
				for (var k = 0; k < valid.length; k++) {
					if (
						valid[k].indexOf(current_os) != -1 &&
						valid[k].indexOf(current_arch) != -1 &&
						valid[k].indexOf(current_lang) != -1 &&
						valid[k].indexOf(hardwareAcceleration[i].id) != -1
					) {
						isvalidcombo = true;
						break;
					}
				}

				if (isvalidcombo == false && hardwareAcceleration[i].id != current_hw) {
					jq(hardwareAcceleration[i]).addClass('btn-disabled');
				}
			}
		}

		function ot_checkValidity() {
			var current_os = ot_opts['ot_os'];
			var current_scenario = ot_opts['ot_scenario'];
			var current_lang = ot_opts['ot_language'];
			var current_arch = ot_opts['ot_architecture'];
			var current_hw = ot_opts['ot_hardwareAcceleration'];
			var current_build = ot_opts['ot_build'];

			var valid = Object.getOwnPropertyNames(ot_validCombos);

			// scenario section
			for (var i = 0; i < ot_scenario.length; i++) {
				var isvalidcombo = false;
				for (var k = 0; k < valid.length; k++) {
					if (
						valid[k].indexOf(ot_scenario[i].id) != -1 &&
						valid[k].indexOf(current_os) != -1 &&
						valid[k].indexOf(current_arch) != -1 &&
						valid[k].indexOf(current_lang) != -1 &&
						valid[k].indexOf(current_hw) != -1 &&
						valid[k].indexOf(current_build) != -1
					) {
						isvalidcombo = true;
						break;
					}
				}
				if (isvalidcombo == false && ot_scenario[i].id != current_scenario) {
					jq(ot_scenario[i]).addClass('btn-disabled');
				}
			}

			//os section
			for (var i = 0; i < ot_os.length; i++) {
				//disable other selections once item in category selected
				// if(ot_os[i].id!=current_os && current_os!=''){
				//     jq(ot_os[i]).addClass("btn-disabled");
				//     continue;
				// }
				var isvalidcombo = false;
				for (var k = 0; k < valid.length; k++) {
					if (
						valid[k].indexOf(current_scenario) != -1 &&
						valid[k].indexOf(ot_os[i].id) != -1 &&
						valid[k].indexOf(current_arch) != -1 &&
						valid[k].indexOf(current_lang) != -1 &&
						valid[k].indexOf(current_hw) != -1 &&
						valid[k].indexOf(current_build) != -1
					) {
						isvalidcombo = true;
						break;
					}
				}
				if (isvalidcombo == false && ot_os[i].id != current_os) {
					jq(ot_os[i]).addClass('btn-disabled');
				}
			}

			//language section
			for (var i = 0; i < ot_language.length; i++) {
				//disable other selections once item in category selected
				//  if(ot_language[i].id!=current_lang && current_lang!=''){
				//     jq(ot_language[i]).addClass("btn-disabled");
				//      continue;
				//   }
				var isvalidcombo = false;
				for (var k = 0; k < valid.length; k++) {
					if (
						valid[k].indexOf(current_scenario) != -1 &&
						valid[k].indexOf(current_os) != -1 &&
						valid[k].indexOf(current_arch) != -1 &&
						valid[k].indexOf(ot_language[i].id) != -1 &&
						valid[k].indexOf(current_hw) != -1 &&
						valid[k].indexOf(current_build) != -1
					) {
						isvalidcombo = true;
						break;
					}
				}
				if (isvalidcombo == false && ot_language[i].id != current_lang) {
					jq(ot_language[i]).addClass('btn-disabled');
				}
			}

			//architecture section
			for (var i = 0; i < ot_architecture.length; i++) {
				//      //disable other selections once item in category selected
				// if(ot_architecture[i].id!=current_arch && current_arch!=''){
				//     jq(ot_architecture[i]).addClass("btn-disabled");
				//     continue;
				// }
				var isvalidcombo = false;
				for (var k = 0; k < valid.length; k++) {
					if (
						valid[k].indexOf(current_scenario) != -1 &&
						valid[k].indexOf(current_os) != -1 &&
						valid[k].indexOf(ot_architecture[i].id) != -1 &&
						valid[k].indexOf(current_lang) != -1 &&
						valid[k].indexOf(current_hw) != -1 &&
						valid[k].indexOf(current_build) != -1
					) {
						isvalidcombo = true;
						break;
					}
				}
				if (isvalidcombo == false && ot_architecture[i].id != current_arch) {
					jq(ot_architecture[i]).addClass('btn-disabled');
				}
			}

			//accelerator section
			for (var i = 0; i < ot_hardwareAcceleration.length; i++) {
				//disable other selections once item in category selected
				//      if(ot_hardwareAcceleration[i].id!=current_hw && current_hw!=''){
				//       jq(ot_hardwareAcceleration[i]).addClass("btn-disabled");
				//       continue;
				// }
				var isvalidcombo = false;
				for (var k = 0; k < valid.length; k++) {
					if (
						valid[k].indexOf(current_scenario) != -1 &&
						valid[k].indexOf(current_os) != -1 &&
						valid[k].indexOf(current_arch) != -1 &&
						valid[k].indexOf(current_lang) != -1 &&
						valid[k].indexOf(ot_hardwareAcceleration[i].id) != -1 &&
						valid[k].indexOf(current_build) != -1
					) {
						isvalidcombo = true;
						break;
					}
				}

				if (isvalidcombo == false && ot_hardwareAcceleration[i].id != current_hw) {
					jq(ot_hardwareAcceleration[i]).addClass('btn-disabled');
				}
			}

			// build section
			for (var i = 0; i < ot_build.length; i++) {
				var isvalidcombo = false;
				for (var k = 0; k < valid.length; k++) {
					if (
						valid[k].indexOf(current_scenario) != -1 &&
						valid[k].indexOf(current_os) != -1 &&
						valid[k].indexOf(current_arch) != -1 &&
						valid[k].indexOf(current_lang) != -1 &&
						valid[k].indexOf(current_hw) != -1 &&
						valid[k].indexOf(ot_build[i].id) != -1
					) {
						isvalidcombo = true;
						break;
					}
				}
				if (isvalidcombo == false && ot_build[i].id != current_build) {
					jq(ot_build[i]).addClass('btn-disabled');
				}
			}
		}

		function mark_unsupported(selection, training) {
			if (training == true) {
				for (var i = 0; i < selection.length; i++) {
					if (selection[i].id.indexOf('ot_') != -1) {
						jq(selection[i]).addClass('unsupported');
					}
				}
			} else {
				for (var i = 0; i < selection.length; i++) {
					if (selection[i].id.indexOf('ot_') == -1) {
						jq(selection[i]).addClass('unsupported');
					}
				}
			}
		}

		function selectedOption(option, selection, category) {
			//allow deselect
			if (selection.id == opts[category]) {
				jq(selection).removeClass('selected');
				jq(selection).removeClass('unsupported');
				jq(selection).removeClass('btn-primary');
				opts[category] = '';
			} else {
				jq(option).removeClass('selected');
				jq(option).removeClass('unsupported');
				jq(option).removeClass('btn-primary');
				jq(selection).addClass('selected');
				jq(selection).addClass('btn-primary');
				opts[category] = selection.id;
			}

			resetOptions();

			var all_selected = document.getElementsByClassName('selected r-option');

			//get list of supported combos
			var isSupported = commandMessage(buildMatcher());

			//mark unsupported for selected elements
			if (isSupported == false) {
				mark_unsupported(all_selected, false);
			} else {
				for (var i = 0; i < all_selected.length; i++) {
					jq(all_selected[i]).removeClass('unsupported');
				}
			}

			checkValidity();

			//if full selection is valid, don't btn-disabled out other options
			if (
				opts['os'] != '' &&
				opts['architecture'] != '' &&
				opts['hardwareAcceleration'] != '' &&
				opts['language'] != '' &&
				isSupported == true
			) {
				// console.log(opts);
				resetOptions();
			}
		}

		function ot_selectedOption(option, selection, category) {
			//allow deselect, disable for architecture since they only have 1 item
			if (selection.id == ot_opts[category]) {
				jq(selection).removeClass('selected');
				jq(selection).removeClass('unsupported');
				jq(selection).removeClass('btn-primary');
				ot_opts[category] = '';
			} else {
				jq(option).removeClass('selected');
				jq(option).removeClass('unsupported');
				jq(option).removeClass('btn-primary');
				jq(selection).addClass('selected');
				jq(selection).addClass('btn-primary');
				ot_opts[category] = selection.id;
			}

			ot_resetOptions();

			var all_selected = document.getElementsByClassName('selected r-option');
			var isSupported = ot_commandMessage(ot_buildMatcher());

			//mark unsupported combos
			if (isSupported == false) {
				mark_unsupported(all_selected, true);
			} else {
				for (var i = 0; i < all_selected.length; i++) {
					jq(all_selected[i]).removeClass('unsupported');
				}
			}

			ot_checkValidity();

			//if full selection is valid, don't btn-disabled out other options
			if (
				ot_opts['scenario'] != '' &&
				ot_opts['os'] != '' &&
				ot_opts['architecture'] != '' &&
				ot_opts['hardwareAcceleration'] != '' &&
				ot_opts['language'] != '' &&
				isSupported == true
			) {
				// console.log(opts);
				// ot_resetOptions();
			}
		}

		function resetOptions() {
			for (var i = 0; i < os.length; i++) {
				jq(os[i]).removeClass('btn-disabled');
			}
			for (var i = 0; i < language.length; i++) {
				jq(language[i]).removeClass('btn-disabled');
			}
			for (var i = 0; i < architecture.length; i++) {
				jq(architecture[i]).removeClass('btn-disabled');
			}
			for (var i = 0; i < hardwareAcceleration.length; i++) {
				jq(hardwareAcceleration[i]).removeClass('btn-disabled');
			}
		}

		function ot_resetOptions() {
			for (var i = 0; i < ot_os.length; i++) {
				jq(ot_os[i]).removeClass('btn-disabled');
			}
			for (var i = 0; i < ot_scenario.length; i++) {
				jq(ot_scenario[i]).removeClass('btn-disabled');
			}
			for (var i = 0; i < ot_language.length; i++) {
				jq(ot_language[i]).removeClass('btn-disabled');
			}
			for (var i = 0; i < ot_architecture.length; i++) {
				jq(ot_architecture[i]).removeClass('btn-disabled');
			}
			for (var i = 0; i < ot_hardwareAcceleration.length; i++) {
				jq(ot_hardwareAcceleration[i]).removeClass('btn-disabled');
			}
			for (var i = 0; i < ot_build.length; i++) {
				jq(ot_build[i]).removeClass('btn-disabled');
			}
			ot_defaultSelection = false;
		}

		function display(selection, id, category) {
			var container = document.getElementById(id);
			// Check if there's a container to display the selection
			if (container === null) {
				return;
			}
			var elements = container.getElementsByClassName(category);
			for (var i = 0; i < elements.length; i++) {
				if (elements[i].classList.contains(selection)) {
					jq(elements[i]).addClass('selected');
					jq(elements[i]).addClass('btn-primary');
				} else {
					jq(elements[i]).removeClass('selected');
					jq(elements[i]).removeClass('btn-primary');
				}
			}
		}

		function buildMatcher() {
			return (
				opts.os + ',' + opts.language + ',' + opts.architecture + ',' + opts.hardwareAcceleration
			);
		}

		function ot_buildMatcher() {
			return (
				ot_opts.ot_os +
				',' +
				ot_opts.ot_scenario +
				',' +
				ot_opts.ot_language +
				',' +
				ot_opts.ot_architecture +
				',' +
				ot_opts.ot_hardwareAcceleration +
				',' +
				ot_opts.ot_build
			);
		}

		function ot_commandMessage(key) {
			jq('#ot_command').removeClass('valid');
			jq('#ot_command').removeClass('invalid');

			if (
				ot_opts['ot_os'] == '' ||
				ot_opts['ot_scenario'] == '' ||
				ot_opts['ot_architecture'] == '' ||
				ot_opts['ot_language'] == '' ||
				ot_opts['ot_hardwareAcceleration'] == '' ||
				ot_opts['ot_build'] == ''
			) {
				jq('#ot_command span').html('Please select a combination of resources');
			} else if (!ot_validCombos.hasOwnProperty(key)) {
				jq('#ot_command span').html('This combination is not supported. Make another selection.');
				jq('#ot_command').addClass('invalid');
				return false;
			} else {
				jq('#ot_command span').html(ot_validCombos[key]);
				jq('#ot_command').addClass('valid');
				return true;
			}
		}

		function commandMessage(key) {
			jq('#command').removeClass('valid');
			jq('#command').removeClass('invalid');

			if (opts['os'] == 'web' && opts['language'] == 'JS' && validCombos.hasOwnProperty(key)) {
				jq('#command span').html(validCombos[key]);
				jq('#command').addClass('valid');
				return true;
			} else if (
				opts['os'] == '' ||
				opts['architecture'] == '' ||
				opts['language'] == '' ||
				opts['hardwareAcceleration'] == ''
			) {
				jq('#command span').html('Please select a combination of resources');
			} else if (!validCombos.hasOwnProperty(key)) {
				jq('#command span').html('This combination is not supported. Make another selection.');
				jq('#command').addClass('invalid');
				return false;
			} else {
				jq('#command span').html(validCombos[key]);
				jq('#command').addClass('valid');
				return true;
			}
		}

		//Accesibility Get started tabel
		var KEYCODE = {
			DOWN: 40,
			LEFT: 37,
			RIGHT: 39,
			SPACE: 32,
			UP: 38
		};

		window.addEventListener('load', function () {
			var radiobuttons = document.querySelectorAll('[role=option]');
			for (var i = 0; i < radiobuttons.length; i++) {
				var rb = radiobuttons[i];
				rb.addEventListener('click', clickRadioGroup);
				rb.addEventListener('keydown', keyDownRadioGroup);
				rb.addEventListener('focus', focusRadioButton);
				rb.addEventListener('blur', blurRadioButton);
			}
		});

		function firstRadioButton(node) {
			var first = node.parentNode.firstChild;
			while (first) {
				if (first.nodeType === Node.ELEMENT_NODE) {
					if (first.getAttribute('role') === 'option') return first;
				}
				first = first.nextSibling;
			}
			return null;
		}

		function lastRadioButton(node) {
			var last = node.parentNode.lastChild;
			while (last) {
				if (last.nodeType === Node.ELEMENT_NODE) {
					if (last.getAttribute('role') === 'option') return last;
				}
				last = last.previousSibling;
			}
			return last;
		}

		function nextRadioButton(node) {
			var next = node.nextSibling;
			while (next) {
				if (next.nodeType === Node.ELEMENT_NODE) {
					if (next.getAttribute('role') === 'option') return next;
				}
				next = next.nextSibling;
			}
			return null;
		}

		function previousRadioButton(node) {
			var prev = node.previousSibling;
			while (prev) {
				if (prev.nodeType === Node.ELEMENT_NODE) {
					if (prev.getAttribute('role') === 'option') return prev;
				}
				prev = prev.previousSibling;
			}
			return null;
		}

		function getImage(node) {
			var child = node.firstChild;
			while (child) {
				if (child.nodeType === Node.ELEMENT_NODE) {
					if (child.tagName === 'IMG') return child;
				}
				child = child.nextSibling;
			}
			return null;
		}

		function setRadioButton(node, state) {
			var image = getImage(node);
			if (state == 'true') {
				node.setAttribute('aria-selected', 'true');
				// jq(node).trigger()
				node.tabIndex = 0;
				node.focus();
			} else {
				node.setAttribute('aria-selected', 'false');
				node.tabIndex = -1;
			}
		}

		function clickRadioGroup(event) {
			var type = event.type;
			if (type === 'click') {
				var node = event.currentTarget;
				var radioButton = firstRadioButton(node);
				while (radioButton) {
					setRadioButton(radioButton, 'false');
					radioButton = nextRadioButton(radioButton);
				}
				setRadioButton(node, 'true');
				event.preventDefault();
				event.stopPropagation();
			}
		}

		function keyDownRadioGroup(event) {
			var type = event.type;
			var next = false;
			if (type === 'keydown') {
				var node = event.currentTarget;
				switch (event.keyCode) {
					case KEYCODE.DOWN:
					case KEYCODE.RIGHT:
						var next = nextRadioButton(node);
						if (!next) next = firstRadioButton(node); //if node is the last node, node cycles to first.
						break;
					case KEYCODE.UP:
					case KEYCODE.LEFT:
						next = previousRadioButton(node);
						if (!next) next = lastRadioButton(node); //if node is the last node, node cycles to first.
						break;
					case KEYCODE.SPACE:
						next = node;
						break;
				}
				if (next) {
					var radioButton = firstRadioButton(node);
					while (radioButton) {
						setRadioButton(radioButton, 'false');
						radioButton = nextRadioButton(radioButton);
					}
					setRadioButton(next, 'true');
					event.preventDefault();
					event.stopPropagation();
				}
			}
		}

		function focusRadioButton(event) {
			event.currentTarget.className += ' focus';
			document.getElementById('command').innerHTML;
		}

		function blurRadioButton(event) {
			event.currentTarget.className = event.currentTarget.className.replace(' focus', '');
		}

		jq(document).ready(function () {
			jq(".tbl_tablist li[role='tab']").click(function () {
				jq(".tbl_tablist li[role='tab']:not(this)").attr('aria-selected', 'false');
				jq(this).attr('aria-selected', 'true');
				var tabpanid = jq(this).attr('aria-controls');
				var tabpan = jq('#' + tabpanid);
				jq("div[role='tabpanel']:not(tabpan)").attr('aria-hidden', 'true');
				jq("div[role='tabpanel']:not(tabpan)").addClass('hidden');

				tabpan.removeClass('hidden');
				tabpan.attr('aria-hidden', 'false');
			});

			jq(".tbl_tablist li[role='tab']").keydown(function (ev) {
				if (ev.which == 13) {
					jq(this).click();
				}
			});

			//This adds keyboard function that pressing an arrow left or arrow right from the tabs toggel the tabs.
			jq(".tbl_tablist li[role='tab']").keydown(function (ev) {
				if (ev.which == 39 || ev.which == 37) {
					var selected = jq(this).attr('aria-selected');
					if (selected == 'true') {
						jq("li[aria-selected='false']").attr('aria-selected', 'true').focus();
						jq(this).attr('aria-selected', 'false');

						var tabpanid = jq("li[aria-selected='true']").attr('aria-controls');
						var tabpan = jq('#' + tabpanid);
						jq("div[role='tabpanel']:not(tabpan)").attr('aria-hidden', 'true');
						jq("div[role='tabpanel']:not(tabpan)").addClass('hidden');

						tabpan.attr('aria-hidden', 'false');
						tabpan.removeClass('hidden');
					}
				}
			});
		});

		// Modal Extension
		// ===============================

		// jq('.modal-dialog').attr({ role: 'document' });
		// var modalhide = jq.fn.modal.Constructor.prototype.hide;
		// jq.fn.modal.Constructor.prototype.hide = function () {
		// 	modalhide.apply(this, arguments);
		// 	jq(document).off('keydown.bs.modal');
		// };

		// var modalfocus = jq.fn.modal.Constructor.prototype.enforceFocus;
		// jq.fn.modal.Constructor.prototype.enforceFocus = function () {
		// 	var jqcontent = this.jqelement.find('.modal-content');
		// 	var focEls = jqcontent.find(':tabbable'),
		// 		jqlastEl = jq(focEls[focEls.length - 1]),
		// 		jqfirstEl = jq(focEls[0]);
		// 	jqlastEl.on(
		// 		'keydown.bs.modal',
		// 		jq.proxy(function (ev) {
		// 			if (ev.keyCode === 9 && !(ev.shiftKey | ev.ctrlKey | ev.metaKey | ev.altKey)) {
		// 				// TAB pressed
		// 				ev.preventDefault();
		// 				jqfirstEl.focus();
		// 			}
		// 		}, this)
		// 	);
		// 	jqfirstEl.on(
		// 		'keydown.bs.modal',
		// 		jq.proxy(function (ev) {
		// 			if (ev.keyCode === 9 && ev.shiftKey) {
		// 				// SHIFT-TAB pressed
		// 				ev.preventDefault();
		// 				jqlastEl.focus();
		// 			}
		// 		}, this)
		// 	);
		// 	modalfocus.apply(this, arguments);
		// };

		jq(function () {
			var tabs = jq('.custom-tab');

			// For each individual tab DIV, set class and aria role attributes, and hide it
			jq(tabs)
				.find('.tab-content > div.tab-pane')
				.attr({
					class: 'tabPanel',
					role: 'tabpanel',
					'aria-hidden': 'true'
				})
				.hide();

			// Get the list of tab links
			var tabsList = tabs.find('ul:first').attr({
				role: 'tablist'
			});

			// For each item in the tabs list...
			jq(tabsList)
				.find('li > a')
				.each(function (a) {
					var tab = jq(this);

					// Create a unique id using the tab link's href
					var tabId = 'tab-' + tab.attr('href').slice(1);

					// Assign tab id, aria and tabindex attributes to the tab control, but do not remove the href
					tab
						.attr({
							id: tabId,
							role: 'tab',
							'aria-selected': 'false'
							//   "tabindex": "-1"
						})
						.parent()
						.attr('role', 'presentation');

					// Assign aria attribute to the relevant tab panel
					jq(tabs).find('.tabPanel').eq(a).attr('aria-labelledby', tabId);

					// Set the click event for each tab link
					tab.click(function (e) {
						// Prevent default click event
						e.preventDefault();

						// Change state of previously selected tabList item
						jq(tabsList).find('> li.active').removeClass('active').find('> a').attr({
							'aria-selected': 'false'
							//   "tabindex": "-1"
						});

						// Hide previously selected tabPanel
						jq(tabs).find('.tabPanel:visible').attr('aria-hidden', 'true').hide();

						// Show newly selected tabPanel
						jq(tabs).find('.tabPanel').eq(tab.parent().index()).attr('aria-hidden', 'false').show();

						// Set state of newly selected tab list item
						tab
							.attr({
								'aria-selected': 'true',
								tabindex: '0'
							})
							.parent()
							.addClass('active');
						tab.focus();
					});
				});

			// Set keydown events on tabList item for navigating tabs
			jq(tabsList).delegate('a', 'keydown', function (e) {
				var tab = jq(this);
				switch (e.which) {
					case 37:
						//case 38:
						if (tab.parent().prev().length != 0) {
							tab.parent().prev().find('> a').click();
						} else {
							jq(tabsList).find('li:last > a').click();
						}
						break;
					case 39:
						//case 40:
						if (tab.parent().next().length != 0) {
							tab.parent().next().find('> a').click();
						} else {
							jq(tabsList).find('li:first > a').click();
						}
						break;
				}
			});

			// Show the first tabPanel
			jq(tabs).find('.tabPanel:first').attr('aria-hidden', 'false').show();

			// Set state for the first tabsList li
			jq(tabsList).find('li:first').addClass('active').find(' > a').attr({
				'aria-selected': 'true',
				tabindex: '0'
			});
		});
	});
	let tabs = ['Optimize Inferencing', 'Optimize Training'];
	let activeTab = 0;
</script>

<section class="pb-4" id="getStartedTable">
	<!-- <div class="pb-3">
		<h2 class="text-3xl font-bold">Installation Instructions</h2>
	</div> -->
	<noscript>
		<div class="javascript-is-disabled flex">
			<div class="w-full">
				<div class="p-4 bg-red-200 rounded">
					<h2>Please enable JavaScript to use the interactive installation guide.</h2>
					<p class="mb-0">
						Need help enabling JavaScript? Follow the instructions
						<a
							href="https://www.whatismybrowser.com/guides/how-to-enable-javascript/auto"
							target="_blank">here</a
						>.
					</p>
				</div>
			</div>
		</div>
	</noscript>
	<p class="pb-4">
		Select the configuration you want to use and run the corresponding installation script.
	</p>
	<div>
		<div class="tabs">
			{#each tabs as tab, index}
				<li
					class="nav-item tab tab-bordered tab-lg"
					class:tab-active={activeTab == index}
					on:click={() => (activeTab = index)}
					on:keypress={() => (activeTab = index)}
					id={index == 0 ? 'OI_tab' : 'OT_tab'}
					data-toggle="pill"
					aria-controls="panel{index}"
					tabindex={0}
					role="tab"
				>
					{tab}
				</li>
			{/each}
		</div>
		<div
			class:hidden={activeTab != 0}
			id="panel1"
			aria-labelledby="tab1"
			role="tabpanel"
			aria-hidden="false"
		>
			<div class="grid grid-cols-5 gap-4 container mx-auto p-5">
				<div class="col-span-1 bg-success rounded p-2">
					<div class="r-heading text-xl">
						<h3 id="selectOS">Platform</h3>
						<p id="decriptionOS" class="sr-only">Platform list contains six items</p>
					</div>
				</div>
				<div class="col-span-4">
					<div
						class="r-content os grid grid-cols-6 gap-4"
						role="listbox"
						id="listbox-1"
						aria-labelledby="selectOS"
						aria-describedby="decriptionOS"
					>
						{#each platforms as platform, i}
							<a
								class="r-option version btn rounded"
								role="option"
								aria-selected="false"
								aria-label={platform}
								id={platformIDs[i]}>{platform}</a
							>
						{/each}
					</div>
				</div>

				<div class="col-span-1 bg-success rounded p-2 text-xl">
					<div class="r-heading">
						<h3 id="selectLanguage">API</h3>
						<p id="decriptionLanguage" class="sr-only">API list contains eight items</p>
					</div>
				</div>
				<div class="col-span-4">
					<div
						role="listbox"
						id="listbox-2"
						aria-labelledby="selectLanguage"
						aria-describedby="decriptionLanguage"
						class="r-content language grid grid-cols-8 gap-4"
					>
						{#each apis as api, i}
							<a
								class="r-option btn rounded w-full"
								role="option"
								aria-selected="false"
								aria-label={api}
								id={apiIDs[i]}>{api}</a
							>
						{/each}
					</div>
				</div>
				<div class="col-span-1 bg-success rounded p-2 text-xl">
					<div class="r-heading">
						<h3 id="selectArchitecture" class="self-center">Architecture</h3>
						<p id="decriptionArchitecture" class="sr-only">Architecture list contains five items</p>
					</div>
				</div>
				<div class="col-span-4">
					<div
						class="r-content architecture grid grid-cols-5 gap-4"
						role="listbox"
						id="listbox-3"
						aria-labelledby="selectArchitecture"
						aria-describedby="decriptionArchitecture"
					>
						{#each architectures as architecture, i}
							<a
								class="r-option join-item btn rounded"
								role="option"
								aria-selected="false"
								aria-label={architecture}
								id={architecturesIDs[i]}>{architecture}</a
							>
						{/each}
					</div>
				</div>
				<div class="col-span-1 bg-success rounded p-2 text-xl">
					<div class="r-heading">
						<h3 id="selectHardwareAcceleration">Hardware Acceleration</h3>
						<p id="decriptionHardwareAcceleration" class="sr-only">
							Hardware Acceleration list contains seventeen items
						</p>
					</div>
				</div>
				<div class="col-span-4">
					<div
						class="r-content hardwareAcceleration grid grid-cols-5 gap-4"
						role="listbox"
						id="listbox-4"
						aria-labelledby="selectHardwareAcceleration"
						aria-describedby="decriptionHardwareAcceleration"
					>
						{#each hardwareAccelerations as hardware, i}
							<a
								class="r-option version join-item btn rounded"
								role="option"
								aria-selected="false"
								aria-label={hardware}
								id={hardwareAccelerationIDs[i]}>{hardware}</a
							>
						{/each}
					</div>
				</div>
				<div class="col-span-1 r-heading bg-success rounded p-2 text-xl">
					<h3 id="selectRunCommand">Installation Instructions</h3>
				</div>
				<div class="col-span-4 r-content bg-base-300 rounded">
					<div class="command-container p-4" id="command" role="status">
						<span class=""> Please select a combination of resources </span>
					</div>
				</div>
			</div>
		</div>

		<div
			id="panel2"
			class="grid grid-cols-5 gap-4 container mx-auto p-5"
			aria-labelledby="tab2"
			role="tabpanel"
			aria-hidden="true"
			class:hidden={activeTab != 1}
		>
			<div class="col-span-1 bg-success rounded p-2">
				<div class="r-heading text-xl">
					<h3 id="ot_selectScenario">Scenario</h3>
					<p id="ot_decriptionScenario" class="sr-only">Scenario list contains two items</p>
				</div>
			</div>
			<div class="col-span-4 w-full">
				<div
					class="ot_scenario r-content grid grid-cols-2 gap-4"
					role="listbox"
					aria-labelledby="ot_selectScenario"
					aria-describedby="ot_decriptionScenario"
				>
					{#each TrainingScenarios as trainingscenario, i}
						<a
							class="r-option version join-item btn rounded"
							role="option"
							aria-selected="false"
							aria-label={trainingscenario}
							id={TrainingScenarioIds[i]}>{trainingscenario}</a
						>
					{/each}
				</div>
			</div>
			<div class="col-span-1 bg-success r-heading rounded p-2 text-xl">
				<h3 id="ot_selectOS">Platform</h3>
				<p id="ot_decriptionOS" class="sr-only">Platform list contains five items</p>
			</div>
			<div
				class="col-span-4 w-full r-content"
				role="listbox"
				id="ot_listbox-1"
				aria-labelledby="ot_selectOS"
				aria-describedby="ot_decriptionOS"
			>
				<div class="grid grid-cols-5 gap-4 ot_os">
					{#each TrainingPlatforms as trainingplatform, i}
						<a
							class="r-option version join-item btn rounded"
							role="option"
							aria-selected="false"
							aria-label={trainingplatform}
							id={TrainingPlatformIds[i]}>{trainingplatform}</a
						>
					{/each}
				</div>
			</div>

			<div class="col-span-1 bg-success r-heading rounded p-2 text-xl">
				<h3 id="ot_selectLanguage">API</h3>
				<p id="ot_decriptionLanguage" class="sr-only">API list contains six items</p>
			</div>
			<div
				class="col-span-4 w-full r-content"
				role="listbox"
				id="ot_listbox-2"
				aria-labelledby="ot_selectLanguage"
				aria-describedby="ot_decriptionLanguage"
			>
				<div class="grid grid-cols-6 gap-4 ot_language">
					{#each TrainingAPIs as trainingapi, i}
						<a
							class="r-option version join-item btn rounded"
							role="option"
							aria-selected="false"
							aria-label={trainingapi}
							id={TrainingAPIIds[i]}>{trainingapi}</a
						>
					{/each}
				</div>
			</div>

			<div class="col-span-1 bg-success r-heading rounded p-2 text-xl">
				<h3 id="ot_selectHardwareAcceleration">Hardware Acceleration</h3>
				<p id="ot_decriptionHardwareAcceleration" class="sr-only">
					Hardware Acceleration list contains three items
				</p>
			</div>
			<div
				class="col-span-4 w-full r-content"
				role="listbox"
				id="ot_listbox-4"
				aria-labelledby="ot_selectHardwareAcceleration"
				aria-describedby="ot_decriptionHardwareAcceleration"
			>
				<div class="grid grid-cols-3 gap-4 ot_hardwareAcceleration">
					{#each TrainingVersions as version, i}
						<a
							class="r-option version join-item btn rounded"
							role="option"
							aria-selected="false"
							aria-label={version}
							id={TrainingVersionIds[i]}>{version}</a
						>
					{/each}
				</div>
			</div>
			<div class="col-span-1 bg-success rounded p-2 text-xl">
				<h3 id="ot_selectBuild">Build</h3>
				<p id="ot_decriptionBuild" class="sr-only">Build list contains two items</p>
			</div>
			<div
				class="col-span-4 r-content"
				role="listbox"
				id="ot_listbox-4"
				aria-labelledby="ot_selectBuild"
				aria-describedby="ot_decriptionBuild"
			>
				<div class="grid grid-cols-2 gap-4 ot_build">
					{#each TrainingBuilds as build, i}
						<a
							class="r-option version join-item btn rounded"
							role="option"
							aria-selected="false"
							aria-label={build}
							id={TrainingBuildIds[i]}>{build}</a
						>
					{/each}
				</div>
			</div>
			<div class="col-span-1 bg-success rounded p-2">
				<h3 class="text-xl r-heading" id="ot_selectRunCommand">Installation Instructions</h3>
			</div>
			<div class="col-span-4 w-full bg-base-300 rounded">
				<div class="r-content">
					<div class="command-container p-4" id="ot_command" role="status">
						<span> Please select a combination of resources </span>
					</div>
				</div>
			</div>
		</div>
	</div>
</section>
