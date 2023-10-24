<svelte:head>
	<meta
		name="description"
		content="Learn about how ONNX Runtime can be used in various Machine Learning Inferencing scenarios."
	/>
</svelte:head>
<script lang="ts">
	import LandingHero from '../components/landing-hero.svelte';
	import ImageInference2 from '../../images/undraw/image_inference2.svelte';
	import ImageInference1 from '../../images/undraw/image_inference1.svg';
	import OnnxLight from '../../images/ONNX-Light.svelte';
	
	const title = 'ONNX Runtime for Inferencing';
	const description =
		'ONNX Runtime Inference powers machine learning models in key Microsoft products and services across Office, Azure, Bing, as well as thousands of community projects.';
	const imgsrc = 'onnxruntimelogo';
	const imgalt = 'ONNX Runtime Logo';
</script>

<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-1 md:grid-cols-3 gap-4 lg:gap-10">
		<div class="col-span-2">
			<h1 class="text-4xl">{title}</h1>
			<br /><br />
			<p class="text-xl">
				{description}
			</p>
			<br />
			<a href="https://onnxruntime.ai/docs/build/inferencing.html" class="btn btn-primary"
				>Learn how to build ONNX Runtime for inferencing →</a
			>
		</div>
		<div class="m-auto">
			<OnnxLight height={250} width={250} />
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-4">Benefits</h1>
	<div class="grid gap-10 grid-cols-1 md:grid-cols-3 pb-10">
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Cost savings vs. running models in the cloud</h2>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Better latency and availability than request in the cloud</h2>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">More privacy since data stays on device</h2>
			</div>
		</div>
	</div>
	<div class="grid gap-10 grid-cols-1 md:grid-cols-2 mx-auto">
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">
					Easily enable cross-platform portability with the same implementation through the browser
				</h2>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">
					Simplify the distribution experience without needing any additional libraries and driver
					installations
				</h2>
			</div>
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-4">Use Cases</h1>
	<div class="grid gap-10 grid-cols-1 md:grid-cols-2 lg:grid-cols-4 mx-auto">
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Improve inference performance for a wide variety of ML models</h2>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Run on different hardware and operating systems</h2>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">Train in Python but deploy into a C#/C++/Java app</h2>
			</div>
		</div>
		<div class="card bg-base-300">
			<div class="card-body items-center text-center">
				<h2 class="card-title">
					Train and perform inference with models created in different frameworks
				</h2>
			</div>
		</div>
	</div>
</div>
<LandingHero
	title="ONNX Runtime Mobile"
	description="ONNX Runtime Mobile allows you to run model inferencing on mobile devices (iOS and Android)."
	imgsrc={ImageInference1}
	imgalt=""
/>
<div class="container mx-auto px-10 my-10">
	<div class="mx-auto bg-secondary rounded m-4 p-4">
		<h1 class="text-3xl mb-2">Examples</h1>
		<div class="grid gap-10 grid-cols-1 md:grid-cols-2 lg:grid-cols-4">
			<div class="card bg-base-300">
				<div class="card-body items-center text-center">
					<h2 class="card-title">Image Classification</h2>
					The example app uses image classification which is able to continuously classify the objects
					it sees from the device's camera in real-time and displays the most probable inference results
					on the screen.
					<div class="card-actions mt-auto mb-2 justify-center">
						<a
							href="https://github.com/microsoft/onnxruntime-inference-examples/blob/main/mobile/examples/image_classification/android"
							class="btn btn-primary">Android Image Classifier →</a
						>
					</div>
				</div>
			</div>
			<div class="card bg-base-300">
				<div class="card-body items-center text-center">
					<h2 class="card-title">Speech Recognition</h2>
					The example app uses speech recognition to transcribe speech from audio recorded by the device.
					<div class="card-actions mt-auto mb-2 justify-center">
						<a
							href="https://github.com/microsoft/onnxruntime-inference-examples/blob/main/mobile/examples/speech_recognition/android"
							class="btn btn-primary">Android Speech Recognition →</a
						>
						<a
							href="https://github.com/microsoft/onnxruntime-inference-examples/blob/main/mobile/examples/speech_recognition/ios"
							class="btn btn-primary">iOS Speech Recognition →</a
						>
					</div>
				</div>
			</div>
			<div class="card bg-base-300">
				<div class="card-body items-center text-center">
					<h2 class="card-title">Object Detection</h2>
					The example app uses object detection which is able to continuously detect the objects in the
					frames seen by your iOS device's back camera and display the detected object bounding boxes,
					detected class and corresponding inference confidence on the screen.
					<div class="card-actions mt-auto mb-2 justify-center">
						<a
							href="https://github.com/microsoft/onnxruntime-inference-examples/blob/main/mobile/examples/object_detection/android"
							class="btn btn-primary">Android Object Detection →</a
						>
						<a
							href="https://github.com/microsoft/onnxruntime-inference-examples/blob/main/mobile/examples/object_detection/ios"
							class="btn btn-primary">iOS Object Detection →</a
						>
					</div>
				</div>
			</div>
			<div class="card bg-base-300">
				<div class="card-body items-center text-center">
					<h2 class="card-title">Question Answering</h2>
					The example app gives a demo of introducing question answering models with pre/post processing
					into mobile scenario. Currently supports on platform Android and iOS.
					<div class="card-actions mt-auto mb-2 justify-center">
						<a
							href="https://github.com/microsoft/onnxruntime-inference-examples/blob/main/mobile/examples/question_answering/android"
							class="btn btn-primary">Android Question Answering →</a
						>
						<a
							href="https://github.com/microsoft/onnxruntime-inference-examples/blob/main/mobile/examples/question_answering/ios"
							class="btn btn-primary">iOS Question Answering →</a
						>
					</div>
				</div>
			</div>
		</div>
		<a
			href="https://github.com/microsoft/onnxruntime-inference-examples/tree/main/mobile"
			class="text-2xl text-blue-200"
			>Check out more examples of ONNX Runtime Mobile in action on GitHub. →</a
		>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-1 md:grid-cols-3 gap-4 lg:gap-10">
		<div class="col-span-2">
			<h1 class="text-4xl">ONNX Runtime Web</h1>
			<br /><br />
			<p class="text-xl">
				ONNX Runtime Web allows JavaScript developers to run and deploy machine learning models in
				browsers.
			</p>
			<br />
			<a href="https://www.youtube.com/watch?v=vYzWrT3A7wQ" class="btn btn-primary"
				>Inference in JavaScript with ONNX Runtime Web YouTube Tutorial →</a
			>
		</div>
		<div class="m-auto">
			<ImageInference2 />
		</div>
	</div>
	<div class="mx-auto bg-secondary rounded m-4 p-4">
		<h1 class="text-3xl">Examples</h1>
		<div class="grid gap-10 grid-cols-1 md:grid-cols-2">
			<div class="">
				<h1 class="text-2xl">ONNX Runtime Web Demo</h1>
				<p>
					ONNX Runtime Web demo is an interactive demo portal showing real use cases running ONNX
					Runtime Web in VueJS. It currently supports five examples for you to quickly experience
					the power of ONNX Runtime Web.
				</p>
			</div>
			<div class="join join-vertical gap-4">
				<a
					href="https://microsoft.github.io/onnxruntime-web-demo/#/mobilenet"
					class="btn btn-primary"
				>
					MobileNet, trained on ImageNet →
				</a>
				<a
					href="https://microsoft.github.io/onnxruntime-web-demo/#/squeezenet"
					class="btn btn-primary"
				>
					SqueezeNet, trained on ImageNet →
				</a>
				<a
					href="https://microsoft.github.io/onnxruntime-web-demo/#/emotion_ferplus"
					class="btn btn-primary"
				>
					Emotion FerPlus →
				</a>
				<a
					href="https://microsoft.github.io/onnxruntime-web-demo/#/emotion_ferplus"
					class="btn btn-primary"
				>
					Yolo →
				</a>
				<a href="https://microsoft.github.io/onnxruntime-web-demo/#/mnist" class="btn btn-primary">
					MNIST →
				</a>
			</div>
		</div>
		<div class="grid grid-cols-1 xl:grid-cols-3 gap-4 mt-8">
			<div class="card bg-base-200">
				<div class="card-body items-center text-center">
					<h2 class="card-title">Image Classification</h2>
					The example demonstrates how to use a GitHub repository template to build an image classification
					web app using ONNX Runtime web.
					<div class="card-actions mt-auto mb-2 justify-center">
						<a
							href="https://onnxruntime.ai/docs/tutorials/web/classify-images-nextjs-github-template.html"
							class="btn btn-primary">Classify images in a web application →</a
						>
					</div>
				</div>
			</div>
			<div class="card bg-base-200">
				<div class="card-body items-center text-center">
					<h2 class="card-title">Speech Recognition</h2>
					The example demonstrates how to run whisper tiny.en in your browser using ONNX Runtime Web
					and the browser's audio interfaces.
					<div class="card-actions mt-auto mb-2 justify-center">
						<a
							href="https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/ort-whisper"
							class="btn btn-primary">Run whisper tiny.en in your browser →</a
						>
					</div>
				</div>
			</div>
			<div class="card bg-base-200">
				<div class="card-body items-center text-center">
					<h2 class="card-title">Natural Language Processing (NLP)</h2>
					The example demonstrates how to create custom Excel functions (ORT.Sentiment() and ORT.Question())
					to implement BERT NLP models with ONNX Runtime Web to enable deep learning in spreadsheet tasks.
					<div class="card-actions mt-auto mb-2 justify-center">
						<a
							href="https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/ort-whisper"
							class="btn btn-primary hidden md:inline-flex">Custom Excel Functions for BERT NLP Tasks in JS →</a
						>
						<a
							href="https://github.com/microsoft/onnxruntime-inference-examples/tree/main/js/ort-whisper"
							class="btn btn-primary md:hidden">Custom Excel Functions for BERT NLP →</a
						>
					</div>
				</div>
			</div>
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-1 md:grid-cols-3 gap-4 lg:gap-10">
		<div class="col-span-2">
			<h1 class="text-4xl">On-Device Training</h1>
			<br /><br />
			<p class="text-xl">ONNX Runtime on-device training is an extension of ORT Inferencing.</p>
			<br />
			<a href="https://www.youtube.com/watch?v=vYzWrT3A7wQ" class="btn btn-primary"
				>Learn more about on-device training →</a
			>
		</div>
	</div>
</div>
