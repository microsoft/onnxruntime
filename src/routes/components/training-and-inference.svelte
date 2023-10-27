<script>
	import { base } from '$app/paths';
	import Largemodeltraining from '../../images/undraw/image_largemodeltraining.svelte';
	import Ondevtraining from '../../images/undraw/image_ondevtraining.svelte';
	import Ortmobile from '../../images/undraw/image_ortmobile.svelte';
	import Ortweb from '../../images/undraw/image_ortweb.svelte';
</script>

<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-2">ONNX Runtime Inferencing</h1>
	<p class="text-xl pb-4">
		ONNX Runtime is the same tech that powers AI in Microsoft products like Office, Azure, and Bing, as well as in thousands of other projects across the world.
	</p>
	<a href="./inference" class="btn btn-primary">Learn more about ONNX Runtime Inferencing →</a>
	<div class="grid grid-cols-1 md:grid-cols-2 gap-10 mt-10 my-4 md:mx-10">
		<div class="bg-slate-300 p-4 rounded">
			<div class="grid xl:grid-cols-4 place-items-center">
				<div class="col-span-3 text-black">
					<h1 class="text-2xl pb-2">ONNX Runtime Web</h1>
					<p class="text-lg">
						ONNX Runtime Web allows JavaScript developers to run and deploy machine learning models
						in web browsers.
					</p>
				</div>
				<div class="hidden xl:grid">
					<Ortweb />
				</div>
			</div>
		</div>
		<div class="bg-slate-300 p-4 rounded">
			<div class="grid md:grid-cols-4 place-items-center">
				<div class="col-span-3 text-black">
					<h1 class="text-2xl pb-2">ONNX Runtime Mobile</h1>
					<p class="text-lg">
						ONNX Runtime Mobile allows you to run model inferencing on Android and iOS mobile devices.
					</p>
				</div>
				<div class="hidden xl:grid">
					<Ortmobile />
				</div>
			</div>
		</div>
	</div>
</div>
<div class="container mx-auto px-10">
	<h1 class="text-4xl pb-2">ONNX Runtime Training</h1>
	<p class="text-xl pb-4">
		ONNX Runtime enables on-device training and reduces costs for large model training.
	</p>
	<a href="./training" class="btn btn-primary">Learn more about ONNX Runtime Training →</a>
	<div class="grid grid-cols-1 md:grid-cols-2 gap-10 mt-10 my-4 md:mx-10">
		<div class="bg-slate-300 p-4 rounded">
			<div class="grid xl:grid-cols-4 place-items-center">
				<div class="col-span-3 text-black">
					<h1 class="text-2xl pb-2">Large Model Training</h1>
					<p class="text-lg">
						Accelerated training with ONNX Runtime reduces costs and improves data scientist velocity for many popular models from <a href="https://huggingface.co/" class="text-blue-500">Hugging Face</a>
						and <a href="https://ml.azure.com/" class="text-blue-500">Azure Machine Learning</a>.
					</p>
				</div>
				<div class="hidden xl:grid">
					<Largemodeltraining />
				</div>
			</div>
		</div>
		<div class="bg-slate-300 p-4 rounded">
			<div class="grid md:grid-cols-4 place-items-center">
				<div class="col-span-3 text-black">
					<h1 class="text-2xl pb-2">On-Device Training</h1>
					<p class="text-lg">
						On-device training with ONNX Runtime lets developers take an inference model and train it locally on-device to deliver a more personalized and privacy-respecting experience for customers.
					</p>
				</div>
				<div class="hidden xl:grid">
					<Ondevtraining />
				</div>
			</div>
		</div>
	</div>
</div>
