<script>
	import { base } from '$app/paths';
	import Largemodeltraining from '../../images/undraw/image_largemodeltraining.svelte';
	import Ondevtraining from '../../images/undraw/image_ondevtraining.svelte';
	import Ortmobile from '../../images/undraw/image_ortmobile.svelte';
	import Ortweb from '../../images/undraw/image_ortweb.svelte';
</script>

<div class="container mx-auto px-10">
	<h1 class="text-4xl pb-2">ONNX Runtime Training</h1>
	<p class="text-xl pb-4">
		ONNX Runtime can be used to accelerate both large model training and on-device training.
	</p>
	<a href="./training" class="btn btn-primary">Learn more about ONNX Runtime Training →</a>
	<div class="grid grid-cols-1 md:grid-cols-2 gap-10 mt-10 my-4 md:mx-10">
		<div class="bg-slate-500 p-4 rounded">
			<div class="grid lg:grid-cols-4 place-items-center">
				<div class="col-span-3">
					<h1 class="text-2xl pb-2 text-slate-100">Large Model Training</h1>
					<p class="text-lg">
						ORTModule accelerates training of large transformer PyTorch models. The training time
						and cost are reduced with a single line code change. It is built on top of highly
						successful and proven technologies of ONNX Runtime and ONNX format.
					</p>
				</div>
				<div class="hidden lg:grid">
					<Largemodeltraining />
				</div>
			</div>
		</div>
		<div class="bg-slate-500 p-4 rounded">
			<div class="grid md:grid-cols-4 place-items-center">
				<div class="col-span-3">
					<h1 class="text-2xl pb-2 text-slate-100">On-Device Training</h1>
					<p class="text-lg">
						On-Device Training refers to the process of training a model on an edge device, such as
						mobile phones, embedded devices, gaming consoles, web browsers, etc. This is in contrast
						to training a model on a server or a cloud.
					</p>
				</div>
				<div class="hidden lg:grid">
					<Ondevtraining />
				</div>
			</div>
		</div>
	</div>
</div>
<div class="container mx-auto px-10 my-10">
	<h1 class="text-4xl pb-2">ONNX Runtime Inferencing</h1>
	<p class="text-xl pb-4">
		ONNX Runtime Inference powers machine learning models in key Microsoft products and services
		across Office, Azure, Bing, as well as thousands of community projects.
	</p>
	<a href="./inference" class="btn btn-primary">Learn more about ONNX Runtime Inferencing →</a>
	<div class="grid grid-cols-1 md:grid-cols-2 gap-10 mt-10 my-4 md:mx-10">
		<div class="bg-slate-500 p-4 rounded">
			<div class="grid lg:grid-cols-4 place-items-center">
				<div class="col-span-3">
					<h1 class="text-2xl pb-2 text-slate-100">ONNX Runtime Web</h1>
					<p class="text-lg">
						ONNX Runtime Web allows JavaScript developers to run and deploy machine learning models
						in browsers.
					</p>
				</div>
				<div class="hidden lg:grid">
					<Ortweb />
				</div>
			</div>
		</div>
		<div class="bg-slate-500 p-4 rounded">
			<div class="grid md:grid-cols-4 place-items-center">
				<div class="col-span-3">
					<h1 class="text-2xl pb-2 text-slate-100">ONNX Runtime Mobile</h1>
					<p class="text-lg">
						ONNX Runtime Mobile allows you to run model inferencing on mobile devices (iOS and
						Android).
					</p>
				</div>
				<div class="hidden lg:grid">
					<Ortmobile />
				</div>
			</div>
		</div>
	</div>
</div>
