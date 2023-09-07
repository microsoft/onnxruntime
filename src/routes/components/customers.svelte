<script lang="ts">
	import { base } from '$app/paths';
	import anime from 'animejs';
	import pytorchlogo from '../../images/logos/PyTorch_logo_black.svg'
	import windowslogo from '../../images/logos/Windows_logo_and_wordmark.svg'
	import huggingfacelogo from '../../images/logos/hf-logo-with-title.svg'
	import { onMount } from 'svelte';
	onMount(() => {
		anime({
			targets: '.partner',
			translateY: -20,
			direction: 'alternate',
			loop: false,
			delay: function (el, i, l) {
				return 500 + i * 50;
			},
			endDelay: function (el, i, l) {
				return (l - i) * 50;
			}
		});
	});

	let handleEnter = (e:any) => {
		anime({
			targets: e.target,
			scale: 1.1,
			duration: 1500,
		})
	}
	let handleLeave = (e:any) => {
		anime({
			targets: e.target,
			scale: 1,
			duration: 1500,
		})
	}
</script>

<div class="container mx-auto md:px-10 px-4 lg:my-20">
	<h1 class="text-3xl my-5 text-center">Trusted By </h1>
	<div class="divider" />
	<div
		class="grid lg:grid-cols-8 md:grid-cols-4 grid-cols-2 gap-2 mx-auto justify-items-center pt-10"
	>
		<button on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="partner cursor-default">
			<img src="https://onnxruntime.ai/images/logos/amd-logo.png" alt="AMD" />
		</button>
		<button on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="partner cursor-default">
			<img src="https://onnxruntime.ai/images/logos/huggingface-logo.png" alt="HuggingFace" />
		</button>
		<button on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="partner cursor-default">
			<img src="https://onnxruntime.ai/images/logos/intel-logo.png" alt="Intel" />
		</button>
		<button on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="partner cursor-default">
			<img src="https://onnxruntime.ai/images/logos/nvidia.png" alt="Nvidia" />
		</button>
		<button on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="partner cursor-default">
			<img src="https://onnxruntime.ai/images/logos/adobe-logo.png" alt="Adobe" />
		</button>
		<button on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="partner cursor-default">
			<img src="https://onnxruntime.ai/images/logos/visual-studio-logo.png" alt="Microsoft" />
		</button>
		<button on:mouseenter={handleEnter} on:mouseleave={handleLeave} class="partner cursor-default">
			<img src="https://onnxruntime.ai/images/logos/oracle-logo.png" alt="Oracle" />
		</button>
		<a on:mouseenter={handleEnter} on:mouseleave={handleLeave} href="./testimonials" class="partner">
			<p class="underline text-center pt-20">...and many more</p>
		</a>
	</div>
	<h1 class="text-2xl pt-10 pb-4">Learn more about how to use ONNX Runtime with:</h1>
	<div class="grid md:grid-cols-3 grid-cols-1 gap-4 mx-auto pb-10">
		<a href="./pytorch" class="btn btn-secondary rounded-sm h-full"><img class="px-10 py-5" src={pytorchlogo} alt="PyTorch Logo"></a>
		<a href="./windows" class="btn btn-secondary rounded-sm h-full"><img class="px-10 py-5" src={windowslogo} alt="Windows Logo"></a>
		<a href="./huggingface" class="btn btn-secondary rounded-sm h-full"><img class="px-10 py-5" src={huggingfacelogo} alt="HuggingFace Logo"></a>
	</div>
</div>
