<script lang="ts">
	import { base } from '$app/paths';
	import anime from 'animejs';
	import pytorchlogo from '../../images/logos/PyTorch_logo_black.svg';
	import windowslogo from '../../images/logos/Windows_logo_and_wordmark.svg';
	import huggingfacelogoTitle from '../../images/logos/hf-logo-with-title.svg';
	import amdLogo from '../../images/logos/amd-logo.png';
	import huggingfaceLogo from '../../images/logos/huggingface-logo.png';
	import intelLogo from '../../images/logos/intel-logo.png';
	import nvidiaLogo from '../../images/logos/nvidia.png';
	import adobeLogo from '../../images/logos/adobe-logo.png';
	import oracleLogo from '../../images/logos/oracle-logo.png';
	import unrealEngineLogo from '../../images/logos/ue-logo.png';

	import { onMount } from 'svelte';
	// Prevents animations from being messed up when they first start
	let interact = false;
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
		setTimeout(() => {
			interact = true;
		}, 1000);
	});

	let handleEnter = (e: any) => {
		if (interact) {
			anime({
				targets: e.target,
				scale: 1.1,
				duration: 1500
			});
		}
	};
	let handleLeave = (e: any) => {
		if (interact) {
			anime({
				targets: e.target,
				scale: 1,
				duration: 1500
			});
		}
	};
</script>

<div class="container mx-auto md:px-10 px-4 lg:my-5">
	<h1 class="text-3xl my-5 text-center">Trusted By</h1>
	<div class="divider" />
	<div
		class="grid lg:grid-cols-8 md:grid-cols-4 grid-cols-2 gap-4 mx-auto justify-items-center pt-10"
	>
		<a
			href="./testimonials"
			on:mouseenter={handleEnter}
			on:mouseleave={handleLeave}
			class="partner bg-slate-300 border-solid border-2 border-secondary rounded w-full w-full"
		>
			<img class="h-20 mx-auto"src={amdLogo} alt="AMD" />
		</a>
		<a
			href="./testimonials"
			on:mouseenter={handleEnter}
			on:mouseleave={handleLeave}
			class="partner bg-slate-300 border-solid border-2 border-secondary rounded w-full w-full"
		>
			<img class="h-20 mx-auto"src={huggingfaceLogo} alt="HuggingFace" />
		</a>
		<a
			href="./testimonials"
			on:mouseenter={handleEnter}
			on:mouseleave={handleLeave}
			class="partner bg-slate-300 border-solid border-2 border-secondary rounded w-full"
		>
			<img class="h-20 mx-auto"src={intelLogo} alt="Intel" />
		</a>
		<a
			href="./testimonials"
			on:mouseenter={handleEnter}
			on:mouseleave={handleLeave}
			class="partner bg-slate-300 border-solid border-2 border-secondary rounded w-full"
		>
			<img class="h-20 mx-auto"src={nvidiaLogo} alt="Nvidia" />
		</a>
		<a
			href="./testimonials"
			on:mouseenter={handleEnter}
			on:mouseleave={handleLeave}
			class="partner bg-slate-300 border-solid border-2 border-secondary rounded w-full"
		>
			<img class="h-20 mx-auto"src={adobeLogo} alt="Adobe" />
		</a>
		<a
			href="./testimonials"
			on:mouseenter={handleEnter}
			on:mouseleave={handleLeave}
			class="partner bg-slate-300 border-solid border-2 border-secondary rounded w-full"
		>
			<img class="h-20 mx-auto"src={unrealEngineLogo} alt="Unreal Engine" />
		</a>
		<a
			href="./testimonials"
			on:mouseenter={handleEnter}
			on:mouseleave={handleLeave}
			class="partner bg-slate-300 border-solid border-2 border-secondary rounded w-full"
		>
			<img class="h-20 mx-auto"src={oracleLogo} alt="Oracle" />
		</a>
		<a
			on:mouseenter={handleEnter}
			on:mouseleave={handleLeave}
			href="./testimonials"
			class="w-full partner bg-slate-300 border-solid border-2 border-secondary rounded flex items-center justify-center"
		>
			<p class="underline text-black">...and many more</p>
		</a>
	</div>
	<h1 class="text-2xl pt-10 pb-4">Learn more about how to use ONNX Runtime with</h1>
	<div class="grid md:grid-cols-3 grid-cols-1 gap-4 mx-auto pb-10">
		<a
			href="./pytorch"
			class="btn btn-primary bg-slate-300 border-solid border-2 border-secondary rounded h-full"
			><img class="px-10 py-5" src={pytorchlogo} alt="PyTorch Logo" /></a
		>
		<a
			href="./windows"
			class="btn btn-primary bg-slate-300 border-solid border-2 border-secondary rounded h-full"
			><img class="px-10 py-5" src={windowslogo} alt="Windows Logo" /></a
		>
		<a
			href="./huggingface"
			class="btn btn-primary bg-slate-300 border-solid border-2 border-secondary rounded h-full"
			><img class="px-10 py-5" src={huggingfacelogoTitle} alt="HuggingFace Logo" /></a
		>
	</div>
</div>
