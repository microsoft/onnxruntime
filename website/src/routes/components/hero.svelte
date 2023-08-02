<script>
	import onnximage from '../../images/ONNX-Icon.svg';
	import { onMount } from 'svelte';
	import anime from 'animejs';
	import { text } from '@sveltejs/kit';
	import { Highlight } from 'svelte-highlight';
	import { bash } from 'svelte-highlight/languages';
	onMount(() => {
		anime.timeline({ loop: false }).add({
			targets: '.img',
			easing: 'spring',
			rotate: '1turn',
			duration: 1000
		});
	});
	let pythonCode = 'pip install onnxruntime';
	let nugetCode = 'nuget get onnxruntime';
	import FaRegClipboard from 'svelte-icons/fa/FaRegClipboard.svelte'
	let copy = async (code) => {
    try {
      await navigator.clipboard.writeText(code);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };
</script>

<div class="hero bg-primary">
	<div class="hero-content my-20">
		<div class="grid grid-cols-2 md:grid-cols-3 gap-4">
			<div class="col-span-2 self-center mr-20">
				<h1 class="text-5xl">Cross-platform accelerated <br />machine learning</h1>
				<p class="py-3">
					Built-in optimizations speed up training and inferencing with your existing technology
					stack.
				</p>
				<p class="text-xl my-4">In a rush? Get started easily: </p>
				<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
					<!-- TODO: Make typewriter effect, copy button -->
					<div class="grid grid-cols-6 bg-white">
						<div class="col-span-5">

							<Highlight language={bash} code={pythonCode} />
						</div>
						<button on:click={() => copy(pythonCode)} class="col-span-1 btn rounded-none h-full"><span class="icon"><FaRegClipboard/></span></button>
					</div>
					<div class="grid grid-cols-6 bg-white">
						<div class="col-span-5">

							<Highlight language={bash} code={nugetCode} />
						</div>
						<button on:click={() => copy(nugetCode)} class="col-span-1 btn rounded-none h-full"><span class="icon"><FaRegClipboard/></span></button>
					</div>
				</div>
				<p class="text-lg mt-2">Don't see your favourite platform? We support <a class="underline" href="http://">many</a>!</p>
				<!-- <div class="grid grid-cols-2">
					<p class="text-lg mt-2">Don't see your favourite platform? </p><a href="https://" class="btn btn-secondary">Get Started with another</a>
				</div> -->
			</div>
			<img src={onnximage} class="img mx-auto basis-1/3" alt="ONNX Runtime logo" />
		</div>
	</div>
</div>

<style>
	.img {
		transform: scale(0.75);
	}
	.icon {
		width: 24px;
		height: 24px;
	}
</style>
