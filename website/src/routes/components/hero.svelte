<script>
	import onnximage from '../../images/ONNX-Icon.svg';
	import { onMount } from 'svelte';
	import anime from 'animejs';
	import { text } from '@sveltejs/kit';
	import { Highlight } from 'svelte-highlight';
	import { bash } from 'svelte-highlight/languages';
	import FaRegClipboard from 'svelte-icons/fa/FaRegClipboard.svelte'

	onMount(() => {
		anime.timeline({ loop: false }).add({
			targets: '.img',
			easing: 'spring',
			rotate: '1turn',
			duration: 1000
		})
	});

	
	let pythonCode = 'pip install onnxruntime';
	let nugetCode = 'nuget get onnxruntime';
	let copied = false;
	let copy = async (code) => {
    try {
		copied = true
		setTimeout(() => {
				copied = false;
			}, 3000);
      await navigator.clipboard.writeText(code);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };
</script>
{#if copied}
	<div class="toast toast-top">
		<div class="alert alert-info">
			<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-clipboard2-check" viewBox="0 0 16 16">
				<path d="M9.5 0a.5.5 0 0 1 .5.5.5.5 0 0 0 .5.5.5.5 0 0 1 .5.5V2a.5.5 0 0 1-.5.5h-5A.5.5 0 0 1 5 2v-.5a.5.5 0 0 1 .5-.5.5.5 0 0 0 .5-.5.5.5 0 0 1 .5-.5h3Z"/>
				<path d="M3 2.5a.5.5 0 0 1 .5-.5H4a.5.5 0 0 0 0-1h-.5A1.5 1.5 0 0 0 2 2.5v12A1.5 1.5 0 0 0 3.5 16h9a1.5 1.5 0 0 0 1.5-1.5v-12A1.5 1.5 0 0 0 12.5 1H12a.5.5 0 0 0 0 1h.5a.5.5 0 0 1 .5.5v12a.5.5 0 0 1-.5.5h-9a.5.5 0 0 1-.5-.5v-12Z"/>
				<path d="M10.854 7.854a.5.5 0 0 0-.708-.708L7.5 9.793 6.354 8.646a.5.5 0 1 0-.708.708l1.5 1.5a.5.5 0 0 0 .708 0l3-3Z"/>
			  </svg>
			<span>Code successfully copied!</span>
		</div>
	</div>
{/if}
<!-- TODO: Interesting background -->
<div class="hero bg-primary">
	<div class="hero-content md:my-20">
		<div class="grid grid-cols-2 md:grid-cols-3 gap-4">
			<div class="col-span-2 self-center md:mr-20">
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
						<button on:click={() => copy(pythonCode)} class="col-span-1 btn rounded-none h-full"><span class="w-6 h-6"><FaRegClipboard/></span></button>
					</div>
					<div class="grid grid-cols-6 bg-white">
						<div class="col-span-5">

							<Highlight language={bash} code={nugetCode} />
						</div>
						<button on:click={() => copy(nugetCode)} class="col-span-1 btn rounded-none h-full"><span class="w-6 h-6"><FaRegClipboard/></span></button>
					</div>
				</div>
				<p class="text-lg mt-2">Don't see your favourite platform? We support <a class="underline" href="http://">many</a>!</p>
				<!-- <div class="grid grid-cols-2">
					<p class="text-lg mt-2">Don't see your favourite platform? </p><a href="https://" class="btn btn-secondary">Get Started with another</a>
				</div> -->
			</div>
			<img src={onnximage} class="img mx-auto basis-1/3 hidden md:block" alt="ONNX Runtime logo" />
		</div>
	</div>
</div>