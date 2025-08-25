<script lang="ts">
	import { base } from '$app/paths';
	import { onMount } from 'svelte';
	import anime from 'animejs';
	import { text } from '@sveltejs/kit';
	import { Highlight } from 'svelte-highlight';
	import { bash } from 'svelte-highlight/languages';
	import FaRegClipboard from 'svelte-icons/fa/FaRegClipboard.svelte';
	import FaClipboardCheck from 'svelte-icons/fa/FaClipboardCheck.svelte'
	import OnnxLight from '../../images/ONNX-Light.svelte';
	import OnnxDark from '../../images/ONNX-Dark.svelte';
	import { fade } from 'svelte/transition';
	import { quartInOut } from 'svelte/easing';

	let words = [
		'Cross-Platform',
		'GPU',
		'Python',
		'CPU',
		'Mobile',
		'C#',
		'Edge',
		'JavaScript',
		'Java',
		'C++',
		'Browser'
	];
	let activeWord = 'Edge';
	let currentWord = 0;
	let cycleWord = () => {
		currentWord = (currentWord + 1) % words.length;
		activeWord = words[currentWord];
		if (currentWord == 0) {
			setTimeout(cycleWord, 5000);
		} else {
			setTimeout(cycleWord, 3000);
		}
	};
	setTimeout(cycleWord, 2000);
	let pythonCode = 'pip install onnxruntime';
	let gaiCode = 'pip install onnxruntime-genai';
	let copied = false;
	let copy = async (code: string) => {
		try {
			copied = true;
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
	<div class="toast toast-top top-14 z-50" role="alert">
		<div class="alert alert-info">
			<div class="icon" style="width: 16px; height: 16px;">
				<FaClipboardCheck />
			</div>
			<span>Code successfully copied!</span>
		</div>
	</div>
{/if}
<div role="main" class="hero bg-gradient-to-b from-primary">
	<div class="hero-content md:my-20">
		<div class="grid grid-cols-2 md:grid-cols-5 gap-4">
			<div class="col-span-4 self-center md:mr-20">
				<h1 class="lg:text-5xl text-4xl">
					Accelerated
					{#key activeWord}
						<span
							class="lg:text-5xl text-4xl"
							in:fade={{ delay: 0, duration: 1000, easing: quartInOut }}
						>
							{activeWord}
						</span>
					{/key}
					<br />
					Machine Learning
				</h1>
				<p class="py-3">
					Production-grade AI engine to speed up training and inferencing in your existing
					technology stack.
				</p>
				<p class="text-xl my-4">In a rush? Get started easily:</p>
				<div class="grid grid-cols-1 md:grid-cols-2 gap-4">
					<div class="grid grid-cols-6 border-solid border-2 border-secondary">
						<div class="col-span-5">
							<Highlight language={bash} code={pythonCode} />
						</div>
						<button
							aria-label="copy python code"
							on:click={() => copy(pythonCode)}
							class="col-span-1 btn rounded-none h-full *:hover:scale-125 *:hover:transition *:hover:duration-200"
							><span class="min-w-6 h-6"><FaRegClipboard /></span></button
						>
					</div>
					<div class="grid grid-cols-6 border-solid border-2 border-secondary">
						<div class="col-span-5">
							<Highlight language={bash} code={gaiCode} />
						</div>
						<button
							aria-label="copy nuget code"
							on:click={() => copy(gaiCode)}
							class="col-span-1 btn rounded-none h-full *:hover:scale-125 *:hover:transition *:hover:duration-200"
							><span class="min-w-6 h-6"><FaRegClipboard /></span></button
						>
					</div>
				</div>
				<!-- <p class="text-lg mt-2">
					<a class="underline" href="https://">More interested in training? More info here.</a>
				</p> -->
				<p class="text-lg mt-2">
					<a class="text-primary font-medium hover:text-primary-focus" href="./getting-started"
						>Interested in using other languages? See the many others we support →</a
					>
				</p>
			</div>
			<div class="hidden lg:inline mx-auto hover:rotate-180 transition duration-500">
				<OnnxLight width={300} height={300} />
			</div>
		</div>
	</div>
</div>
