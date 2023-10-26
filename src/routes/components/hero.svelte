<script lang="ts">
	import { base } from '$app/paths';
	import { onMount } from 'svelte';
	import anime from 'animejs';
	import { text } from '@sveltejs/kit';
	import { Highlight } from 'svelte-highlight';
	import { bash } from 'svelte-highlight/languages';
	import FaRegClipboard from 'svelte-icons/fa/FaRegClipboard.svelte';
	import OnnxLight from '../../images/ONNX-Light.svelte';
	import OnnxDark from '../../images/ONNX-Dark.svelte';
	import { fade, fly, blur } from 'svelte/transition';
	import { quintOut } from 'svelte/easing';

	let words = ['Cross-Platform', 'Python', 'C#', 'JavaScript', 'Java', 'C++'];
	let activeWord = 'Cross-Platform';
	let currentWord = 0;
	let cycleWord = () => {
		currentWord = (currentWord + 1) % 6;
		activeWord = words[currentWord];
		if (currentWord == 0) {
			setTimeout(cycleWord, 5000);
		} else {
			setTimeout(cycleWord, 3000);
		}
	};
	setTimeout(cycleWord, 2000);
	let pythonCode = 'pip install onnxruntime';
	let nugetCode = 'nuget install Microsoft.ML.OnnxRuntime';
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
	<div class="toast toast-top top-14 z-50">
		<div class="alert alert-info">
			<svg
				xmlns="http://www.w3.org/2000/svg"
				width="16"
				height="16"
				fill="currentColor"
				class="bi bi-clipboard2-check"
				viewBox="0 0 16 16"
			>
				<path
					d="M9.5 0a.5.5 0 0 1 .5.5.5.5 0 0 0 .5.5.5.5 0 0 1 .5.5V2a.5.5 0 0 1-.5.5h-5A.5.5 0 0 1 5 2v-.5a.5.5 0 0 1 .5-.5.5.5 0 0 0 .5-.5.5.5 0 0 1 .5-.5h3Z"
				/>
				<path
					d="M3 2.5a.5.5 0 0 1 .5-.5H4a.5.5 0 0 0 0-1h-.5A1.5 1.5 0 0 0 2 2.5v12A1.5 1.5 0 0 0 3.5 16h9a1.5 1.5 0 0 0 1.5-1.5v-12A1.5 1.5 0 0 0 12.5 1H12a.5.5 0 0 0 0 1h.5a.5.5 0 0 1 .5.5v12a.5.5 0 0 1-.5.5h-9a.5.5 0 0 1-.5-.5v-12Z"
				/>
				<path
					d="M10.854 7.854a.5.5 0 0 0-.708-.708L7.5 9.793 6.354 8.646a.5.5 0 1 0-.708.708l1.5 1.5a.5.5 0 0 0 .708 0l3-3Z"
				/>
			</svg>
			<span>Code successfully copied!</span>
		</div>
	</div>
{/if}
<div class="hero bg-gradient-to-b from-primary">
	<div class="hero-content md:my-20">
		<div class="grid grid-cols-2 md:grid-cols-5 gap-4">
			<div class="col-span-4 self-center md:mr-20">
				{#key activeWord}
					<h1
						class="text-5xl"
						in:fly={{ delay: 0, duration: 300, x: 200, y: 0, opacity: 1, easing: quintOut }}
					>
						{activeWord}
					</h1>
				{/key}
				<h1 class="text-5xl">accelerated machine learning</h1>
				<p class="py-3">
					Built-in optimizations speed up training and inferencing with your existing technology
					stack.
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
							class="col-span-1 btn rounded-none h-full"
							><span class="w-6 h-6"><FaRegClipboard /></span></button
						>
					</div>
					<div class="grid grid-cols-6 border-solid border-2 border-secondary">
						<div class="col-span-5">
							<Highlight language={bash} code={nugetCode} />
						</div>
						<button
							aria-label="copy nuget code"
							on:click={() => copy(nugetCode)}
							class="col-span-1 btn rounded-none h-full"
							><span class="w-6 h-6"><FaRegClipboard /></span></button
						>
					</div>
				</div>
				<!-- <p class="text-lg mt-2">
					<a class="underline" href="http://">More interested in training? More info here.</a>
				</p> -->
				<p class="text-lg mt-2">
					<a class="text-blue-500 font-medium" href="./getting-started"
						>Don't see your favorite platform? See the many others we support →</a
					>
				</p>
			</div>
			<div class="hidden lg:inline mx-auto">
				<OnnxLight width={300} height={300} />
			</div>
		</div>
	</div>
</div>
