<script lang="ts">
	import Hero from './components/hero.svelte';
	import Customers from './components/customers.svelte';
	import CodeBlocks from './components/code-blocks.svelte';
	import Performance from './components/performance.svelte';
	import CrossPlatform from './components/cross-platform.svelte';
	import GenerativeAi from './components/generative-ai-hero.svelte';
	import TrainingAndInference from './components/training-and-inference.svelte';
	import IoIosClose from 'svelte-icons/io/IoIosClose.svelte';
	import { onMount } from 'svelte';
	import anime from 'animejs';
	import Saos from 'saos';

	let removetoast = (e: any) => {
		e.target.parentNode.parentNode.remove();
	};

	onMount(() => {
		anime({
			targets: '.toast',
			opacity: '1',
			duration: 1000,
			delay: 10000,
			easing: 'easeInOutQuad',
			begin: function () {
				document.querySelector<HTMLElement>('.toast')!.style.display = 'block';
			}
		});
	});
</script>

<Hero />
<Customers />
<CodeBlocks />
<Saos once={true} animation={'slide-in-left 1s cubic-bezier(0.250, 0.460, 0.450, 0.940) both'}>
	<GenerativeAi />
</Saos>
<Saos once={true} animation={'slide-in-right 1s cubic-bezier(0.250, 0.460, 0.450, 0.940) both'}>
	<CrossPlatform />
</Saos>
<Saos once={true} animation={'slide-in-left 1s cubic-bezier(0.250, 0.460, 0.450, 0.940) both'}>
	<Performance />
</Saos>

<div class="divider px-10" />
<Saos once={true} animation={'slide-in-bottom 1s cubic-bezier(0.250, 0.460, 0.450, 0.940) both'}>
	<TrainingAndInference />
</Saos>

<div class="toast z-10 opacity-0 hidden">
	<div class="alert alert-info">
		<svg
			xmlns="http://www.w3.org/2000/svg"
			fill="none"
			viewBox="0 0 24 24"
			class="stroke-current shrink-0 w-6 h-6"
			><path
				stroke-linecap="round"
				stroke-linejoin="round"
				stroke-width="2"
				d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
			/></svg
		><span
			>Please help us improve ONNX Runtime <br class="md:hidden" />by participating in our
			<a class="underline" href="https://ncv.microsoft.com/UySXuzobM9">customer survey</a>.
		</span>
		<button aria-label="close" on:click={removetoast} class="w-8 h-8 pt-1"><IoIosClose /></button>
	</div>
</div>

<style>
	@-webkit-keyframes -global-slide-in-bottom {
		0% {
			-webkit-transform: translateY(1000px);
			transform: translateY(1000px);
			opacity: 0;
		}
		100% {
			-webkit-transform: translateY(0);
			transform: translateY(0);
			opacity: 1;
		}
	}
	@keyframes -global-slide-in-bottom {
		0% {
			-webkit-transform: translateY(1000px);
			transform: translateY(1000px);
			opacity: 0;
		}
		100% {
			-webkit-transform: translateY(0);
			transform: translateY(0);
			opacity: 1;
		}
	}
	@-webkit-keyframes -global-slide-in-left {
		0% {
			-webkit-transform: translateX(-1000px);
			transform: translateX(-1000px);
			opacity: 0;
		}
		100% {
			-webkit-transform: translateX(0);
			transform: translateX(0);
			opacity: 1;
		}
	}
	@keyframes -global-slide-in-left {
		0% {
			-webkit-transform: translateX(-1000px);
			transform: translateX(-1000px);
			opacity: 0;
		}
		100% {
			-webkit-transform: translateX(0);
			transform: translateX(0);
			opacity: 1;
		}
	}
	@-webkit-keyframes -global-slide-in-right {
		0% {
			-webkit-transform: translateX(1000px);
			transform: translateX(1000px);
			opacity: 0;
		}
		100% {
			-webkit-transform: translateX(0);
			transform: translateX(0);
			opacity: 1;
		}
	}
	@keyframes -global-slide-in-right {
		0% {
			-webkit-transform: translateX(1000px);
			transform: translateX(1000px);
			opacity: 0;
		}
		100% {
			-webkit-transform: translateX(0);
			transform: translateX(0);
			opacity: 1;
		}
	}
</style>
