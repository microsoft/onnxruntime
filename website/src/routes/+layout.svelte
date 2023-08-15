<script>
	import '../app.css';
	import Header from './components/header.svelte';
	import Footer from './components/footer.svelte';
	import { onMount } from 'svelte';
	import IoIosClose from 'svelte-icons/io/IoIosClose.svelte';
	import anime from 'animejs';
	import oneLight from 'svelte-highlight/styles/one-light';

	let removetoast = (e) => {
		e.target.parentNode.parentNode.remove();
	};
	onMount(() => {
		anime({
			targets: '.toast',
			opacity: '1',
			duration: 1000,
            delay: 3000,
			easing: 'easeInOutQuad',
			begin: function () { document.querySelector('.toast').style.display = 'block';},
		});
        anime({
            targets: '.toast',
			delay: 3500,
            easing: 'easeOutElastic(1, .25)',
            translateY: -10
        })
	});
</script>
<svelte:head>
	{@html oneLight}
</svelte:head>
<div class="toast z-[2] opacity-0 hidden">
	<div class="alert alert-info">
		<svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" class="stroke-current shrink-0 w-6 h-6"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
		<span
			>Please help us improve ONNX Runtime <br class="md:hidden"/>by participating in our <a
				class="underline"
				href="https://ncv.microsoft.com/UySXuzobM9">customer survey</a
			>.</span
		> <button on:click={removetoast} class="w-8 h-8 pt-1"><IoIosClose /></button>
	</div>
</div>
<Header />
<slot />
<Footer />