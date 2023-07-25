<script>
	import '../app.css';
	import Header from './components/header.svelte';
	import Footer from './components/footer.svelte';
	import { onMount } from 'svelte';
	import IoIosClose from 'svelte-icons/io/IoIosClose.svelte';
	import anime from 'animejs';

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
			begin: function () {
				// @ts-ignore
				document.querySelector('.toast').style.display = 'block';
			},
		});
        anime({
            targets: '.toast',
			delay: 3500,
            easing: 'easeOutElastic(1, .25)',
            translateY: -10
        })
	});
</script>

<div class="toast">
	<div class="alert alert-primary">
		<span
			>Please help us improve ONNX Runtime <br class="md:hidden"/>by participating in our <a
				class="underline"
				href="https://ncv.microsoft.com/UySXuzobM9">customer survey</a
			>.</span
		> <button on:click={removetoast} class="icon pt-1"><IoIosClose /></button>
	</div>
</div>
<Header />
<slot />
<Footer />

<style>
	.icon {
		width: 32px;
		height: 32px;
	}
	.toast {
		display: none;
		opacity: 0;
	}
</style>
