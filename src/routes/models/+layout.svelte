<script lang="ts">
	import Header from '../components/header.svelte';
	import Footer from '../components/footer.svelte';
	import { page } from '$app/stores';
	export let data;
</script>

<div class="selection:bg-info">
	<Header />
	<div id="main-content">
		<slot />
	</div>
	<Footer pathvar="../." />
</div>
