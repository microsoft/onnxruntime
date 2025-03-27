<script>
	import { onMount } from 'svelte';
	import jq from 'jquery';
	import InferenceTable from './inference-table.svelte';
	import GenAITable from './genai-table.svelte';

let tabs = ['Optimize Inferencing', 'Gen AI API'];
let activeTab = 0;
</script>

<section class="pb-4" id="getStartedTable">
	<!-- <div class="pb-3">
		<h2 class="text-3xl font-bold">Installation Instructions</h2>
	</div> -->
	<noscript>
		<div class="javascript-is-disabled flex">
			<div class="w-full">
				<div class="p-4 bg-red-200 rounded">
					<h2>Please enable JavaScript to use the interactive installation guide.</h2>
					<p class="mb-0">
						Need help enabling JavaScript? Follow the instructions
						<a
							href="https://www.whatismybrowser.com/guides/how-to-enable-javascript/auto"
							target="_blank">here</a
						>.
					</p>
				</div>
			</div>
		</div>
	</noscript>
	<p class="pb-4">
		Select the configuration you want to use and run the corresponding installation script.
	</p>
	<div>
		<div class="tabs tabs-bordered" role="tablist">
			{#each tabs as tab, index}
				<li
					class="nav-item tab tab-lg"
					class:tab-active={activeTab == index}
					on:click={() => (activeTab = index)}
					on:keypress={() => (activeTab = index)}
					id={index == 0 ? 'OI_tab' : 'OT_tab'}
					data-toggle="pill"
					tabindex={0}
					role="tab"
				>
					{tab}
				</li>
			{/each}
		</div>
		
		<div
			class:hidden={activeTab != 0}
		>
	<InferenceTable/>
	</div>
	<div
	class:hidden={activeTab != 1}
>
<GenAITable/>
</div>
	</div>
</section>
