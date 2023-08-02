<script>
	let badges = [
		['ONNX', 'primary'],
		['ML', 'primary']
	];
</script>

<div class="card bg-base-100 border border-primary m-10" data-svelte-h="svelte-1oufy6g">
	<figure><img src="../../images/stock/photo-1606107557195-0e29a4b5b4aa.jpg" alt="Shoes" /></figure>
	<div class="card-body">
		<h2 class="card-title">Shoes!</h2>
		<p>If a dog chews shoes whose shoes does he choose?</p>
		<div class="card-actions">
            <div>
				{#each badges as badge}
					<div class="badge badge-{badge[1]} mr-1">{badge[0]}</div>
				{/each}
			</div>
        </div>
	</div>
</div>
