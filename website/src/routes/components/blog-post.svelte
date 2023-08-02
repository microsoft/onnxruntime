<script>
	let badges = [
		['ONNX', 'primary'],
		['ML', 'primary']
	];
</script>
<div class="m-10">
<a href="http://blog.com">
	<div class="card card-side bg-base-100 border border-primary">
		<div class="card-body">
			<h2 class="card-title">Title</h2>
			<p>Description</p>
			<div class="card-actions">
				{#each badges as badge}
					<div class="badge badge-{badge[1]}">{badge[0]}</div>
				{/each}
			</div>
		</div>
		<figure>
			<img src="../../images/stock/photo-1635805737707-575885ab0820.jpg" alt="Movie" />
		</figure>
	</div>
</a>
</div>