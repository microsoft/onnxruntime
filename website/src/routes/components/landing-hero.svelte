<script lang="ts">
export let title: string; 
export let description: string;
export let imgsrc: string;
export let imgalt: string;
</script>

<div class="container mx-auto px-10 my-10">
	<div class="grid grid-cols-1 md:grid-cols-3 gap-10">
		<div class="col-span-2">
			<h1 class="text-4xl">{title}</h1>
			<br /><br />
			<p class="text-xl">
				{description}
			</p>
			<br>
		</div>
		<img class="m-auto" src={imgsrc} alt={imgalt} />
	</div>
</div>