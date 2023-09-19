<script>
	let platforms = ['Windows', 'Linux', 'Mac', 'Android', 'iOS', 'Web Browser'];
	let apis = ['Python', 'C++', 'C#', 'C', 'Java', 'JS', 'Obj-C', 'WinRT'];
	let architectures = ['X64', 'X86', 'ARM64', 'ARM32', 'IBM Power'];
	let hardwareAcceleration = [
		'Default CPU',
		'CoreML',
		'CUDA',
		'DirectML',
		'MIGraphX',
		'NNAPI',
		'oneDNN',
		'OpenVINO',
		'ROCm',
		'QNN',
		'Tensor RT',
		'ACL (Preview)',
		'ArmNN (Preview)',
		'Azure (Preview)',
		'CANN (Preview)',
		'Rockchip NPU (Preview)',
		'TVM (Preview)',
		'Vitis AI (Preview)',
		'XNNPACK (Preview)'
	];
	let radioValue = '';
	function onChange(event) {
		radioValue = event.currentTarget.value;
	}
</script>

<div class="grid grid-cols-5 gap-2 container mx-auto">
	<div class="col-span-1 bg-success px-2 rounded-sm">
		<div class="text-xl col-span-11 my-auto">Platform</div>
	</div>
	<div class="col-span-4 grid gap-2">
		<div class="join gap-2">
			{#each platforms as platform}
				<input
					class="join-item btn flex-auto"
					type="radio"
					name="platform"
					value={platform}
					aria-label={platform}
					on:change={onChange}
				/>
			{/each}
		</div>
	</div>
	<div class="col-span-1 bg-success px-2 rounded-sm">
		<div class="text-xl">API</div>
	</div>
	<div class="col-span-4 grid gap-2">
		<div class="join gap-2">
			{#each apis as api}
				<input class="join-item btn flex-auto" type="radio" name="api" aria-label={api} />
			{/each}
		</div>
	</div>
	<div class="col-span-1 bg-success px-2 rounded-sm">
		<div class="text-xl">Architecture</div>
	</div>
	<div class="col-span-4 grid gap-2">
		<div class="join gap-2">
			{#each architectures as architecture}
				<input
					class="join-item btn flex-auto"
					type="radio"
					name="architecture"
					aria-label={architecture}
				/>
			{/each}
		</div>
	</div>
	<div class="col-span-1 bg-success px-2 rounded-sm">
		<div class="text-xl">Hardware Acceleration</div>
	</div>
	<div class="col-span-4 grid gap-2">
		<div class="grid gap-2 grid-cols-5">
			{#each hardwareAcceleration as hardware}
				<input class="join-item btn flex-auto" type="radio" name="hardware" aria-label={hardware} />
			{/each}
		</div>
	</div>
	<div class="col-span-1 bg-success px-2 rounded-sm">
		<div class="text-xl pt-1">Installation Instructions</div>
	</div>
	<div class="col-span-4 grid gap-2 bg-base-300 p-2 rounded-sm">
		{radioValue == '' ? 'Please select a combination of resources' : radioValue}
	</div>
</div>
