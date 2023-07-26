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
</script>

<div class="grid grid-cols-5 gap-2 container mx-auto">
	<div class="col-span-1 bg-success px-2 rounded-sm">
		<div class="text-xl col-span-11">Platform</div>
	</div>
	<div class="col-span-4 grid gap-2">
		<div class="grid gap-2 grid-cols-6">
			{#each platforms as platform}
				<button class="btn btn-sm btn-secondary">
					{platform}
				</button>
			{/each}
		</div>
	</div>
	<div class="col-span-1 bg-success px-2 rounded-sm">
		<div class="text-xl">API</div>
	</div>
	<div class="col-span-4 grid gap-2">
		<div class="grid gap-2 grid-cols-8">
			{#each apis as api}
				<button class="col-span-1 btn btn-sm btn-secondary">
					{api}
				</button>
			{/each}
		</div>
	</div>
	<div class="col-span-1 bg-success px-2 rounded-sm">
		<div class="text-xl">Architecture</div>
	</div>
	<div class="col-span-4 grid gap-2">
		<div class="grid gap-2 grid-cols-5">
			{#each architectures as architecture}
				<button class="col-span-1 btn btn-sm btn-secondary">
					{architecture}
				</button>
			{/each}
		</div>
	</div>
	<div class="col-span-1 bg-success px-2 rounded-sm">
		<div class="text-xl">Hardware Acceleration</div>
	</div>
	<div class="col-span-4 grid gap-2">
		<div class="grid gap-2 grid-cols-5">
			{#each hardwareAcceleration as hardware}
				<button class="col-span-1 btn btn-sm btn-secondary">
					{hardware}
				</button>
			{/each}
		</div>
	</div>
	<div class="col-span-1 bg-success px-2 rounded-sm">
		<div class="text-xl pt-1">Installation Instructions</div>
	</div>
	<div class="col-span-4 grid gap-2 bg-info p-2 rounded-sm">Install with these instructions</div>
</div>

<!-- 
<div class="grid grid-cols-4 gap-2 container mx-auto">
    <div class="col-span-1 text-right">
        <div class="text-xl">Platform</div>
        <div class="text-xl">API</div>
        <div class="text-xl">Architecture</div>
        <div class="text-xl">Hardware Acceleration</div>
        <div class="text-xl">Installation Instructions</div>
    </div>
    <div class="col-span-3 grid gap-2">
        <div class="grid gap-1 grid-cols-6">
            {#each platforms as platform}
                <button class="col-span-1 btn btn-sm btn-secondary">
                    {platform}
                </button>
            {/each}
        </div>
        <div class="grid gap-2 grid-cols-8">
            {#each apis as api}
                <button class="col-span-1 btn btn-sm btn-secondary">
                    {api}
                </button>
            {/each}
        </div>
        <div class="grid gap-2 grid-cols-5">
            {#each architectures as architecture}
                <button class="col-span-1 btn btn-sm btn-secondary">
                    {architecture}
                </button>
            {/each}
        </div>
        <div class="grid gap-2 grid-cols-4">
            {#each hardwareAcceleration as hardware}
                <button class="col-span-1 btn btn-sm btn-secondary">
                    {hardware}
                </button>
            {/each}
        </div>
    </div>
</div> -->
