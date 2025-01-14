<script>
    import FaRegClipboard from 'svelte-icons/fa/FaRegClipboard.svelte';
	import FaClipboardCheck from 'svelte-icons/fa/FaClipboardCheck.svelte'
	let platforms = ['Windows', 'Linux', 'MacOS'];
	let languages = ['Python', 'C#'];
	let hardwareAccelerations = ['CPU', 'DirectML', 'CUDA'];
	let builds = ['Stable', 'Preview (Nightly)'];

	/**
	 * @type {string | null}
	 */
	let selectedPlatform = null;
	/**
	 * @type {string | null}
	 */
	let selectedLanguage = null;
	/**
	 * @type {string | null}
	 */
	let selectedHardwareAcceleration = null;
	/**
	 * @type {string | null}
	 */
	let selectedBuild = null;

	let installationInstructions = `<p>Please select a combination of resources.</p>`;

	const selectOption = (/** @type {string} */ type, /** @type {string | null} */ value) => {
		if (type === 'platform') selectedPlatform = selectedPlatform === value ? null : value;
		if (type === 'language') selectedLanguage = selectedLanguage === value ? null : value;
		if (type === 'hardwareAcceleration')
			selectedHardwareAcceleration = selectedHardwareAcceleration === value ? null : value;
		if (type === 'build') selectedBuild = selectedBuild === value ? null : value;

		updateInstallationInstructions();
	};

    // Function to dynamically copy text
    let copied = false;
    const copyCodeToClipboard = () => {
        const codeElement = document.querySelector('#installation-code');
        if (codeElement && codeElement.textContent) {
            const textToCopy = codeElement.textContent;
            // textToCopy && navigator.clipboard.writeText(textToCopy).then(() => {
            //     alert('Copied to clipboard!');
            // }).catch(err => {
            //     console.error('Failed to copy text: ', err);
            // });
            try {
			copied = true;
			setTimeout(() => {
				copied = false;
			}, 3000);
			navigator.clipboard.writeText(textToCopy);
		} catch (err) {
			console.error('Failed to copy:', err);
		}
        }
    };

	const updateInstallationInstructions = () => {
        if (!selectedPlatform || !selectedLanguage || !selectedHardwareAcceleration) {
            installationInstructions = `<p>Please select a combination of resources.</p>`;
            return;
        }

        switch (selectedLanguage) {
            case 'Python':
                switch (selectedHardwareAcceleration) {
                    case 'CPU':
                        installationInstructions = `
                            <pre><code id="installation-code">pip install onnxruntime-genai</code></pre>
                        `;
                        break;

                    case 'DirectML':
                        installationInstructions = `
                            <pre><code id="installation-code">pip install onnxruntime-genai-directml</code></pre>
                        `;
                        break;

                    case 'CUDA':
                        installationInstructions = `
                            <ul class="list-decimal pl-4">
                                <li>Ensure that the CUDA toolkit is installed.</li>
                                <li>Download the CUDA toolkit from the <a href="https://developer.nvidia.com/cuda-toolkit-archive" target="_blank">CUDA Toolkit Archive</a>.</li>
                                <li>Set the <code>CUDA_PATH</code> environment variable to the CUDA installation path.</li>
                            </ul>
                            <br/>
                            <p>Install commands:</p>
                            <ul>
                                <li>
                                    For CUDA 12: 
                                    <pre><code id="installation-code">pip install onnxruntime-genai-cuda</code></pre>
                                </li>
                                <br/>
                                 <li>
                                    For CUDA 11: 
                                    <pre class="text-wrap"><code id="installation-code">pip install onnxruntime-genai-cuda --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-11/pypi/simple/</code></pre>
                                </li>
                            </ul>
                        `;
                        break;

                    default:
                        installationInstructions = `<p>Unsupported hardware acceleration option for Python.</p>`;
                }
                break;

            case 'C#':
                switch (selectedHardwareAcceleration) {
                    case 'CPU':
                        installationInstructions = `
                            <pre><code id="installation-code">dotnet add package Microsoft.ML.OnnxRuntimeGenAI</code></pre>
                        `;
                        break;

                    case 'DirectML':
                        installationInstructions = `
                            <pre><code id="installation-code">dotnet add package Microsoft.ML.OnnxRuntimeGenAI.DirectML</code></pre>
                        `;
                        break;

                    case 'CUDA':
                        installationInstructions = `
                            <p>Note: Only CUDA 11 is supported for versions 0.3.0 and earlier, and only CUDA 12 is supported for versions 0.4.0 and later.</p>
                            <br/>
                            <pre><code id="installation-code">dotnet add package Microsoft.ML.OnnxRuntimeGenAI.Cuda</code></pre>
                        `;
                        break;

                    default:
                        installationInstructions = `<p>Unsupported hardware acceleration option for C#.</p>`;
                }
                break;

            default:
                installationInstructions = `<p>Unsupported language selection.</p>`;
        }
    };

	// If required, can use this method to enable disabling.
	// const isDisabled = (/** @type {string} */ type, /** @type {string} */ value) => {
	// 	if (type === 'language' && selectedPlatform === 'MacOS' && value === 'C#') {
	// 		return true;
	// 	}
	// 	return false;
	// };
</script>

<div
	id="panel2"
	class="grid grid-cols-5 gap-4 container mx-auto p-5"
	role="tabpanel"
	aria-labelledby="tab2"
>
{#if copied}
	<div class="toast toast-top top-14 z-50" role="alert">
		<div class="alert alert-info">
			<div class="icon" style="width: 16px; height: 16px;">
				<FaClipboardCheck />
			</div>
			<span>Code successfully copied!</span>
		</div>
	</div>
{/if}
	<div class="col-span-1 bg-success r-heading rounded p-2 text-xl">
		<h3>Platform</h3>
	</div>
	<div class="col-span-4">
		<div class="grid grid-cols-3 gap-4">
			{#each platforms as platform}
				<button
					class="btn rounded {selectedPlatform === platform ? 'btn-active btn-primary' : ''}"
					on:click={() => selectOption('platform', platform)}
				>
					{platform}
				</button>
			{/each}
		</div>
	</div>

	<div class="col-span-1 bg-success r-heading rounded p-2 text-xl">
		<h3>API</h3>
	</div>
	<div class="col-span-4">
		<div class="grid grid-cols-2 gap-4">
			{#each languages as language}
				<button
					class="btn rounded {selectedLanguage === language ? 'btn-active btn-primary' : ''}"
					on:click={() => selectOption('language', language)}
				>
					{language}
				</button>
			{/each}
		</div>
	</div>

	<div class="col-span-1 bg-success r-heading rounded p-2 text-xl">
		<h3>Hardware Acceleration</h3>
	</div>
	<div class="col-span-4">
		<div class="grid grid-cols-3 gap-4">
			{#each hardwareAccelerations as hardwareAcceleration}
				<button
					class="btn h-20 rounded {selectedHardwareAcceleration === hardwareAcceleration
						? 'btn-active btn-primary'
						: ''}"
					on:click={() => selectOption('hardwareAcceleration', hardwareAcceleration)}
				>
					{hardwareAcceleration}
				</button>
			{/each}
		</div>
	</div>

	<div class="col-span-1 bg-success r-heading rounded p-2 text-xl">
		<h3>Build</h3>
	</div>
	<div class="col-span-4">
		<div class="grid grid-cols-2 gap-4">
			{#each builds as build}
				<button
					class="btn rounded {selectedBuild === build ? 'btn-active btn-primary btn-primary' : ''}"
					on:click={() => selectOption('build', build)}
				>
					{build}
				</button>
			{/each}
		</div>
	</div>

	<div class="col-span-1 bg-success rounded p-2">
		<h3 class="text-xl">Installation Instructions</h3>
	</div>
	<div class="col-span-4 bg-base-300 rounded">
		<div class="p-4">
            <div id="installation-instructions">
                {@html installationInstructions}
            </div>
            {#if installationInstructions.includes('<pre><code')}
                <button class="btn btn-primary btn-sm mt-4" on:click={copyCodeToClipboard}>
                    {" "}Copy code 
                    <span class="copy-btn-icon w-4"><FaRegClipboard/></span>
                </button>
            {/if}
        </div>
	</div>
</div>