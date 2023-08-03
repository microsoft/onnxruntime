&nvidia-smi
$ort_src_root = (Get-Item $PSScriptRoot)
npm install
npx playwright install chromium
npx playwright test gpu.spec.js
