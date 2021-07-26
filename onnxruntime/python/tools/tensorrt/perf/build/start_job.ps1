param([string]$file_folder, [string]$account_key)

Add-Type -AssemblyName System.Web
$wheel_file = [System.IO.Path]::GetFileName((Get-ChildItem $file_folder))
$ort_trt_ep_pkg_blob_path = 'ort-trt-ep/' + $wheel_file
$expiredays = New-TimeSpan -Days 1
$end = (Get-Date) + $expiredays

$ort_trt_ep_pkg_sas_uri = az storage blob generate-sas -c upload -n $ort_trt_ep_pkg_blob_path --account-name anubiscustomerstorage --account-key $account_key --full-uri --permissions r --expiry $end.ToString("yyyy-MM-ddTHH:mmZ") --https-only

$ort_trt_ep_pkg_sas_uri = $ort_trt_ep_pkg_sas_uri.Substring(1, $ort_trt_ep_pkg_sas_uri.Length-2)

$body_trt_perf_compare = @{
   "Name"="TRT_PERF_COMPARE";
   "Parameters" = @{         
        “TRT_VERSION”="21.04";
        “BUILD_NUMBER”=$env:BUILD_BUILDNUMBER;
        "ORT_TRT_EP_PKG_SAS_URI"=$ort_trt_ep_pkg_sas_uri};
}

$anubissvctesturl = "https://anubistest.azurewebsites.net/api/mlperf/jobs"

Write-Host ($body_trt_perf_compare|ConvertTo-Json)

Invoke-RestMethod -Method 'Post' -Uri $anubissvctesturl -Body ($body_trt_perf_compare|ConvertTo-Json) -ContentType "application/json"
$body.Parameters
