param([string]$file_folder, [string]$account_key, [string]$trt_container, [string]$csc)

Add-Type -AssemblyName System.Web
$wheel_file = [System.IO.Path]::GetFileName((Get-ChildItem $file_folder))
$ort_trt_ep_pkg_blob_path = 'ort-trt-ep/' + $env:BUILD_BUILDNUMBER + '/' + $wheel_file
$expiredays = New-TimeSpan -Days 1
$end = (Get-Date) + $expiredays

$body = @{grant_type='client_credentials'
client_id='bcb87687-5d9d-4c21-801e-317980c8b1d5'
client_secret=$csc
scope='api://2227e307-9325-4dbe-9894-5c3b25d62a2d/.default'}
$contentType = 'application/x-www-form-urlencoded'
$res = Invoke-WebRequest -Method POST -Uri https://login.microsoftonline.com/cc38825a-ff99-423f-bdde-dd14d00e33b8/oauth2/v2.0/token -body $body -ContentType $contentType | ConvertFrom-Json

Write-Host "Before send"
$token = $res.access_token

$headers = @{Authorization = "Bearer $token"}

$ort_trt_ep_pkg_sas_uri = az storage blob generate-sas -c upload -n $ort_trt_ep_pkg_blob_path --account-name anubiscustomerstorage --account-key $account_key --full-uri --permissions r --expiry $end.ToString("yyyy-MM-ddTHH:mmZ") --https-only

$ort_trt_ep_pkg_sas_uri = $ort_trt_ep_pkg_sas_uri.Substring(1, $ort_trt_ep_pkg_sas_uri.Length-2)

$body_trt_perf_compare = @{
   "Name"="TRT_PERF_COMPARE";
   "Parameters" = @{         
        “TRT_VERSION”=$trt_container;
        “BUILD_NUMBER”=$env:BUILD_BUILDNUMBER;
        "ORT_TRT_EP_PKG_SAS_URI"=$ort_trt_ep_pkg_sas_uri};
}

$anubissvctesturl = "https://anubistest.azurewebsites.net/api/mlperf/jobs"

Write-Host ($body_trt_perf_compare|ConvertTo-Json)

Invoke-RestMethod -Method 'Post' -Uri $anubissvctesturl -Body ($body_trt_perf_compare|ConvertTo-Json) -ContentType "application/json" -Headers $headers -UseBasicParsing
$body.Parameters