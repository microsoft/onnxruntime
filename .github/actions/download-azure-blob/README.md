# Azure Blob SDK downloads

This Windows composite action uses GitHub OIDC via pinned `azure/login`, then
AzCopy's [Azure CLI authentication](https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-authorize-user-identity#authorize-by-using-azure-cli).
The CLI login remains available until the copy completes. Each invocation has a
unique Azure CLI configuration and AzCopy log/plan directory, removed by an
`always()` step after success or failure. It does not use `azcopy login`, the
shared OS token store, or a VM managed-identity fallback. Abrupt runner termination
can prevent cleanup; runner lifecycle cleanup must still remove temporary files.

Callers must grant the downloading job `contents: read` and `id-token: write`,
and pass `vars.STORAGE_BLOB_READ_CLIENT_ID` as `client-id` and
`vars.AZURE_TENANT_ID` as `tenant-id`. Composite actions cannot read the `vars`
context directly. Missing configuration fails before login.

The identity must have GitHub federation for the caller's event/ref subjects and
Blob data read/list access to the existing storage sources (for example, Storage
Blob Data Reader at the appropriate scope). Its variable name does not establish
or verify read-only RBAC. No subscription ID is needed for Blob data-plane access;
login uses `allow-no-subscriptions: true`. Federation, RBAC, and network access
are infrastructure prerequisites, not configured by this action.

Both `ort-github-win-16-amd-scus-ss` and `ort-github-win-6-a10-scus-ss` images need
Azure CLI >= 2.30.0 and AzCopy supporting `AZCOPY_AUTO_LOGIN_TYPE=AZCLI`.
Preflight checks tool compatibility without requesting a token. Provision these
tools on the shared runner images if necessary; no SDK version or storage
location changes are part of this authentication migration.
