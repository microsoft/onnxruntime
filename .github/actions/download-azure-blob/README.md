# Azure Blob SDK downloads

This Windows composite action uses GitHub OIDC via pinned `azure/login`, then
AzCopy's [Azure CLI authentication](https://learn.microsoft.com/azure/storage/common/storage-use-azcopy-authorize-user-identity#authorize-by-using-azure-cli).
The CLI login remains available until the copy completes. Each invocation has a
unique Azure CLI configuration and AzCopy log/plan directory, removed by an
`always()` step after success or failure. It does not use `azcopy login`, the
shared OS token store, or a VM managed-identity fallback. Abrupt runner termination
can prevent cleanup; runner lifecycle cleanup must still remove temporary files.

Callers must grant the downloading job `contents: read` and `id-token: write`.
The SDK workflows use the action's checked-in identity defaults; no repository
variables are required. Callers may explicitly override `client-id` and `tenant-id`.
Omitted or explicitly empty inputs use the upstream Blob reader client ID
`bcb3be91-097b-4440-8714-45ad5deeeae7` and tenant ID
`72f988bf-86f1-41af-91ab-2d7cd011db47`, respectively. This also handles fork
workflows where upstream repository variables are unavailable. These public IDs
are not credentials: OIDC permissions, federation, and Blob RBAC are still required.
Invalid configuration fails before login.

The identity must have GitHub federation for the caller's event/ref subjects and
Blob data read/list access to the existing storage sources (for example, Storage
Blob Data Reader at the appropriate scope). The identity defaults do not establish
or verify read-only RBAC. No subscription ID is needed for Blob data-plane access;
login uses `allow-no-subscriptions: true`. Federation, RBAC, and network access
are infrastructure prerequisites, not configured by this action.

Both `ort-github-win-16-amd-scus-ss` and `ort-github-win-6-a10-scus-ss` images need
Azure CLI >= 2.30.0 and AzCopy supporting `AZCOPY_AUTO_LOGIN_TYPE=AZCLI`.
Preflight checks tool compatibility without requesting a token. Provision these
tools on the shared runner images if necessary; no SDK version or storage
location changes are part of this authentication migration.
