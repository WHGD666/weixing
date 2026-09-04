[CmdletBinding()]
param(
    [string]$Output = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Parent = Split-Path $Root -Parent
$ExpectedModelHash = "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24"
$Model = Join-Path $Root "models\pretrained\yolo11x.pt"

if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) {
    throw "Missing verified model: $Model"
}
$ModelHash = (Get-FileHash -LiteralPath $Model -Algorithm SHA256).Hash.ToLowerInvariant()
if ($ModelHash -ne $ExpectedModelHash) {
    throw "Unexpected yolo11x.pt SHA256: $ModelHash"
}

$ImageCount = @(Get-ChildItem -LiteralPath (Join-Path $Root "data3\images\train") -File).Count
$LabelCount = @(Get-ChildItem -LiteralPath (Join-Path $Root "data3\labels\train") -Filter "*.txt" -File |
    Where-Object { $_.Name -ne "classes.txt" }).Count
if ($ImageCount -ne 4481 -or $LabelCount -ne 4481) {
    throw "Data3 count mismatch: images=$ImageCount labels=$LabelCount"
}

if ([string]::IsNullOrWhiteSpace($Output)) {
    $Output = Join-Path $Parent "overshiyan_remote_20260904.tar"
}
$Archive = [IO.Path]::GetFullPath($Output)
if (-not $Archive.StartsWith($Parent, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Archive must be written under $Parent"
}
if ((Test-Path -LiteralPath $Archive) -and -not $Force) {
    throw "Archive already exists: $Archive. Use -Force to replace it."
}

$Arguments = @(
    "-cf", $Archive,
    "--exclude=overshiyan/workspace",
    "--exclude=overshiyan/runs",
    "--exclude=overshiyan/artifacts",
    "--exclude=overshiyan/__pycache__",
    "--exclude=overshiyan/*/__pycache__",
    "--exclude=overshiyan/*/*/__pycache__",
    "--exclude=overshiyan/yolo11x.pt",
    "overshiyan"
)

Push-Location $Parent
try {
    & tar.exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "tar.exe failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

$ArchiveHash = (Get-FileHash -LiteralPath $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
$HashPath = "$Archive.sha256"
"$ArchiveHash  $([IO.Path]::GetFileName($Archive))" | Set-Content -LiteralPath $HashPath -Encoding ascii
$SizeGB = [Math]::Round((Get-Item -LiteralPath $Archive).Length / 1GB, 3)
Write-Host "ok=true archive=$Archive size_gb=$SizeGB sha256=$ArchiveHash"
Write-Host "hash_file=$HashPath images=$ImageCount labels=$LabelCount model_sha256=$ModelHash"
