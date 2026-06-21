param(
    [string]$Remote = "user@server.example.com",
    [string]$RemoteRoot = "~/automine-loop/data_select",
    [string]$LocalRoot = "C:\path\to\automine-loop\data_select"
)

$ErrorActionPreference = "Stop"

New-Item -ItemType Directory -Force -Path "$LocalRoot\reports" | Out-Null
New-Item -ItemType Directory -Force -Path "$LocalRoot\data\manifests" | Out-Null
New-Item -ItemType Directory -Force -Path "$LocalRoot\data\embeddings" | Out-Null
New-Item -ItemType Directory -Force -Path "$LocalRoot\data\indexes" | Out-Null

scp -r "${Remote}:$RemoteRoot/reports/*" "$LocalRoot\reports\"
scp -r "${Remote}:$RemoteRoot/data/manifests/*" "$LocalRoot\data\manifests\"
scp -r "${Remote}:$RemoteRoot/data/embeddings/*" "$LocalRoot\data\embeddings\"
scp -r "${Remote}:$RemoteRoot/data/indexes/*" "$LocalRoot\data\indexes\"
