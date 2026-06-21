param(
    [string]$Remote = "user@server.example.com",
    [string]$RemoteRoot = "~/automine-loop",
    [string]$LocalRoot = "C:\path\to\automine-loop"
)

$ErrorActionPreference = "Stop"

ssh $Remote "mkdir -p $RemoteRoot/data_select"
scp -r `
    "$LocalRoot\data_select\app" `
    "$LocalRoot\data_select\configs" `
    "$LocalRoot\data_select\scripts" `
    "$LocalRoot\data_select\src" `
    "$LocalRoot\data_select\tests" `
    "$LocalRoot\data_select\pyproject.toml" `
    "$LocalRoot\data_select\README.md" `
    "${Remote}:$RemoteRoot/data_select/"
