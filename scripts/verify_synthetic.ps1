$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    python -m compileall -q data models explanations training tests
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    python -m pytest -q
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
