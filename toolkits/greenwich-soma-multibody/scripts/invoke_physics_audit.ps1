param(
    [string]$Source,
    [string]$Robot = 'h2',
    [string]$Robots,
    [Parameter(Mandatory=$true)][string]$Output,
    [string]$Cache,
    [string]$Config,
    [ValidateSet('final','projected_model_height','projected_supplied_height')][string]$Stage='final'
)
$ErrorActionPreference='Stop'
$kit=Split-Path $PSScriptRoot -Parent
$sdk=Split-Path (Split-Path $kit -Parent) -Parent
$paths=Get-Content -LiteralPath (Join-Path $kit 'config/local-paths.json') -Raw | ConvertFrom-Json
$env:PYTHONPATH=Join-Path $sdk 'src'
$argsList=@('-m','greenwich_motion_sdk.physics_audit','--output',$Output)
if ($Cache) {$argsList+=@('--cache',$Cache)}
else {
    if (-not $Source) {throw 'Supply Source or Cache'}
    if (-not $Robots) { $Robots=Join-Path $kit 'config/robots.example.json' }
    $argsList+=@('--source',$Source,'--robot',$Robot,'--robots',$Robots,'--repo',$paths.repo,'--pipeline',$paths.pipeline,'--stage',$Stage)
}
if ($Config) {$argsList+=@('--config',$Config)}
& $paths.python @argsList
if ($LASTEXITCODE -ne 0) {throw "Physics audit execution failed: $LASTEXITCODE"}
