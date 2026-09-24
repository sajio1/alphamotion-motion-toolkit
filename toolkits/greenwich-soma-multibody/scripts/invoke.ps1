[CmdletBinding()]
param(
    [ValidateSet('All','Generate','Convert','Evaluate','Audit','Render','ScreenSample','ScreenAll','ScreenStatistical','StatisticsReport','SupportSweep','SurfaceAudit','RescreenSurfaces','RankPreviews','SelectPreviews','Gallery','PackageSource')]
    [string]$Stage = 'All',
    [string]$Indices,
    [Parameter(Mandatory=$true)]
    [string]$Output,
    [string]$Robots,
    [string]$DatasetWorkspace,
    [string]$SourceManifest,
    [ValidateSet('bumblebee','megatron','prime')]
    [string]$Model = 'prime',
    [string]$RunDir,
    [string]$ContactConfig,
    [ValidateSet('soma77','smpl22-derived')]
    [string]$InputRepresentation = 'soma77',
    [switch]$SkipPreview,
    [switch]$Compact,
    [switch]$SkipSummary,
    [string]$Cache,
    [double]$Fps = 30,
    [ValidateSet('auto','row','grid')]
    [string]$RenderLayout = 'auto',
    [int]$RenderHeight = 1080,
    [double]$RenderFormationScale = 1.0,
    [double]$RenderCameraElevation = 0.65,
    [double]$RenderCameraZoom = 1.0,
    [double]$RenderCameraAzimuth = -45.0,
    [double]$RenderLookAtHeightBias = 0.0,
    [switch]$HideRobotLabels,
    [switch]$LeadSource,
    [double]$ContactOverlayAlpha = 0.85,
    [double]$MaxSeconds = 15,
    [string]$ReviewManifest,
    [string]$ViewerDescriptor,
    [string]$ViewerXml,
    [int]$ReviewCount = 10,
    [switch]$ReviewSourceComparison,
    [string]$BaseCard,
    [string]$HfRepo = 'sajio/alphamotion-soma-h2',
    [int]$GalleryWidth = 960,
    [int]$GalleryFps = 10,
    [double]$GallerySeconds = 5,
    [string]$GalleryAssetDirectory,
    [string[]]$GalleryClips,
    [string[]]$GalleryAdditionalManifests,
    [int]$GalleryColumns = 2,
    [switch]$GalleryContactLabels,
    [switch]$GalleryQaSheet,
    [switch]$GallerySeparate,
    [switch]$GalleryHideUI,
    [switch]$GalleryContactGlow,
    [string]$GalleryAssetName = 'soma-h2-grid',
    [string]$GalleryOldLabel,
    [string]$GalleryNewLabel,
    [int[]]$GalleryReplacePanels = @(1),
    [switch]$Publish,
    [switch]$UploadPrepared,
    [switch]$StreamlinedRelease,
    [string]$SourceMetadata,
    [string]$ReleaseIndex,
    [string]$ArchiveRoot,
    [string]$CheckpointConfig,
    [string]$SourceArchive,
    [string]$SourceCache,
    [int]$SampleCount = 1000,
    [int]$ScreenLimit = 0,
    [string]$ReuseDirectory,
    [int]$SampleSeed = 20260912,
    [string]$SampleRobot = 'h2',
    [double]$SlipCmS = 15.0,
    [double]$PenetrationCm = 0.5,
    [double]$BadDurationSeconds = 0.1,
    [switch]$PrepareOnly,
    [string]$SupportConfig,
    [ValidateSet('legacy','kimodo')]
    [string]$MetricProfile = 'legacy',
    [string]$AuditReport,
    [string]$SweepSpeeds = '7.5,10,15,20,25',
    [string]$SweepDurations = '0.1',
    [switch]$MemoryOnly
)

$ErrorActionPreference = 'Stop'
$toolkit = Split-Path -Parent $PSScriptRoot
$paths = Get-Content (Join-Path $toolkit 'config/local-paths.json') -Raw | ConvertFrom-Json
if ($Stage -eq 'PackageSource') {
    $sdkRoot = Split-Path (Split-Path $toolkit -Parent) -Parent
    $env:PYTHONPATH = Join-Path $sdkRoot 'src'
    & $paths.python -m greenwich_motion_sdk.source_package --sdk-root $sdkRoot --runtime $paths.pipeline --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Source packaging failed' }
    return
}
if ($Stage -eq 'Convert') {
    if (-not $SourceManifest -or -not $Robots) { throw 'Convert requires SourceManifest and a robot roster' }
    if (-not $ContactConfig) { $ContactConfig=Join-Path $toolkit 'config/contact.default.json' }
    $env:PYTHONPATH=Join-Path (Split-Path (Split-Path $toolkit -Parent) -Parent) 'src'
    $convertArgs=@((Join-Path $PSScriptRoot 'run_soma_locomotion.py'),'--source-manifest',$SourceManifest,'--repo',$paths.repo,'--pipeline',$paths.pipeline,'--robots',$Robots,'--output',$Output,'--fps',$Fps,'--max-seconds',$MaxSeconds,'--contact-config',$ContactConfig,'--ffmpeg',$paths.ffmpeg)
    if ($SkipPreview) { $convertArgs+='--skip-preview' }
    if ($Compact) { $convertArgs+='--compact' }
    if ($Cache) { $convertArgs+=@('--cache',$Cache) }
    $convertArgs+=@('--model',$Model)
    if ($RunDir) { $convertArgs+=@('--run-dir',$RunDir) }
    & $paths.python @convertArgs
    if ($LASTEXITCODE -ne 0) { throw 'Full-body source conversion failed' }
    return
}
if ($Stage -eq 'Evaluate') {
    if (-not $AuditReport -or -not $Robots -or -not $ViewerDescriptor) { throw 'Evaluate requires conversion root in AuditReport, Robots and ViewerDescriptor' }
    $env:PYTHONPATH=(Join-Path (Split-Path (Split-Path $toolkit -Parent) -Parent) 'src')+';'+(Join-Path $paths.repo 'src')+';'+$paths.sole_backend
    & $paths.python -m greenwich_motion_sdk.evaluation --root $AuditReport --robots $Robots --robot $SampleRobot --descriptor $ViewerDescriptor --output $Output --slip-cm-s $SlipCmS --penetration-cm $PenetrationCm --bad-duration-s $BadDurationSeconds
    if ($LASTEXITCODE -ne 0) { throw 'Saved output evaluation failed' }
    return
}
if ($Stage -eq 'RankPreviews') {
    if (-not $ReleaseIndex -or -not $ArchiveRoot -or -not $ViewerDescriptor -or -not $SourceCache) { throw 'RankPreviews requires release index/root, descriptor and cached source directory' }
    $env:PYTHONPATH = Join-Path (Split-Path -Parent (Split-Path -Parent $toolkit)) 'src'
    & $paths.python -m greenwich_motion_sdk.preview_candidates --index $ReleaseIndex --archive-root $ArchiveRoot --descriptor $ViewerDescriptor --cached-roots $SourceCache --output $Output --seconds $GallerySeconds
    if ($LASTEXITCODE -ne 0) { throw 'Candidate ranking failed' }
    return
}
if ($Stage -eq 'SelectPreviews') {
    if (-not $ReleaseIndex -or -not $ArchiveRoot -or -not $GalleryClips -or -not $SourceCache) { throw 'SelectPreviews requires ReleaseIndex, ArchiveRoot, GalleryClips and SourceCache' }
    $env:PYTHONPATH = Join-Path (Split-Path -Parent (Split-Path -Parent $toolkit)) 'src'
    & $paths.python -m greenwich_motion_sdk.review_sources --select-cached --index $ReleaseIndex --archive-root $ArchiveRoot --output $Output --cached-roots $SourceCache --clips @GalleryClips
    if ($LASTEXITCODE -ne 0) { throw 'Preview selection failed' }
    return
}
if ($Stage -eq 'RescreenSurfaces') {
    if (-not $AuditReport -or -not $SourceCache) { throw 'RescreenSurfaces requires foot directory in AuditReport and surface directory in SourceCache' }
    $env:PYTHONPATH = Join-Path (Split-Path -Parent (Split-Path -Parent $toolkit)) 'src'
    & $paths.python -m greenwich_motion_sdk.surface_audit --rescreen --foot-directory $AuditReport --surface-directory $SourceCache --output $Output --slip-cm-s $SlipCmS --penetration-cm $PenetrationCm --duration-s $BadDurationSeconds
    if ($LASTEXITCODE -ne 0) { throw 'Cached whole-surface rescreen failed' }
    return
}
if ($Stage -eq 'SurfaceAudit') {
    $env:OPENBLAS_NUM_THREADS = '4'
    $env:OMP_NUM_THREADS = '4'
    if (-not $ReviewManifest -or -not $AuditReport -or -not $ViewerDescriptor -or -not $ViewerXml -or -not $SourceArchive) { throw 'SurfaceAudit requires selection in ReviewManifest, foot ledger in AuditReport, descriptor/XML and source archive' }
    if (-not $SupportConfig) { $SupportConfig = Join-Path $toolkit 'config/support-screen.conservative.json' }
    $env:PYTHONPATH = (Join-Path (Split-Path -Parent (Split-Path -Parent $toolkit)) 'src') + ';' + $paths.optional_pythonlibs
    $env:PYTHONDONTWRITEBYTECODE = '1'
    & $paths.python -m greenwich_motion_sdk.surface_audit --selection $ReviewManifest --foot-ledger $AuditReport --descriptor $ViewerDescriptor --xml $ViewerXml --source-archive $SourceArchive --output $Output --support-config $SupportConfig --slip-cm-s $SlipCmS --penetration-cm $PenetrationCm --duration-s $BadDurationSeconds
    if ($LASTEXITCODE -ne 0) { throw 'Whole-surface audit failed' }
    return
}
if ($Stage -eq 'StatisticsReport') {
    if (-not $AuditReport) { throw 'StatisticsReport requires AuditReport pointing to statistics.json' }
    $env:PYTHONPATH = Join-Path (Split-Path -Parent (Split-Path -Parent $toolkit)) 'src'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    & $paths.python -m greenwich_motion_sdk.sampling_statistics --input $AuditReport --output $Output
    if ($LASTEXITCODE -ne 0) { throw 'Statistical report rendering failed' }
    return
}
if ($Stage -eq 'SupportSweep') {
    if (-not $AuditReport) { throw 'SupportSweep requires AuditReport with cached fixed-support CSVs' }
    $env:PYTHONPATH = Join-Path (Split-Path -Parent (Split-Path -Parent $toolkit)) 'src'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    & $paths.python -m greenwich_motion_sdk.support_sweep --report $AuditReport --output $Output --speeds $SweepSpeeds --durations $SweepDurations --provenance (Join-Path $toolkit 'config/support-screen.references.json')
    if ($LASTEXITCODE -ne 0) { throw 'Cached fixed-support threshold sweep failed' }
    return
}
if ($Stage -in @('ScreenSample','ScreenAll','ScreenStatistical')) {
    if (-not $ReleaseIndex -or -not $ArchiveRoot -or -not $SourceArchive -or -not $ViewerDescriptor) {
        throw 'ScreenSample requires ReleaseIndex, ArchiveRoot, SourceArchive and ViewerDescriptor'
    }
    if (-not $Robots) { $Robots = Join-Path $toolkit 'config/robots.a3-h2-g1.json' }
    $env:PYTHONPATH = (Join-Path (Split-Path -Parent (Split-Path -Parent $toolkit)) 'src') + ';' +
        (Join-Path $paths.repo 'src') + ';' + $paths.sole_backend + ';' + $paths.optional_pythonlibs
    $env:PYTHONDONTWRITEBYTECODE = '1'
    if ($Stage -in @('ScreenAll','ScreenStatistical')) {
        if (-not $SupportConfig) { $SupportConfig = Join-Path $toolkit 'config/support-screen.conservative.json' }
        $fullArgs = @('-m','greenwich_motion_sdk.corpus_support','--index',$ReleaseIndex,'--archive-root',$ArchiveRoot,'--source-archive',$SourceArchive,'--robots',$Robots,'--descriptor',$ViewerDescriptor,'--output',$Output,'--support-config',$SupportConfig,'--slip-cm-s',$SlipCmS,'--duration-s',$BadDurationSeconds,'--penetration-cm',$PenetrationCm,'--robot',$SampleRobot)
        if ($ScreenLimit -gt 0) { $fullArgs += @('--limit',$ScreenLimit) }
        if ($Stage -eq 'ScreenStatistical') { $fullArgs += @('--sample-count',$SampleCount,'--sample-seed',$SampleSeed) }
        if ($ReuseDirectory) { $fullArgs += @('--reuse-directory',$ReuseDirectory) }
        & $paths.python @fullArgs
        if ($LASTEXITCODE -ne 0) { throw 'Full-corpus fixed-support screen failed' }
        return
    }
    $screenArgs = @('-m','greenwich_motion_sdk.corpus_screen','--index',$ReleaseIndex,
        '--archive-root',$ArchiveRoot,'--source-archive',$SourceArchive,'--robots',$Robots,
        '--descriptor',$ViewerDescriptor,'--count',$SampleCount,'--seed',$SampleSeed,'--robot',$SampleRobot,
        '--slip-cm-s',$SlipCmS,'--penetration-cm',$PenetrationCm,'--bad-duration-s',$BadDurationSeconds,
        '--metric-profile',$MetricProfile)
    if (-not $MemoryOnly) { $screenArgs += @('--output',$Output) }
    if ($PrepareOnly) { $screenArgs += '--prepare-only' }
    if ($SupportConfig) { $screenArgs += @('--support-config',$SupportConfig) }
    if ($SourceCache) { $screenArgs += @('--source-cache',$SourceCache) }
    & $paths.python @screenArgs
    if ($LASTEXITCODE -ne 0) { throw 'Corpus sample screen failed' }
    return
}
if ($Stage -eq 'Gallery') {
    if (-not $ReviewManifest -or -not $GalleryClips) { throw 'Gallery requires ReviewManifest and GalleryClips' }
    if ($Publish) { throw 'Gallery is local preview only; publication is not supported' }
    $env:PYTHONPATH = Join-Path (Split-Path -Parent (Split-Path -Parent $toolkit)) 'src'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $galleryArgs = @('-m','greenwich_motion_sdk.card_gallery','--manifest',$ReviewManifest,'--preview-only',
        '--asset-directory',$Output,'--asset-name',$GalleryAssetName,'--width',$GalleryWidth,
        '--fps',$GalleryFps,'--duration',$GallerySeconds,'--columns',$GalleryColumns,'--clips') + $GalleryClips
    if ($GalleryContactLabels) { $galleryArgs += '--contact-labels' }
    if ($GalleryQaSheet) { $galleryArgs += '--qa-sheet' }
    if ($GallerySeparate) { $galleryArgs += '--separate' }
    if ($GalleryHideUI) { $galleryArgs += '--hide-ui' }
    if ($GalleryContactGlow) { $galleryArgs += '--contact-glow' }
    if ($ViewerDescriptor) { $galleryArgs += @('--viewer-descriptor',$ViewerDescriptor) }
    if ($GalleryAdditionalManifests) { $galleryArgs += @('--additional-manifests') + $GalleryAdditionalManifests }
    & $paths.python @galleryArgs
    if ($LASTEXITCODE -ne 0) { throw 'Local gallery generation failed' }
    return
}
if ($ReviewManifest) {
    if ($Stage -ne 'Render') { throw 'ReviewManifest requires Stage Render' }
    if (-not $ViewerDescriptor -or -not $ViewerXml) { throw 'ViewerDescriptor and ViewerXml are required' }
    $env:PYTHONPATH = Join-Path (Split-Path -Parent (Split-Path -Parent $toolkit)) 'src'
    $env:PYTHONDONTWRITEBYTECODE = '1'
    $reviewArgs = @('-m','greenwich_motion_sdk.render_review','--manifest',$ReviewManifest,'--descriptor',$ViewerDescriptor,'--xml',$ViewerXml,'--output',$Output,'--ffmpeg',$paths.ffmpeg,'--count',$ReviewCount,'--height',$RenderHeight)
    if ($ReviewSourceComparison) { $reviewArgs += '--compare-source' }
    & $paths.python @reviewArgs
    if ($LASTEXITCODE -ne 0) { throw 'Saved review rendering failed' }
    return
}
if (-not $Indices) { throw 'Indices is required for pipeline generation/rendering' }
if (-not $Robots) { $Robots = Join-Path $toolkit 'config/robots.a3-h2-x2.json' }
if (-not $DatasetWorkspace) { $DatasetWorkspace = $paths.dataset_workspace }
if (-not $ContactConfig) { $ContactConfig = Join-Path $toolkit 'config/contact.default.json' }
$resultRoot = [System.IO.Path]::GetFullPath($Output)
$videoRoot = Join-Path $resultRoot 'vid'

$required = @($paths.repo,$paths.pipeline,$DatasetWorkspace,$paths.python,$paths.ffmpeg,$Robots,$ContactConfig)
foreach ($item in $required) {
    if (-not (Test-Path -LiteralPath $item)) { throw "Required path does not exist: $item" }
}
New-Item -ItemType Directory -Force -Path $resultRoot | Out-Null
$env:PYTHONDONTWRITEBYTECODE = '1'

if ($Stage -in @('All','Generate')) {
    $generationArgs = @(
        (Join-Path $PSScriptRoot 'run_soma_locomotion.py'),
        '--repo',$paths.repo,'--pipeline',$paths.pipeline,
        '--dataset-workspace',$DatasetWorkspace,'--robots',$Robots,
        '--output',$resultRoot,'--indices',$Indices,'--fps',$Fps,
        '--max-seconds',$MaxSeconds,
        '--contact-config',$ContactConfig,'--ffmpeg',$paths.ffmpeg,'--input-representation',$InputRepresentation
    )
    if ($SkipPreview) { $generationArgs += '--skip-preview' }
    if ($Compact) { $generationArgs += '--compact' }
    if ($Cache) { $generationArgs += @('--cache',$Cache) }
    $generationArgs += @('--model',$Model)
    if ($RunDir) { $generationArgs += @('--run-dir',$RunDir) }
    & $paths.python @generationArgs
    if ($LASTEXITCODE -ne 0) { throw "Generation failed with exit code $LASTEXITCODE" }
}

if ($Stage -in @('All','Audit')) {
    & $paths.python (Join-Path $PSScriptRoot 'audit_transfer_accuracy.py') --root $resultRoot --repo $paths.repo --robots $Robots
    if ($LASTEXITCODE -ne 0) { throw "Accuracy audit failed with exit code $LASTEXITCODE" }
    & $paths.python (Join-Path $PSScriptRoot 'audit_soma_contact.py') $resultRoot
    if ($LASTEXITCODE -ne 0) { throw "Contact audit failed with exit code $LASTEXITCODE" }
    & $paths.python (Join-Path $PSScriptRoot 'audit_soma_foot_phases.py') `
        --repo $paths.repo --pipeline $paths.pipeline --root $resultRoot --robots $Robots
    if ($LASTEXITCODE -ne 0) { throw "Foot-phase audit failed with exit code $LASTEXITCODE" }
}

if ($Stage -in @('All','Render')) {
    New-Item -ItemType Directory -Force -Path $videoRoot | Out-Null
    $renderArgs = @(
        (Join-Path $PSScriptRoot 'render_soma_robot_comparison.py'),
        '--repo',$paths.repo,'--pipeline',$paths.pipeline,'--root',$resultRoot,
        '--robots',$Robots,'--output',$videoRoot,'--ffmpeg',$paths.ffmpeg,
        '--layout',$RenderLayout,'--indices',$Indices,'--height',$RenderHeight,
        '--formation-scale',$RenderFormationScale,'--camera-elevation',$RenderCameraElevation,
        '--camera-zoom',$RenderCameraZoom,'--camera-azimuth',$RenderCameraAzimuth,
        '--lookat-height-bias',$RenderLookAtHeightBias,'--contact-overlay-alpha',$ContactOverlayAlpha
    )
    if ($HideRobotLabels) { $renderArgs += '--hide-robot-labels' }
    if ($LeadSource) { $renderArgs += '--lead-source' }
    & $paths.python @renderArgs
    if ($LASTEXITCODE -ne 0) { throw "Rendering failed with exit code $LASTEXITCODE" }
}

if (-not $SkipSummary) {
    & $paths.python (Join-Path $PSScriptRoot 'summarize_soma_locomotion.py') $resultRoot
    if ($LASTEXITCODE -ne 0) { throw "Summary failed with exit code $LASTEXITCODE" }
}
Write-Host "Completed $Stage pipeline: $resultRoot"
