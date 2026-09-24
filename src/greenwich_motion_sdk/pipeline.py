"""Python entrypoint for the existing multibody backend; no service ports."""
from dataclasses import dataclass
from pathlib import Path
import json,subprocess

@dataclass(frozen=True)
class RunRequest:
    indices: tuple[int,...]
    output: Path
    robots: Path
    representation: str='soma77'
    stage: str='All'
    fps: float=30.
    max_seconds: float=15.
    contact_config: Path | None=None

class Pipeline:
    """SDK façade over the shared full-body conversion backend.

    Use run for indexed SOMA batches, convert for explicit format manifests.
    Caller owns toolkit/config/local-paths.json; no machine paths in this API.
    """
    def __init__(self,toolkit:Path,powershell='pwsh'):
        self.toolkit=Path(toolkit).resolve();self.powershell=powershell

    def command(self,request:RunRequest):
        if not request.indices or any(i<1 for i in request.indices):raise ValueError('Positive source indices required')
        if request.representation not in ('soma77','smpl22-derived'):raise ValueError('Backend representation unsupported')
        if request.stage not in ('All','Generate','Audit','Render'):raise ValueError('Invalid stage')
        if request.fps<=0 or request.max_seconds<=0:raise ValueError('Invalid run budget')
        return [self.powershell,'-NoProfile','-File',str(self.toolkit/'scripts/invoke.ps1'),
            '-Indices',','.join(map(str,request.indices)),'-Output',str(Path(request.output).resolve()),
            '-Robots',str(Path(request.robots).resolve()),'-Stage',request.stage,
            '-InputRepresentation',request.representation,'-Fps',str(request.fps),
            '-MaxSeconds',str(request.max_seconds)
            ]+(['-ContactConfig',str(Path(request.contact_config).resolve())] if request.contact_config else [])

    def run(self,request:RunRequest):
        """Synchronous, errors propagated. Call from a job queue for background use."""
        command=self.command(request);output=Path(request.output);output.mkdir(parents=True,exist_ok=True)
        (output/'sdk-request.json').write_text(json.dumps({'sdk_version':'0.1.0','command':command},indent=2))
        subprocess.run(command,check=True)
        return {'output':str(output.resolve()),'summary':str(output/'summary.json'),'accuracy':str(output/'accuracy.json')}

    def convert(self,source_manifest,output,robots,*,fps=30.,max_seconds=15.,contact_config=None,compact=False,model='prime',run_dir=None):
        """Run native SMPL/BVH/SOMA/canonical inputs through the shared backend.

        Source manifest declares adapter options, exact units/floor and roles.
        No resampling, trajectory resizing or pose presets are introduced.
        Output is numeric only; use Render separately for review videos.
        """
        if fps<=0 or max_seconds<=0:raise ValueError('Invalid conversion budget')
        if model not in ('bumblebee','megatron','prime'):raise ValueError('Choose a v2 model: bumblebee/megatron/prime')
        from .source_batch import read_sources
        read_sources(source_manifest)  # Validate all source contracts before model startup.
        output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
        command=[self.powershell,'-NoProfile','-File',str(self.toolkit/'scripts/invoke.ps1'),
            '-Stage','Convert','-SourceManifest',str(Path(source_manifest).resolve()),
            '-Output',str(output),'-Robots',str(Path(robots).resolve()),'-Fps',str(fps),
            '-MaxSeconds',str(max_seconds),'-Model',model,'-SkipPreview']
        if compact:command+=['-Compact']
        if contact_config:command+=['-ContactConfig',str(Path(contact_config).resolve())]
        if run_dir:command+=['-RunDir',str(Path(run_dir).resolve())]
        (output/'sdk-request.json').write_text(json.dumps({'sdk_version':'0.1.0','command':command},indent=2),encoding='utf-8')
        subprocess.run(command,check=True)
        return {'output':str(output),'results':str(output/'results.json')}
