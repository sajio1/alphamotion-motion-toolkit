import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pytest
from greenwich_motion_sdk.source_batch import read_sources,sample_clip
from greenwich_motion_sdk.pipeline import Pipeline


def fixture(tmp_path):
    names=['pelvis','head','lh','rh','lf','rf','lt','rt']
    offsets=[[0,0,0],[0,.5,0],[-.3,.2,0],[.3,.2,0],[-.1,-1,0],[.1,-1,0],[0,0,.1],[0,0,.1]]
    np.savez(tmp_path/'native.npz',poses=np.zeros((6,24)),trans=np.tile([2.,1.,3.],(6,1)),mocap_framerate=30.)
    options={'names':names,'parents':[-1,0,0,0,0,0,4,5],'rest_offsets':offsets,
             'pelvis_offset_source_units':[0,0,0],
             'frame':{'basis_to_yup':np.eye(3).tolist(),'cm_per_unit':100.,'ground_y_cm':0.}}
    (tmp_path/'profile.json').write_text(json.dumps(options))
    source={'id':'native','format':'smpl','path':'native.npz','options_file':'profile.json',
            'roles':names[:6],'foot_pairs':[['lf','lt'],['rf','rt']]}
    path=tmp_path/'sources.json';path.write_text(json.dumps({'schema':'alphamotion.sources.v1','sources':[source]}))
    return path,source


def test_native_absolute_units_timing_and_canonical(tmp_path):
    path,source=fixture(tmp_path);row=read_sources(path)[0];clip=row['_clip']
    np.testing.assert_allclose(clip.world_position_cm[0,0],[200,100,300])
    np.testing.assert_allclose(clip.world_position_cm[:,4,1],0)
    take,_=sample_clip(clip,15.,15.);np.testing.assert_array_equal(take,[0,2,4])
    clip.save(tmp_path/'canonical.npz')
    source.update(format='canonical',path='canonical.npz');source.pop('options_file')
    path.write_text(json.dumps({'schema':'alphamotion.sources.v1','sources':[source]}))
    np.testing.assert_allclose(read_sources(path)[0]['_clip'].world_position_cm,clip.world_position_cm)


def test_invalid_semantics_and_resampling_rejected(tmp_path):
    path,source=fixture(tmp_path);clip=read_sources(path)[0]['_clip']
    with pytest.raises(ValueError):sample_clip(clip,24.,15.)
    clip.timestamps_s[2]+=.001
    with pytest.raises(ValueError):sample_clip(clip,30.,15.)
    source['roles'][0]='head';path.write_text(json.dumps({'schema':'alphamotion.sources.v1','sources':[source]}))
    with pytest.raises(ValueError):read_sources(path)


def test_convert_command_uses_shared_entrypoint(tmp_path):
    path,_=fixture(tmp_path)
    with patch('greenwich_motion_sdk.pipeline.subprocess.run') as run:
        result=Pipeline(tmp_path/'toolkit').convert(path,tmp_path/'out',tmp_path/'robots.json')
    cmd=run.call_args.args[0]
    assert cmd[cmd.index('-Stage')+1]=='Convert'
    assert cmd[cmd.index('-SourceManifest')+1]==str(path.resolve())
    assert '-ContactIterations' not in cmd
    assert '-SkipPreview' in cmd and result['results'].endswith('results.json')
