from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STAGES = ('modeling', 'regions', 'photoz')

def available_systems():
    from .discovery import discover_systems
    return discover_systems()

def system_record(name):
    for item in available_systems():
        if item['id'] == name:
            return item
    raise ValueError(f'Unknown system: {name}')

def stage_artifacts(name, stage):
    system_record(name)
    if stage == 'modeling':
        return [f'1.lens-modeling/{name}/output/stage_previews/stage{i}_fit.png' for i in range(1, 5)]
    if stage == 'regions':
        return [f'2.region-detection/regions/{name}/{name}_regions.png']
    if stage == 'photoz':
        return [f'3.photometric-estimation/results/{name}/{file}' for file in ['flux.png', 'pz.png']]
    raise ValueError(stage)

def run_stage(name, stage, *, mode='render', model_python=None, photo_python=None):
    """Run the same worker used by notebooks and the local web interface."""
    system_record(name)
    if stage not in STAGES or mode not in ('render', 'fit'):
        raise ValueError('Unknown stage or modeling mode')
    executable = (model_python or os.environ.get('PHOTO_LENS_MODEL_PYTHON') or sys.executable) if stage == 'modeling' else (photo_python or os.environ.get('PHOTO_LENS_PHOTO_PYTHON') or sys.executable)
    command = [str(executable), '-u', str(ROOT / 'scripts/run_stage.py'), stage, '--system', name, '--mode', mode]
    subprocess.run(command, cwd=ROOT, check=True)
    return [ROOT / path for path in stage_artifacts(name, stage)]
