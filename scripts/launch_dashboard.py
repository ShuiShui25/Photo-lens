"""Locate installed environments, start the local dashboard, and open a browser."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parents[1]


def environments():
    roots = [Path(sys.prefix).parent]
    if os.environ.get('CONDA_PREFIX'):
        roots.append(Path(os.environ['CONDA_PREFIX']).parent)
    for name in ('miniforge3', 'miniconda3', 'anaconda3', 'mambaforge'):
        roots.append(Path.home() / name / 'envs')
    paths = []
    conda = os.environ.get('CONDA_EXE') or shutil.which('conda')
    if conda:
        try:
            result = subprocess.run([conda, 'env', 'list', '--json'],
                                    capture_output=True, text=True, timeout=15, check=True)
            paths = [Path(p) for p in json.loads(result.stdout)['envs']]
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            pass
    return paths, roots


def python_path(explicit, names, paths, roots):
    if explicit:
        candidates = [Path(explicit).expanduser()]
    else:
        candidates = [p / 'bin/python' for name in names
                      for p in [*[p for p in paths if p.name == name],
                                *[root / name for root in roots]]]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            # Keep the environment path: resolving a Python symlink can select
            # the base environment instead of the intended virtual environment.
            return str(candidate.absolute())
    raise RuntimeError(f'Python environment not found: {explicit or " / ".join(names)}. '
                       'Install it using README.md or provide --photo-python / --model-python.')


def dashboard_running(url):
    # Local requests must bypass HTTP proxy environment variables.
    http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with http.open(url + '/api/systems', timeout=1) as response:
            systems = json.load(response)
        with http.open(url + '/api/session', timeout=1) as response:
            session = json.load(response)
        return (isinstance(systems, list) and all(isinstance(s, dict) and 'id' in s for s in systems)
                and isinstance(session.get('token'), str))
    except (OSError, ValueError, AttributeError):
        return False


def open_dashboard(url, no_browser):
    print(f'Photo-lens: {url}', flush=True)
    if not no_browser:
        try:
            opened = webbrowser.open(url, new=2)
        except webbrowser.Error:
            opened = False
        if not opened:
            print('Could not open a browser automatically. Open the URL above manually.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--photo-python', default=os.environ.get('PHOTO_LENS_PHOTO_PYTHON'))
    parser.add_argument('--model-python', default=os.environ.get('PHOTO_LENS_MODEL_PYTHON'))
    parser.add_argument('--sextractor', default=os.environ.get('PHOTO_LENS_SEXTRACTOR'))
    parser.add_argument('--no-browser', action='store_true', help='Start without opening a browser')
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('--port must be between 1 and 65535')
    url = f'http://127.0.0.1:{args.port}'
    if dashboard_running(url):
        print('Dashboard is already running; opening the existing page.', flush=True)
        open_dashboard(url, args.no_browser)
        return 0
    paths, roots = environments()
    photo = python_path(args.photo_python, ('photo-lens', 'bcnz'), paths, roots)
    model = python_path(args.model_python, ('photo-lens-modeling', 'autolens-jax'), paths, roots)
    sextractor = args.sextractor or next((str(Path(photo).parent / name)
        for name in ('source-extractor', 'sex') if (Path(photo).parent / name).is_file()), None)
    sextractor = sextractor or shutil.which('source-extractor') or shutil.which('sex')
    if not sextractor or not os.access(sextractor, os.X_OK):
        raise RuntimeError('SExtractor not found. Install source-extractor or provide --sextractor.')
    print(f'Photometry Python: {photo}\nModeling Python: {model}', flush=True)
    process = subprocess.Popen([photo, '-u', str(ROOT / 'serve.py'), '--port', str(args.port),
        '--model-python', model, '--photo-python', photo, '--sextractor', sextractor],
        cwd=ROOT, start_new_session=True)
    try:
        deadline = time.monotonic() + 20
        while not dashboard_running(url):
            if process.poll() is not None:
                raise RuntimeError('Dashboard failed to start; see the error above. '
                                   'If the port is occupied, try --port 8766.')
            if time.monotonic() >= deadline:
                raise RuntimeError('Dashboard startup timed out.')
            time.sleep(0.2)
        if process.poll() is not None:
            raise RuntimeError('Dashboard exited during startup; see the error above.')
        open_dashboard(url, args.no_browser)
        print('Keep this terminal open. Press Ctrl+C to stop the dashboard.', flush=True)
        return process.wait()
    except KeyboardInterrupt:
        return 0
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (RuntimeError, OSError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        sys.exit(1)
