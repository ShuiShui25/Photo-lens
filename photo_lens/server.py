"""Local web interface with a serial background worker queue."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlsplit
from .workflow import ROOT, STAGES, available_systems, stage_artifacts, system_record

class Jobs:
    def __init__(self, model_python, photo_python):
        self.model_python, self.photo_python = model_python, photo_python
        self.jobs = {}; self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.directory = ROOT / '.runtime/jobs'; self.directory.mkdir(parents=True, exist_ok=True)
        self.stopping = False

    def submit(self, name, stage, mode):
        record = system_record(name)
        if stage not in STAGES or mode not in ('render','fit'): raise ValueError('Invalid stage or mode')
        if not record.get('can_run', {}).get(stage, True):
            raise ValueError('Complete the required inputs and preceding stage first. '
                             + ' '.join(record.get('input_issues', [])))
        with self.lock:
            if self.stopping: raise ValueError('Server is stopping')
            if any(j['system']==name and j['state'] in ('queued','running') for j in self.jobs.values()):
                raise ValueError('A task is already queued or running for this system')
            ident = secrets.token_hex(8)
            job = {'id':ident,'system':name,'stage':stage,'mode':mode,'state':'queued','started':None,'finished':None,'returncode':None}
            self.jobs[ident]=job
            self.pool.submit(self.execute, ident)
            return dict(job)

    def execute(self, ident):
        with self.lock:
            job=self.jobs[ident]
            if job['state']=='cancelled' or self.stopping:return
            job['state']='running';job['started']=time.time()
        executable=self.model_python if job['stage']=='modeling' else self.photo_python
        cmd=[executable,'-u',str(ROOT/'scripts/run_stage.py'),job['stage'],'--system',job['system'],'--mode',job['mode']]
        path=self.directory/f'{ident}.log'
        try:
            with path.open('wb') as log:
                log.write(f'Starting {job["stage"]} for {job["system"]} (mode: {job["mode"]})\nLaunching Python worker; initial imports may take a moment.\n'.encode())
                log.flush()
                with self.lock:
                    if job['state']=='cancelled' or self.stopping:return
                    process=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env={**os.environ,'MPLBACKEND':'Agg','OPENBLAS_NUM_THREADS':'1'})
                    job['_process']=process
                code=process.wait()
            with self.lock:
                job['returncode']=code
                if job['state']!='cancelled':job['state']='succeeded' if code==0 else 'failed'
        except Exception as exc:
            with path.open('a') as log:log.write(str(exc)+'\n')
            with self.lock:job['state']='failed'
        finally:
            with self.lock:
                job.pop('_process',None);job['finished']=time.time()
                (self.directory/f'{ident}.json').write_text(json.dumps(job,indent=2)+'\n')

    def get(self, ident):
        with self.lock:
            if ident not in self.jobs:raise KeyError(ident)
            job={k:v for k,v in self.jobs[ident].items() if not k.startswith('_')}
        path=self.directory/f'{ident}.log'
        if path.exists():
            with path.open('rb') as log:
                log.seek(max(0,path.stat().st_size-65536));job['log']=log.read().decode('utf-8','replace')
        else:job['log']=''
        return job

    def cancel(self,ident):
        with self.lock:
            job=self.jobs[ident]
            if job['state'] not in ('queued','running'):return
            job['state']='cancelled';job['finished']=time.time()
            process=job.get('_process')
            if process and process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass

    def close(self):
        self.stopping=True
        for ident in list(self.jobs):self.cancel(ident)
        self.pool.shutdown(wait=True,cancel_futures=True)

class Handler(SimpleHTTPRequestHandler):
    def __init__(self,*args, jobs, token, **kwargs):
        self.jobs,self.token=jobs,token
        super().__init__(*args,directory=str(ROOT),**kwargs)

    def allowed_host(self):
        return self.headers.get('Host') in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}

    def reply(self,code,data):
        body=json.dumps(data).encode();self.send_response(code)
        self.send_header('Content-Type','application/json; charset=utf-8');self.send_header('Content-Length',str(len(body)))
        self.end_headers();self.wfile.write(body)

    def do_GET(self):
        if not self.allowed_host():return self.reply(403,{'error':'Invalid host'})
        path=urlsplit(self.path).path
        if path=='/api/session':return self.reply(200,{'token':self.token})
        if path=='/api/systems':return self.reply(200,available_systems())
        if path=='/api/jobs':
            with self.jobs.lock:ids=list(self.jobs.jobs)
            return self.reply(200,[self.jobs.get(i) for i in ids])
        if path.startswith('/api/jobs/'):
            try:return self.reply(200,self.jobs.get(path.rsplit('/',1)[-1]))
            except KeyError:return self.reply(404,{'error':'Job not found'})
        relative=unquote(path).lstrip('/') or 'index.html'
        target=(ROOT/relative).resolve()
        if not target.is_relative_to(ROOT) or any(part.startswith('.') for part in Path(relative).parts) or target.name.endswith('.local.json'):
            return self.reply(403,{'error':'Path is not public'})
        if not target.is_file():return self.reply(404,{'error':'File not found'})
        super().do_GET()

    def do_POST(self):
        if not self.allowed_host():return self.reply(403,{'error':'Invalid host'})
        if not secrets.compare_digest(self.headers.get('X-Photo-Lens-Token',''),self.token):
            return self.reply(403,{'code':'session_expired','error':'The backend session changed. Reconnect before submitting the task.'})
        origin=self.headers.get('Origin')
        if origin and origin != f'http://{self.headers.get("Host")}':return self.reply(403,{'error':'Invalid origin'})
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=4096:raise ValueError('Invalid request size')
            body=json.loads(self.rfile.read(length))
            path=urlsplit(self.path).path
            if path=='/api/jobs':return self.reply(202,self.jobs.submit(body['system'],body['stage'],body.get('mode','render')))
            if path.startswith('/api/jobs/') and path.endswith('/cancel'):
                ident=path.split('/')[3];self.jobs.cancel(ident);return self.reply(200,self.jobs.get(ident))
            return self.reply(404,{'error':'Endpoint not found'})
        except (ValueError,KeyError,TypeError) as exc:return self.reply(400,{'error':str(exc)})

    def end_headers(self):
        # HTML, scripts and plots all change in place during local analysis.
        # Apply this to static responses too, not just the JSON endpoints.
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','same-origin')
        super().end_headers()


def main():
    parser=argparse.ArgumentParser(description='Serve the Photo-lens dashboard on localhost')
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--model-python',default=os.environ.get('PHOTO_LENS_MODEL_PYTHON',sys.executable))
    parser.add_argument('--photo-python',default=os.environ.get('PHOTO_LENS_PHOTO_PYTHON',sys.executable))
    parser.add_argument('--sextractor',default=os.environ.get('PHOTO_LENS_SEXTRACTOR'))
    args=parser.parse_args()
    if args.sextractor:os.environ['PHOTO_LENS_SEXTRACTOR']=args.sextractor
    jobs=Jobs(args.model_python,args.photo_python)
    server=ThreadingHTTPServer(('127.0.0.1',args.port),partial(Handler,jobs=jobs,token=secrets.token_urlsafe(32)))
    print(f'Photo-lens: http://127.0.0.1:{args.port}',flush=True)
    try:server.serve_forever()
    except KeyboardInterrupt:pass
    finally:server.server_close();jobs.close()
if __name__=='__main__':main()
