"""Run browser QA with private temporary DB/video/logs and owned local services.

  python e2e/run_browser_qa.py --backend-python /workspace/.venvs/attendance/bin/python \
      --artifacts /tmp/attendance-ui-qa --report /tmp/attendance-ui-qa.json
Needs installed frontend dependencies/models, Python Playwright, Chromium, ffmpeg.
Ports8000/5173 must be free; this never stops an existing user's service.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time

import httpx
from browser_ui import run

ROOT = Path(__file__).resolve().parents[1]


def available(port):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(('127.0.0.1', port))
        except OSError as error:
            raise RuntimeError(f'Port{port} already in use; stop your own service or use standalone browser_ui.py') from error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-python', type=Path, required=True)
    parser.add_argument('--artifacts', type=Path, default=Path('/tmp/attendance-ui-qa'))
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    available(8000); available(5173)
    with tempfile.TemporaryDirectory(prefix='attendance-browser-qa-') as folder:
        folder = Path(folder)
        video = folder / 'simulated-static-camera.y4m'
        subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-loop', '1', '-i', str(ROOT/'face1.jpg'),
                        '-vf', 'scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2',
                        '-r', '15', '-t', '3', '-pix_fmt', 'yuv420p', '-y', str(video)], check=True)
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", ATTENDANCE_DATABASE_URL=f'sqlite:///{folder}/qa.db',
                           ATTENDANCE_MODEL_DIR=str(ROOT/'.local/models'), ATTENDANCE_INFERENCE_DEVICE='cpu')
        services=[]
        with open(folder/'backend.log','w+') as backend_log, open(folder/'frontend.log','w+') as frontend_log:
            try:
                services.append(subprocess.Popen([str(args.backend_python.absolute()),'-m','uvicorn','main:app',
                    '--host','127.0.0.1','--port','8000'],cwd=ROOT,env=environment,stdout=backend_log,stderr=backend_log,
                    start_new_session=True))
                services.append(subprocess.Popen([str(ROOT/'frontend/node_modules/.bin/vite'),'--host','127.0.0.1',
                    '--port','5173','--strictPort'],cwd=ROOT/'frontend',stdout=frontend_log,stderr=frontend_log,
                    start_new_session=True))
                for _ in range(200):
                    if any(service.poll() is not None for service in services):
                        raise RuntimeError('Owned QA service exited before readiness')
                    try:
                        api=httpx.get('http://127.0.0.1:5173/api/health',timeout=1)
                        if api.status_code==200 and api.json()['models_ready']:
                            break
                    except (httpx.RequestError, ValueError):
                        pass
                    time.sleep(.1)
                else:
                    raise RuntimeError('Owned QA services did not become ready')
                result=run('http://127.0.0.1:5173',video,args.artifacts)
                output=json.dumps(result,indent=2);print(output)
                if args.report:args.report.write_text(output+'\n')
            except BaseException:
                for log in (backend_log,frontend_log):
                    log.seek(0);print(log.read()[-6000:])
                raise
            finally:
                for service in reversed(services):
                    if service.poll() is None:
                        os.killpg(service.pid,signal.SIGTERM)
                        try:service.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(service.pid,signal.SIGKILL);service.wait(timeout=5)
