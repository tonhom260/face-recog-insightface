"""Start an isolated real-model server, run independent API checks, delete DB.

  PYTHONPATH=. python e2e/run_api_qa.py --different-person /tmp/attendance-astronaut.jpg
The selected Python environment needs the attendance application's dependencies.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time

import httpx
from api_real_inference import run

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--different-person', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix='attendance-api-qa-') as folder:
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", ATTENDANCE_DATABASE_URL=f'sqlite:///{folder}/qa.db',
                   ATTENDANCE_MODEL_DIR=str(ROOT / '.local/models'), ATTENDANCE_INFERENCE_DEVICE='cpu')
        with open(Path(folder) / 'server.log', 'w+') as log:
            server = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'main:app', '--host', '127.0.0.1',
                                       '--port', str(port)], cwd=ROOT, env=env, stdout=log, stderr=log)
            url = f'http://127.0.0.1:{port}'
            try:
                for _ in range(300):
                    if server.poll() is not None:
                        raise RuntimeError('Disposable QA server exited before becoming ready')
                    try:
                        if httpx.get(url + '/api/health', timeout=1).status_code == 200:
                            break
                    except httpx.RequestError:
                        pass
                    time.sleep(.1)
                else:
                    raise RuntimeError('Disposable QA server did not become ready')
                report = run(url, args.different_person)
                output = json.dumps(report, indent=2)
                print(output)
                if args.report:
                    args.report.write_text(output + '\n')
            except BaseException:
                log.seek(0)
                print(log.read()[-12000:], file=sys.stderr)
                raise
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill(); server.wait(timeout=5)
