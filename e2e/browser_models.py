"""Actual MediaPipe browser inference with a local pinned npm package and fixtures.

  python e2e/browser_models.py --package-dir frontend/node_modules/@mediapipe/tasks-vision
Uses installed Python Playwright and system Chromium. No camera or liveness claim.
"""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def run(package_dir, executable='/usr/bin/chromium'):
    with tempfile.TemporaryDirectory(prefix='attendance-mediapipe-browser-') as folder:
        folder = Path(folder)
        (folder / 'package').symlink_to(package_dir.resolve(), target_is_directory=True)
        (folder / 'model.task').symlink_to(ROOT / 'frontend/public/models/face_landmarker.task')
        for fixture in ['face1.jpg', 'face2.jpg']:
            (folder / fixture).symlink_to(ROOT / fixture)
        (folder / 'index.html').write_text('<!doctype html><title>Independent actual MediaPipe fixture inference</title>')
        server = ThreadingHTTPServer(('127.0.0.1', 0), partial(QuietHandler, directory=str(folder)))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(executable_path=executable, headless=True, args=['--no-sandbox'])
                page = browser.new_page()
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}/index.html')
                result = page.evaluate('''async () => {
                  const { FaceLandmarker, FilesetResolver } = await import('/package/vision_bundle.mjs');
                  const fileset = await FilesetResolver.forVisionTasks('/package/wasm');
                  const model = await FaceLandmarker.createFromOptions(fileset, {
                    baseOptions: { modelAssetPath:'/model.task', delegate:'CPU' }, runningMode:'IMAGE',
                    numFaces:2, outputFaceBlendshapes:true, outputFacialTransformationMatrixes:true
                  });
                  const results = [];
                  const rotation = d => ({yaw:Math.atan2(d[8],d[10])*180/Math.PI,
                    pitch:Math.atan2(-d[9],Math.hypot(d[8],d[10]))*180/Math.PI});
                  try {
                    for (const name of ['face1.jpg','face2.jpg']) {
                      const image = new Image(); image.src='/'+name; await image.decode();
                      for (const mirrored of [false,true]) {
                        const canvas = document.createElement('canvas');
                        canvas.width=image.naturalWidth;canvas.height=image.naturalHeight;
                        const context=canvas.getContext('2d');
                        if(mirrored){context.translate(canvas.width,0);context.scale(-1,1);}
                        context.drawImage(image,0,0);
                        const output=model.detect(canvas);
                        const matrix=output.facialTransformationMatrixes[0];
                        const face=output.faceLandmarks[0];
                        results.push({fixture:name,mirrored,faces:output.faceLandmarks.length,
                          landmarks:face?.length??0, matrix, rotation:matrix?rotation(matrix.data):null,
                          noseRelativeToEyeMidpoint:face?face[1].x-(face[33].x+face[263].x)/2:null,
                          blendshapeCount:output.faceBlendshapes[0]?.categories.length??0});
                      }
                    }
                  } finally { model.close(); }
                  return {results,version:'0.10.32',delegate:'CPU',camera:false};
                }''')
                assert not errors, errors
                for row in result['results']:
                    assert row['faces'] == 1 and row['landmarks'] == 478 and row['blendshapeCount'] > 0, row
                    matrix = row['matrix']
                    assert matrix['rows'] == 4 and matrix['columns'] == 4 and len(matrix['data']) == 16
                    data = matrix['data']
                    assert max(abs(data[i]) for i in [3, 7, 11]) < 1e-5 and abs(data[15]-1) < 1e-5
                    assert abs(data[14]) > 1, 'Expected column-major translation in index14'
                for index in [0, 2]:
                    normal, mirrored = result['results'][index:index+2]
                    assert normal['rotation']['yaw'] * mirrored['rotation']['yaw'] < 0
                    assert abs(normal['rotation']['yaw'] + mirrored['rotation']['yaw']) < 5
                result['checks'] = ['actual local task+WASM loaded', 'one face and478landmarks on both fixtures',
                                    'actual matrix translation confirms column-major layout',
                                    'horizontal mirroring reverses measured yaw sign', 'model lifecycle close executed']
                result['limitations'] = 'Static-image browser inference only; no real phone, camera, or physical motion validation.'
                browser.close()
                return result
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-dir', type=Path, default=ROOT / 'frontend/node_modules/@mediapipe/tasks-vision')
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    result = run(args.package_dir)
    output = json.dumps(result, indent=2)
    print(output)
    if args.report:
        args.report.write_text(output+'\n')
