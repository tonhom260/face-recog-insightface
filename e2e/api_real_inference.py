"""Independent API verification against a disposable attendance server.

Run with attendance Python environment:
  python e2e/api_real_inference.py --url http://127.0.0.1:8011 \
      --different-person /tmp/attendance-astronaut.jpg

The server MUST point to a fresh temporary database. The script refuses initialized
servers, creates randomized temporary credentials, uses real model inference, and
never modifies shipped employee records. The different-person input must depict a
known different individual (the public-domain skimage astronaut is suitable).
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import uuid

import cv2
import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def run(url: str, different_person: Path) -> dict:
    results: list[str] = []
    similarities: dict[str, float] = {}
    admin = httpx.Client(base_url=url, timeout=90)
    user = httpx.Client(base_url=url, timeout=90)
    stranger = httpx.Client(base_url=url, timeout=90)

    def expect(response, status, label, code=None):
        assert response.status_code == status, (label, response.status_code, response.text)
        if code:
            assert response.json()['error']['code'] == code, (label, response.text)
        results.append(label)
        return response.json() if response.content else None

    def photo(path):
        return ('fixture.jpg', Path(path).read_bytes(), 'image/jpeg')

    def attend(client, action='check_in', request_id=None, image=None, **extra):
        fields = dict(action=action, request_id=request_id or str(uuid.uuid4()),
                      client_challenge_passed='true',
                      challenge_actions=json.dumps(['turn_left', 'smile', 'blink']))
        fields.update(extra)
        return client.post('/api/attendance', data=fields,
                           files={'image': image or photo(ROOT / 'face2.jpg')})

    try:
        health = expect(admin.get('/api/health'), 200, 'health ready')
        assert health['database_ready'] and health['models_ready'], health
        setup = expect(admin.get('/api/setup'), 200, 'fresh disposable database')
        assert setup['needs_setup'], 'Refusing initialized database: use a fresh temporary DB.'
        expect(stranger.get('/api/auth/me'), 401, 'anonymous session rejected')
        expect(stranger.get('/api/attendance'), 401, 'anonymous history rejected')
        password = secrets.token_urlsafe(24)
        suffix = secrets.token_hex(6)
        setup_body = {'employee_code': f'qa-admin-{suffix}', 'name': 'QA Temporary Admin', 'password': password}
        setup_response = admin.post('/api/setup', json=setup_body)
        admin_employee = expect(setup_response, 201, 'first administrator setup')['employee']
        cookie_header = setup_response.headers['set-cookie'].lower()
        assert 'httponly' in cookie_header and 'samesite=lax' in cookie_header
        results.append('authentication cookie is HttpOnly and SameSite Lax')
        expect(admin.post('/api/employees', json=setup_body, headers={'Origin': 'https://untrusted.example'}), 403, 'cross-origin mutation rejected')
        expect(admin.post('/api/setup', json=setup_body), 409, 'repeat setup rejected')
        assert not admin.get('/api/setup').json()['needs_setup']
        employee_body = {'employee_code': f'qa-employee-{suffix}', 'name': 'QA Temporary Employee', 'password': password}
        employee = expect(admin.post('/api/employees', json=employee_body), 201, 'admin creates employee')['employee']
        other = expect(admin.post('/api/employees', json={**employee_body, 'employee_code': f'qa-other-{suffix}'}), 201,
                       'second employee created')['employee']
        expect(admin.post('/api/employees', json=employee_body), 409, 'duplicate employee code rejected')
        login_body = {'employee_code': employee_body['employee_code'], 'password': password}
        expect(user.post('/api/auth/login', json={**login_body, 'password': 'incorrect'}), 401, 'wrong password rejected')
        expect(user.post('/api/auth/login', json=login_body), 200, 'employee login')
        assert user.get('/api/auth/me').json()['employee']['id'] == employee['id']
        expect(user.get('/api/employees'), 403, 'employee admin list forbidden')
        expect(user.post('/api/employees', json=employee_body), 403, 'employee account creation forbidden')
        expect(user.get('/api/attendance', params={'employee_id': other['id']}), 403, 'employee cross-user history forbidden')
        expect(user.post('/api/face/enroll', data={'employee_id': other['id']},
                         files={'image': photo(ROOT / 'face1.jpg')}), 403, 'employee cross-user enrollment forbidden')
        expect(attend(user), 422, 'unenrolled attendance rejected', 'FACE_NOT_ENROLLED')
        malformed = ('broken.jpg', b'not a photograph', 'image/jpeg')
        expect(user.post('/api/face/enroll', files={'image': malformed}), 422, 'malformed enrollment rejected')
        success, blank = cv2.imencode('.jpg', np.full((480, 640, 3), 127, dtype=np.uint8))
        assert success
        noface = ('blank.jpg', blank.tobytes(), 'image/jpeg')
        expect(user.post('/api/face/enroll', files={'image': noface}), 422, 'no-face enrollment rejected', 'FACE_NOT_FOUND')
        pair = np.concatenate([cv2.resize(cv2.imread(str(ROOT / name)), (338, 600)) for name in ('face1.jpg', 'face2.jpg')], axis=1)
        ok, pair_data = cv2.imencode('.jpg', pair)
        assert ok
        multiface = ('two-faces.jpg', pair_data.tobytes(), 'image/jpeg')
        expect(user.post('/api/face/enroll', files={'image': multiface}), 422, 'multiple-face enrollment rejected', 'MULTIPLE_FACES')
        enrolled = expect(user.post('/api/face/enroll', files={'image': photo(ROOT / 'face1.jpg')}), 200,
                          'real face1 enrollment')
        assert enrolled['employee']['face_enrolled'] and enrolled['model_version']
        expect(user.post('/api/face/enroll', files={'image': photo(ROOT / 'face1.jpg')}), 409,
               'employee cannot replace enrollment')
        expect(attend(user, image=noface), 422, 'no-face attendance rejected', 'FACE_NOT_FOUND')
        expect(attend(user, image=malformed), 422, 'malformed attendance rejected')
        expect(attend(user, image=multiface), 422, 'multiple-face attendance rejected', 'MULTIPLE_FACES')
        expect(attend(user, image=photo(different_person)), 422, 'known different-person face rejected', 'FACE_MISMATCH')
        for fields, label in [({'client_challenge_passed': 'false'}, 'failed client challenge rejected'),
                              ({'challenge_actions': '["blink","blink","smile"]'}, 'repeated challenge actions rejected'),
                              ({'challenge_actions': '["blink","smile"]'}, 'short challenge rejected'),
                              ({'challenge_actions': '["blink","smile","unsupported"]'}, 'unsupported challenge rejected'),
                              ({'challenge_actions': '{}'}, 'non-array challenge rejected'),
                              ({'request_id': 'not-a-uuid'}, 'invalid request ID rejected'),
                              ({'employee_id': other['id']}, 'client employee identity override rejected'),
                              ({'similarity': '1.0'}, 'client similarity override rejected')]:
            expect(attend(user, **fields), 422, label)
        assert user.get('/api/attendance').json()['events'] == []
        assert user.get('/api/attendance/status').json() == {'next_action': 'check_in', 'last_event': None}
        request_id = str(uuid.uuid4())
        before = datetime.now(timezone.utc)
        checked_in = expect(attend(user, request_id=request_id), 201, 'real face2 genuine check-in')
        after = datetime.now(timezone.utc)
        event = checked_in['event']
        assert not checked_in['duplicate'] and event['employee_id'] == employee['id']
        assert event['client_challenge_passed'] and event['request_id'] == request_id
        timestamp = datetime.fromisoformat(event['occurred_at'].replace('Z', '+00:00'))
        assert timestamp.utcoffset().total_seconds() == 0 and before <= timestamp <= after
        similarities['genuine_face1_face2'] = event['similarity']
        assert isinstance(event['similarity'], (int, float))
        duplicate = expect(attend(user, request_id=request_id), 200, 'same-request retry deduplicated')
        assert duplicate['duplicate'] and duplicate['event'] == event
        expect(attend(user, request_id=request_id, action='check_out'), 409, 'request ID cannot be reused with different action')
        expect(attend(user), 409, 'second check-in transition rejected', 'INVALID_TRANSITION')
        assert len(user.get('/api/attendance').json()['events']) == 1
        request_id_out = str(uuid.uuid4())
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(attend, user, action='check_out', request_id=request_id_out) for _ in range(2)]
            responses = [f.result() for f in futures]
        assert sorted(r.status_code for r in responses) == [200, 201], [(r.status_code, r.text) for r in responses]
        assert len({r.json()['event']['id'] for r in responses}) == 1
        results.append('concurrent same-request submission creates exactly one event')
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(attend, user) for _ in range(2)]
            responses = [f.result() for f in futures]
        assert sorted(r.status_code for r in responses) == [201, 409], [(r.status_code, r.text) for r in responses]
        results.append('concurrent different-request check-ins enforce transition')
        history = expect(user.get('/api/attendance'), 200, 'employee history') ['events']
        assert len(history) == 3 and all(e['employee_id'] == employee['id'] for e in history)
        assert [e['action'] for e in history] == ['check_in', 'check_out', 'check_in']
        assert history == sorted(history, key=lambda e: e['occurred_at'], reverse=True)
        admin_history = expect(admin.get('/api/attendance', params={'employee_id': employee['id']}), 200,
                               'admin can filter employee history')['events']
        assert admin_history == history
        assert admin.get('/api/attendance').json()['events'] == []
        expect(admin.delete(f"/api/employees/{employee['id']}/face"), 204, 'admin enrollment reset')
        expect(attend(user, action='check_out'), 422, 'reset enrollment cannot attend', 'FACE_NOT_ENROLLED')
        expect(user.post('/api/face/enroll', files={'image': photo(ROOT / 'face1.jpg')}), 200, 'employee re-enrollment after reset')
        expect(admin.patch(f"/api/employees/{employee['id']}", json={'active': False}), 200, 'admin disables employee')
        expect(user.get('/api/auth/me'), 401, 'disabled account session revoked')
        expect(user.post('/api/auth/login', json=login_body), 403, 'disabled employee login rejected')
        expect(admin.patch(f"/api/employees/{employee['id']}", json={'active': True}), 200, 'admin reactivates employee')
        expect(user.post('/api/auth/login', json=login_body), 200, 'reactivated employee login')
        session_cookie = user.cookies.get('attendance_session')
        expect(user.post('/api/auth/logout'), 204, 'logout')
        expect(user.get('/api/auth/me'), 401, 'logged-out session rejected')
        stranger.cookies.set('attendance_session', session_cookie)
        expect(stranger.get('/api/auth/me'), 401, 'logged-out session cookie replay rejected')
        expect(admin.get('/api/challenge'), 404, 'no server challenge-code endpoint')
        return {'passed': len(results), 'checks': results, 'similarities': similarities,
                'inference_device': health.get('inference_device'), 'inference_providers': health.get('inference_providers'),
                'model_version': enrolled['model_version'], 'different_person_fixture': different_person.name}
    finally:
        admin.close(); user.close(); stranger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--different-person', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args()
    report = run(args.url, args.different_person)
    output = json.dumps(report, indent=2)
    print(output)
    if args.report:
        args.report.write_text(output + '\n')
