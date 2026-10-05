"""Independent browser workflows on a fresh disposable attendance server.

  python e2e/browser_ui.py --url http://127.0.0.1:5173 \
      --camera-file /tmp/attendance-camera-fixture.y4m --artifacts /tmp/attendance-ui-qa
The Y4M file is a SIMULATED static camera sourced from face1.jpg. Real MediaPipe
runs for enrollment and static-photo rejection. Attendance completion additionally
uses explicit test-only synthetic landmark/motion data in this browser process;
production code has no bypass. Backend face inference remains real throughout.
Never use an employee/production database: this script creates randomized accounts.
"""
import argparse
import json
from pathlib import Path
import secrets

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
SIZES = [(390, 844), (768, 1024), (1440, 900)]


def run(url, camera_file, artifacts):
    artifacts.mkdir(parents=True, exist_ok=True)
    results, layouts, small_targets, console_errors = [], [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path='/usr/bin/chromium', headless=True, args=[
            '--no-sandbox', '--use-fake-ui-for-media-stream', '--use-fake-device-for-media-stream',
            '--use-file-for-fake-video-capture=' + str(camera_file.resolve())])
        context = browser.new_context(viewport={'width':390,'height':844}, locale='th-TH')
        context.add_init_script('''(() => {
          window.__qaStreams=[];window.__qaLighting=1;window.__qaMotionEnabled=true;
          const original=navigator.mediaDevices?.getUserMedia.bind(navigator.mediaDevices);
          if(original) navigator.mediaDevices.getUserMedia=async options=>{
            const sourceStream=await original(options);
            const source=document.createElement('video');source.muted=true;source.playsInline=true;
            source.srcObject=sourceStream;await source.play();
            const canvas=document.createElement('canvas');canvas.width=640;canvas.height=480;
            const context=canvas.getContext('2d');let stopped=false,raf=0;
            const draw=()=>{if(stopped)return;context.filter=`brightness(${window.__qaLighting})`;
              context.drawImage(source,0,0,640,480);raf=requestAnimationFrame(draw);};draw();
            const stream=canvas.captureStream(15);
            stream.getTracks().forEach(track=>{const stop=track.stop.bind(track);track.stop=()=>{
              stopped=true;cancelAnimationFrame(raf);sourceStream.getTracks().forEach(t=>t.stop());
              source.pause();source.srcObject=null;stop();};});
            window.__qaStreams.push(stream);return stream;
          };
        })();''')
        page = context.new_page()
        page.on('pageerror', lambda error: console_errors.append(str(error)))
        page.goto(url)
        assert context.request.get(url + '/api/setup').json()['needs_setup'], 'Refusing initialized server: use a fresh disposable DB.'
        expect(page.get_by_role('heading', name='สร้างผู้ดูแลคนแรก')).to_be_visible()

        def snapshot(name, widths=SIZES):
            for width, height in widths:
                page.set_viewport_size({'width':width, 'height':height})
                page.wait_for_timeout(120)
                extent = page.evaluate('({width:innerWidth,scroll:document.documentElement.scrollWidth})')
                assert extent['scroll'] <= extent['width'], (name, width, extent)
                layouts.append({'screen':name,'width':width,'height':height,'horizontal_overflow':False})
                targets = page.evaluate('''() => [...document.querySelectorAll('button,input:not([type=file])')].filter(e=>{
                  const r=e.getBoundingClientRect();return r.width>0&&r.height>0&&getComputedStyle(e).visibility!=='hidden';
                }).map(e=>({text:(e.getAttribute('aria-label')||e.innerText||e.name).trim(),
                  width:e.getBoundingClientRect().width,height:e.getBoundingClientRect().height})).filter(e=>e.width<44||e.height<44)''')
                if width == 390 and targets:
                    small_targets.append({'screen':name,'targets':targets})
                page.screenshot(path=str(artifacts / f'{name}-{width}.png'), full_page=True)
            page.set_viewport_size({'width':390,'height':844})

        def nav(name):
            page.get_by_role('button', name=name, exact=True).click()

        def assert_stopped():
            assert page.evaluate('window.__qaStreams.every(s=>s.getTracks().every(t=>t.readyState==="ended"))')
            results.append('camera tracks stopped after leaving camera')

        suffix = secrets.token_hex(4)
        password = secrets.token_urlsafe(20)
        admin_code, employee_code = 'qa-admin-'+suffix, 'qa-employee-'+suffix
        snapshot('setup')
        page.get_by_label('ชื่อผู้ดูแล', exact=True).fill('QA Admin')
        page.get_by_label('รหัสพนักงาน', exact=True).fill(admin_code)
        page.get_by_label('รหัสผ่าน', exact=True).fill(password)
        page.get_by_role('button', name='สร้างบัญชีและเริ่มใช้งาน').click()
        expect(page.get_by_role('heading', name='สวัสดี QA')).to_be_visible()
        results.append('first admin setup through same-origin browser proxy')
        snapshot('admin-home')
        nav('ประวัติ')
        expect(page.get_by_role('heading', name='ยังไม่มีประวัติลงเวลา')).to_be_visible()
        snapshot('empty-history')
        nav('พนักงาน')
        page.get_by_role('button', name='เพิ่มพนักงาน', exact=True).click()
        snapshot('employee-create')
        page.get_by_label('ชื่อพนักงาน', exact=True).fill('QA Employee')
        page.get_by_label('รหัสพนักงาน', exact=True).fill(employee_code)
        page.get_by_label('รหัสผ่านเริ่มต้น', exact=True).fill(password)
        page.get_by_role('button', name='สร้างบัญชี', exact=True).click()
        row = page.locator('.employee-row').filter(has=page.get_by_role('heading', name='QA Employee', exact=True))
        expect(row).to_be_visible()
        results.append('admin employee creation')
        row.get_by_role('button', name='เพิ่มใบหน้า').click()
        page.locator('input[type=file]').set_input_files(ROOT / 'face1.jpg')
        expect(page.get_by_role('button', name='ยืนยันลงทะเบียน')).to_be_visible()
        snapshot('enrollment-review')
        page.get_by_role('button', name='ยืนยันลงทะเบียน').click()
        expect(row.get_by_text('มีใบหน้าแล้ว', exact=True)).to_be_visible()
        results.append('admin photo enrollment with real backend inference')
        row.get_by_role('button', name='ล้างใบหน้า', exact=True).click()
        row.get_by_role('button', name='ยืนยัน', exact=True).click()
        expect(row.get_by_text('ยังไม่มีใบหน้า', exact=True)).to_be_visible()
        row.get_by_role('button', name='ระงับบัญชี', exact=True).click()
        row.get_by_role('button', name='ยืนยัน', exact=True).click()
        expect(row.get_by_text('ระงับบัญชี', exact=True)).to_be_visible()
        row.get_by_role('button', name='เปิดใช้งาน', exact=True).click()
        expect(row.get_by_text('ใช้งานปกติ', exact=True)).to_be_visible()
        results.append('admin reset, disable confirmation, and re-enable')
        page.get_by_label('ค้นหาพนักงาน').fill('no-such-person')
        expect(page.get_by_role('heading', name='ไม่พบพนักงาน')).to_be_visible()
        snapshot('employee-search-empty')
        nav('บัญชี')
        page.get_by_role('button', name='ออกจากระบบ', exact=True).click()
        expect(page.get_by_role('heading', name='เข้าสู่ระบบ')).to_be_visible()
        snapshot('login')
        page.get_by_label('รหัสพนักงาน', exact=True).fill(employee_code)
        page.get_by_label('รหัสผ่าน', exact=True).fill('incorrect')
        page.get_by_role('button', name='เข้าสู่ระบบ', exact=True).click()
        expect(page.get_by_role('alert')).to_be_visible()
        snapshot('login-error')
        page.get_by_label('รหัสผ่าน', exact=True).fill(password)
        page.get_by_role('button', name='เข้าสู่ระบบ', exact=True).click()
        expect(page.get_by_role('heading', name='สวัสดี QA')).to_be_visible()
        expect(page.get_by_role('button', name='พนักงาน', exact=True)).to_have_count(0)
        results.append('employee login and admin navigation omitted')
        package_script = next(u for u in page.evaluate('performance.getEntriesByType("resource").map(x=>x.name)')
                              if '@mediapipe_tasks-vision' in u and '.js' in u)
        page.evaluate('''async moduleURL=>{
          const {FaceLandmarker}=await import(moduleURL);
          const original=FaceLandmarker.prototype.detectForVideo;window.__qaInferenceCount=0;
          FaceLandmarker.prototype.detectForVideo=function(...args){window.__qaInferenceCount++;return original.apply(this,args);};
        }''',package_script)
        page.evaluate('window.__qaLighting=0')
        page.get_by_role('button', name='ลงทะเบียนใบหน้า', exact=True).click()
        expect(page.get_by_role('button', name='ถ่ายภาพใบหน้า')).to_be_disabled()
        expect(page.get_by_role('heading', name='แสงน้อยเกินไป', exact=True)).to_be_visible(timeout=30000)
        page.wait_for_timeout(1200)
        expect(page.get_by_role('button', name='ถ่ายภาพใบหน้า')).to_be_disabled()
        assert page.evaluate('window.__qaInferenceCount')==0, 'Black frame gate must stop model inference'
        expect(page.get_by_role('button', name='เริ่มตรวจต่อ', exact=True)).to_be_disabled()
        snapshot('camera-black-blocked')
        page.evaluate('window.__qaLighting=.08')
        page.wait_for_timeout(1200)
        expect(page.get_by_role('button', name='ถ่ายภาพใบหน้า')).to_be_disabled()
        expect(page.get_by_role('heading', name='แสงน้อยเกินไป', exact=True)).to_be_visible()
        assert page.evaluate('window.__qaInferenceCount')==0, 'Dim frame gate must stop model inference'
        expect(page.get_by_role('button', name='เริ่มตรวจต่อ', exact=True)).to_be_disabled()
        snapshot('camera-dim-blocked')
        page.evaluate('window.__qaLighting=1')
        expect(page.get_by_role('button', name='เริ่มตรวจต่อ', exact=True)).to_be_enabled(timeout=30000)
        page.wait_for_timeout(800)
        assert page.evaluate('window.__qaInferenceCount')==0, 'Bright recovery must wait for explicit resume'
        expect(page.get_by_role('button', name='ถ่ายภาพใบหน้า')).to_be_disabled()
        snapshot('camera-bright-awaiting-resume')
        page.get_by_role('button', name='เริ่มตรวจต่อ', exact=True).click()
        expect(page.get_by_role('button', name='ถ่ายภาพใบหน้า')).to_be_enabled(timeout=30000)
        assert page.evaluate('window.__qaInferenceCount')>0
        results.append('SIMULATED black and dim camera: Thai warning, zero model inference, enrollment blocked; bright recovery keeps inference paused until manual resume, then capture works')
        snapshot('camera-enrollment')
        results.append('SIMULATED static camera: actual MediaPipe loads and permits frontal enrollment')
        page.get_by_role('button', name='ถ่ายภาพใบหน้า').click()
        expect(page.get_by_role('button', name='ยืนยันลงทะเบียน')).to_be_visible()
        assert_stopped()
        page.get_by_role('button', name='ยืนยันลงทะเบียน').click()
        expect(page.get_by_role('button', name='ลงเวลาเข้างาน')).to_be_enabled(timeout=30000)
        results.append('SIMULATED camera snapshot enrolls through actual backend SFace')
        page.get_by_role('button', name='ลงเวลาเข้างาน').click()
        page.wait_for_timeout(6000)
        expect(page.get_by_role('button', name='ยืนยันเข้างาน')).to_have_count(0)
        assert context.request.get(url + '/api/attendance').json()['events'] == []
        snapshot('camera-static-challenge')
        results.append('SIMULATED static-photo camera cannot complete motion challenge or record attendance')
        page.get_by_role('button', name='ยกเลิก', exact=True).click()
        assert_stopped()

        # TEST ONLY: dynamically replace browser inference output with motion frames.
        # The production randomized challenge reducer, UI, snapshot, and real server
        # inference remain active. No successful challenge state is assigned directly.
        package_script = next(u for u in page.evaluate('performance.getEntriesByType("resource").map(x=>x.name)')
                              if '@mediapipe_tasks-vision' in u and '.js' in u)
        page.evaluate('''async moduleURL=>{
          const {FaceLandmarker}=await import(moduleURL);
          FaceLandmarker.prototype.detectForVideo=function(video,at){
            window.__qaInferenceCount++;
            const text=document.querySelector('.camera-instructions h2')?.textContent??'';
            let yaw=0,pitch=0,smile=.05,blink=.05;
            if(window.__qaMotionEnabled && text.includes('ซ้ายของคุณ'))yaw=25;
            if(window.__qaMotionEnabled && text.includes('ขวาของคุณ'))yaw=-25;
            if(window.__qaMotionEnabled && text.includes('ก้มหน้า'))pitch=20;
            if(window.__qaMotionEnabled && text.includes('เงยหน้า'))pitch=-20;
            if(window.__qaMotionEnabled && text.includes('ยิ้มให้กล้อง'))smile=.85;
            if(window.__qaMotionEnabled && text.includes('หลับตาสั้น'))blink=.85;
            const y=yaw*Math.PI/180,x=pitch*Math.PI/180;
            const matrix=[Math.cos(y),0,-Math.sin(y),0,0,Math.cos(x),Math.sin(x),0,
              Math.sin(y),-Math.sin(x),Math.cos(y)*Math.cos(x),0,0,0,-55,1];
            return {faceLandmarks:[Array.from({length:478},()=>({x:.5,y:.5,z:0}))],
              facialTransformationMatrixes:[{rows:4,columns:4,data:matrix}],
              faceBlendshapes:[{categories:[{categoryName:'mouthSmileLeft',score:smile},
                {categoryName:'mouthSmileRight',score:smile},{categoryName:'eyeBlinkLeft',score:blink},
                {categoryName:'eyeBlinkRight',score:blink}]}]};
          };
        }''', package_script)
        page.get_by_role('button', name='ลงเวลาเข้างาน').click()
        expect(page.locator('.step-track')).to_have_attribute('aria-label','ขั้นตอน 2 จาก 3',timeout=30000)
        page.evaluate('window.__qaMotionEnabled=false;window.__qaLighting=.08')
        expect(page.get_by_role('heading', name='แสงน้อยเกินไป', exact=True)).to_be_visible()
        page.wait_for_timeout(300)
        inference_paused=page.evaluate('window.__qaInferenceCount')
        page.wait_for_timeout(700)
        assert page.evaluate('window.__qaInferenceCount')==inference_paused
        expect(page.get_by_role('button', name='ยืนยันเข้างาน')).to_have_count(0)
        assert context.request.get(url+'/api/attendance').json()['events']==[]
        snapshot('camera-dim-challenge-paused')
        expect(page.get_by_role('button', name='เริ่มตรวจต่อ', exact=True)).to_be_disabled()
        page.evaluate('window.__qaLighting=1')
        expect(page.get_by_role('button', name='เริ่มตรวจต่อ', exact=True)).to_be_enabled(timeout=10000)
        page.wait_for_timeout(800)
        assert page.evaluate('window.__qaInferenceCount')==inference_paused
        expect(page.get_by_role('button', name='ยืนยันเข้างาน')).to_have_count(0)
        snapshot('camera-bright-challenge-awaiting-resume')
        page.get_by_role('button', name='เริ่มตรวจต่อ', exact=True).click()
        expect(page.locator('.step-track')).to_have_attribute('aria-label','ขั้นตอน 1 จาก 3',timeout=10000)
        page.wait_for_timeout(1600)
        expect(page.locator('.step-track')).to_have_attribute('aria-label','ขั้นตอน 1 จาก 3')
        results.append('SIMULATED dim video interrupts step2; bright recovery remains paused until manual resume, then restarts challenge at step1')
        page.evaluate('window.__qaMotionEnabled=true')
        expect(page.get_by_role('button', name='ยืนยันเข้างาน')).to_be_visible(timeout=30000)
        assert_stopped()
        snapshot('simulated-motion-review')
        accepted=[]
        def lose_accepted_response(route):
            response=route.fetch();assert response.status==201
            accepted.append(response.json()['event']);route.abort()
        page.route('**/api/attendance',lose_accepted_response)
        page.get_by_role('button', name='ยืนยันเข้างาน').click()
        expect(page.get_by_role('alert')).to_be_visible()
        assert len(accepted)==1
        recorded=context.request.get(url+'/api/attendance').json()['events']
        assert len(recorded)==1 and recorded[0]['id']==accepted[0]['id']
        page.unroute('**/api/attendance',lose_accepted_response)
        with page.expect_response(lambda response: response.url==url+'/api/attendance' and response.request.method=='POST') as retried:
            page.get_by_role('button', name='ยืนยันเข้างาน').click()
        assert retried.value.status==200 and retried.value.json()['duplicate']
        assert retried.value.json()['event']['request_id']==accepted[0]['request_id']
        results.append('browser lost accepted response retry preserves request ID and creates no duplicate event')
        expect(page.get_by_role('button', name='ลงเวลาออกงาน')).to_be_enabled(timeout=30000)
        events = context.request.get(url+'/api/attendance').json()['events']
        assert len(events)==1 and events[0]['action']=='check_in'
        results.append('EXPLICIT SYNTHETIC MOTION: randomized three-step UI captured photo; actual SFace recorded check-in')
        page.get_by_role('button', name='ลงเวลาออกงาน').click()
        expect(page.get_by_role('button', name='ยืนยันออกงาน')).to_be_visible(timeout=30000)
        page.get_by_role('button', name='ยืนยันออกงาน').click()
        expect(page.get_by_role('button', name='ลงเวลาเข้างาน')).to_be_enabled(timeout=30000)
        results.append('EXPLICIT SYNTHETIC MOTION: actual SFace recorded check-out and next state')
        nav('ประวัติ')
        expect(page.locator('.event-row')).to_have_count(2)
        snapshot('populated-history')
        nav('บัญชี'); snapshot('employee-account')

        # Explicit denied browser permission with the current isolated account cookies.
        denied_browser=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox','--use-fake-device-for-media-stream'])
        denied_context=denied_browser.new_context(viewport={'width':390,'height':844},locale='th-TH')
        denied_context.add_cookies(context.cookies())
        denied_page=denied_context.new_page()
        cdp=denied_context.new_cdp_session(denied_page)
        context_id=cdp.send('Target.getTargetInfo')['targetInfo']['browserContextId']
        cdp.send('Browser.setPermission',{'permission':{'name':'camera'},'setting':'denied','origin':url,'browserContextId':context_id})
        denied_page.goto(url)
        denied_page.get_by_role('button',name='ลงเวลาเข้างาน').click()
        expect(denied_page.get_by_role('alert')).to_contain_text('อนุญาตให้ใช้กล้อง')
        denied_page.screenshot(path=str(artifacts/'camera-permission-denied-390.png'),full_page=True)
        results.append('actual Chromium denied camera permission displays helpful retry state')
        denied_browser.close()

        # Exercise a frontend network error and its retry affordance.
        page.route('**/api/attendance?limit=100',lambda route:route.abort())
        nav('ประวัติ')
        expect(page.get_by_role('alert')).to_be_visible()
        snapshot('history-network-error')
        page.unroute('**/api/attendance?limit=100')
        page.get_by_role('button',name='โหลดประวัติใหม่').click()
        expect(page.locator('.event-row')).to_have_count(2)
        results.append('history network error and successful retry')
        nav('บัญชี');page.get_by_role('button',name='ออกจากระบบ',exact=True).click()
        expect(page.get_by_role('heading',name='เข้าสู่ระบบ')).to_be_visible()
        page.reload();expect(page.get_by_role('heading',name='เข้าสู่ระบบ')).to_be_visible()
        results.append('browser logout and reload preserve unauthenticated state')
        assert not console_errors, console_errors
        browser.close()
        return {'passed':len(results),'checks':results,'layout_checks':layouts,'small_touch_targets':small_targets,
                'uncaught_browser_errors':console_errors,'artifacts':str(artifacts),
                'limitations':'Fake static video camera plus explicit test-only synthetic motion. No physical phone or live-person camera validation.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url',required=True)
    parser.add_argument('--camera-file',type=Path,required=True)
    parser.add_argument('--artifacts',type=Path,default=Path('/tmp/attendance-ui-qa'))
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    result=run(args.url,args.camera_file,args.artifacts)
    output=json.dumps(result,indent=2);print(output)
    if args.report:args.report.write_text(output+'\n')
