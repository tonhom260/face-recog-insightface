import { useEffect, useRef, useState, type FormEvent } from 'react';
import { ArrowLeft, ArrowRight, Camera as CameraIcon, CheckCircle2, Clock3, Home as HomeIcon, Fingerprint, ImagePlus, LogOut, ShieldCheck, UserRound, Users } from 'lucide-react';
import Camera, { type Capture } from './Camera';
import { Account, Alert, Employees, Field, History, Home, Loading, PageHeading } from './pages';
import { ApiError, actionLabel, api, errorMessage, json, type AttendanceAction, type AttendanceEvent, type Employee } from './api';

type View = 'home' | 'history' | 'account' | 'employees';
type Flow = { mode: 'attendance' | 'enroll'; employee: Employee; action?: AttendanceAction; source: 'choice' | 'camera' };
type Submission = Capture & { employeeId: string; mode: 'attendance' | 'enroll'; action?: AttendanceAction; requestId: string };
const nav = [{ view: 'home' as const, title: 'ลงเวลา', icon: HomeIcon }, { view: 'history' as const, title: 'ประวัติ', icon: Clock3 }, { view: 'employees' as const, title: 'พนักงาน', icon: Users }, { view: 'account' as const, title: 'บัญชี', icon: UserRound }];
function Brand({ light = false }: { light?: boolean }) { return <div className={`brand ${light ? 'brand-light' : ''}`}><span className="brand-symbol"><Fingerprint size={26} /></span><div><strong>พร้อม<span>●</span></strong><small>เวลาเริ่มต้นที่ดี</small></div></div>; }

export default function App() {
  const [booting, setBooting] = useState(true), [bootError, setBootError] = useState(''), [needsSetup, setNeedsSetup] = useState(false), [employee, setEmployee] = useState<Employee | null>(null);
  const [view, setView] = useState<View>('home'), [flow, setFlow] = useState<Flow | null>(null), [submission, setSubmission] = useState<Submission | null>(null), [busy, setBusy] = useState(false), [error, setError] = useState(''), [notice, setNotice] = useState(''), [success, setSuccess] = useState<AttendanceEvent | null>(null), [refresh, setRefresh] = useState(0);
  const epoch = useRef(0); const busyRef = useRef(false);
  async function bootstrap() {
    const current = ++epoch.current; setBooting(true); setBootError('');
    try {
      const setup = await api<{ needs_setup: boolean }>('/setup'); if (epoch.current !== current) return;
      setNeedsSetup(setup.needs_setup);
      if (!setup.needs_setup) { try { const me = await api<{ employee: Employee }>('/auth/me'); if (epoch.current === current) setEmployee(me.employee); } catch(e) { if (!(e instanceof ApiError && e.status === 401)) throw e; } }
    } catch(e) { if (epoch.current === current) setBootError(errorMessage(e)); }
    finally { if (epoch.current === current) setBooting(false); }
  }
  useEffect(() => { void bootstrap(); return () => { epoch.current++; }; }, []);
  useEffect(() => {
    const expired = () => { epoch.current++; setEmployee(null); setFlow(null); setSubmission(null); setBusy(false); busyRef.current = false; setView('home'); setError(''); setNotice('เซสชันหมดอายุ กรุณาเข้าสู่ระบบอีกครั้ง'); };
    window.addEventListener('attendance:session-expired', expired);
    return () => window.removeEventListener('attendance:session-expired', expired);
  }, []);
  function signedIn(person: Employee) { epoch.current++; setEmployee(person); setNeedsSetup(false); setFlow(null); setSubmission(null); setError(''); setNotice(''); setSuccess(null); setView('home'); }
  function navigate(next: View) { if (busyRef.current) return; setView(next); setFlow(null); setSubmission(null); setError(''); window.scrollTo({ top: 0 }); }
  function enroll(person = employee!) { setError(''); setSubmission(null); setFlow({ mode: 'enroll', employee: person, source: person.id !== employee!.id ? 'choice' : 'camera' }); }
  function capture(result: Capture) {
    if (!flow || !employee) return;
    setSubmission({ ...result, mode: flow.mode, employeeId: flow.employee.id, action: flow.action, requestId: crypto.randomUUID() }); setError('');
  }
  async function submit() {
    if (!submission || !employee || busyRef.current) return;
    const current = epoch.current, payload = submission, owner = employee.id; busyRef.current = true; setBusy(true); setError('');
    try {
      const body = new FormData(); body.set('image', payload.image, 'face.jpg');
      if (payload.mode === 'attendance') {
        body.set('action', payload.action!); body.set('client_challenge_passed', 'true'); body.set('challenge_actions', JSON.stringify(payload.actions)); body.set('request_id', payload.requestId);
        const response = await api<{ event: AttendanceEvent; duplicate: boolean }>('/attendance', { method: 'POST', body });
        if (epoch.current !== current) return;
        setSuccess(response.event); setRefresh(r => r + 1); setView('home'); setFlow(null); setSubmission(null);
      } else {
        if (payload.employeeId !== owner) body.set('employee_id', payload.employeeId);
        const response = await api<{ employee: Employee; model_version: string }>('/face/enroll', { method: 'POST', body });
        if (epoch.current !== current) return;
        if (response.employee.id === owner) setEmployee(response.employee);
        setNotice(`ลงทะเบียนใบหน้าของ ${response.employee.name} แล้ว`); setView(payload.employeeId === owner ? 'home' : 'employees'); setRefresh(r => r + 1); setFlow(null); setSubmission(null);
      }
    } catch(e) {
      if (epoch.current !== current) return;
      setError(errorMessage(e));
      // An administrator can reset enrollment in another session. Reconcile the
      // authenticated employee rather than leaving a stale enrolled-only UI.
      if (e instanceof ApiError && (e.code === 'FACE_NOT_ENROLLED' || e.code === 'ENROLLMENT_EXISTS')) {
        try {
          const me = await api<{ employee: Employee }>('/auth/me');
          if (epoch.current === current) {
            setEmployee(me.employee); setFlow(null); setSubmission(null); setView('home'); setRefresh(r => r + 1);
          }
        } catch { /* Keep the original helpful error; the user can retry or leave. */ }
      }
    }
    finally { if (epoch.current === current) { busyRef.current = false; setBusy(false); } }
  }
  async function logout() {
    if (busyRef.current) return; const current = epoch.current; busyRef.current = true; setBusy(true); setError('');
    // Cancel camera immediately while the cookie logout request completes.
    setFlow(null); setSubmission(null);
    try { await api<void>('/auth/logout', { method: 'POST' }); if (epoch.current === current) { epoch.current++; setEmployee(null); setSuccess(null); setNotice(''); setView('home'); } }
    catch(e) { if (epoch.current === current) setError(errorMessage(e)); }
    finally { busyRef.current = false; setBusy(false); }
  }
  if (booting) return <div className="boot-screen"><Brand /><Loading /></div>;
  if (bootError) return <div className="boot-screen"><Brand /><Alert>{bootError}</Alert><button className="button primary" onClick={() => void bootstrap()}>ลองอีกครั้ง</button></div>;
  if (!employee) return <Auth key={needsSetup ? 'setup' : 'login'} setup={needsSetup} onLogin={signedIn} onSetupCompleted={() => { setNeedsSetup(false); setNotice('ตั้งค่าระบบแล้ว กรุณาเข้าสู่ระบบ'); }} notice={notice} />;
  const visibleNav = nav.filter(item => item.view !== 'employees' || employee.role === 'admin');
  const renderedEpoch = epoch.current;
  return <div className={`app-shell ${flow ? 'flow-open' : ''}`}>
    <aside className="sidebar"><Brand /><p className="sidebar-caption">พื้นที่ทำงานของคุณ</p><nav aria-label="เมนูหลัก">{visibleNav.map(({ view: next, title, icon: Icon }) => <button key={next} className={view === next ? 'active' : ''} onClick={() => navigate(next)} aria-current={view === next ? 'page' : undefined} disabled={busy}><Icon size={21} />{title}{view === next && <span />}</button>)}</nav><div className="sidebar-bottom"><div className="user-summary"><div className="avatar">{employee.name.slice(0,1)}</div><div><strong>{employee.name}</strong><small>{employee.role === 'admin' ? 'ผู้ดูแลระบบ' : employee.employee_code}</small></div></div><button className="button text" onClick={() => void logout()} disabled={busy}><LogOut size={18} />ออกจากระบบ</button></div></aside>
    <header className="mobile-header"><Brand /><button className="avatar-button" onClick={() => navigate('account')} aria-label="เปิดบัญชีของคุณ">{employee.name.slice(0,1)}</button></header>
    <main className={`main-content ${flow ? 'flow-active' : ''}`} key={employee.id}>
      {submission ? <CaptureReview submission={submission} employee={flow!.employee} busy={busy} error={error} onSubmit={() => void submit()} onRetake={() => { setSubmission(null); setError(''); if (flow) setFlow({ ...flow, source: 'camera' }); }} onCancel={() => { setFlow(null); setSubmission(null); setError(''); }} />
      : flow ? flow.source === 'choice' ? <EnrollmentChoice employee={flow.employee} onCamera={() => setFlow({ ...flow, source: 'camera' })} onUpload={image => capture({ image, actions: [] })} onCancel={() => setFlow(null)} /> : <Camera key={`${employee.id}-${flow.employee.id}-${flow.mode}`} mode={flow.mode} onCapture={capture} onCancel={() => setFlow(null)} />
      : <>{error && view !== 'account' && <Alert>{error}</Alert>}{notice && <div className="dismissible-notice"><Alert kind="success">{notice}</Alert><button className="text-link" onClick={() => setNotice('')}>ปิด</button></div>}{view === 'home' ? <Home employee={employee} refresh={refresh} success={success} onCamera={action => { setSuccess(null); setError(''); setFlow({ mode: 'attendance', employee, action, source: 'camera' }); }} onEnroll={() => enroll()} onHistory={() => navigate('history')} /> : view === 'history' ? <History employee={employee} /> : view === 'employees' && employee.role === 'admin' ? <Employees key={refresh} selfId={employee.id} onEnroll={enroll} onSelfChange={updated => { if (epoch.current === renderedEpoch && updated.id === employee.id) setEmployee(updated); }} /> : <Account employee={employee} onEnroll={() => enroll()} onLogout={() => void logout()} logoutBusy={busy} error={error} />}</>}
    </main>
    <nav className="bottom-nav" aria-label="เมนูหลักบนมือถือ">{visibleNav.map(({ view: next, title, icon: Icon }) => <button key={next} className={view === next ? 'active' : ''} onClick={() => navigate(next)} disabled={busy} aria-current={view === next ? 'page' : undefined}><Icon size={21} /><span>{title}</span></button>)}</nav>
  </div>;
}

function Auth({ setup, onLogin, onSetupCompleted, notice }: { setup: boolean; onLogin: (employee: Employee) => void; onSetupCompleted: () => void; notice: string }) {
  const [busy, setBusy] = useState(false), [error, setError] = useState(''); const mounted = useRef(true);
  useEffect(() => { mounted.current = true; return () => { mounted.current = false; }; }, []);
  async function handle(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); if (busy) return; setBusy(true); setError(''); const values = new FormData(event.currentTarget);
    try { const result = await api<{ employee: Employee }>(setup ? '/setup' : '/auth/login', json({ employee_code: String(values.get('employee_code')).trim(), password: values.get('password'), ...(setup ? { name: String(values.get('name')).trim() } : {}) })); if (mounted.current) onLogin(result.employee); }
    catch(e) { if (mounted.current) { if (setup && e instanceof ApiError && e.status === 409) onSetupCompleted(); else setError(errorMessage(e)); } }
    finally { if (mounted.current) setBusy(false); }
  }
  return <div className="auth-shell"><div className="auth-art"><Brand light /><div className="auth-art-content"><span className="auth-kicker">วันดี ๆ เริ่มต้นที่ความพร้อม</span><h1>ทุกการเริ่มต้น<br />มีเวลาของมัน<span>●</span></h1><p>ลงเวลาได้ง่ายด้วยใบหน้าของคุณ<br />เพื่อให้คุณโฟกัสกับงานที่สำคัญ</p><div className="auth-orbits" aria-hidden="true"><Fingerprint size={100} strokeWidth={1} /><span /><span /></div></div><small className="auth-art-footer">พร้อม · ระบบลงเวลาสำหรับทีม</small></div><main className="auth-main"><div className="auth-mobile-brand"><Brand /></div><div className="auth-form"><p className="eyebrow">{setup ? 'เริ่มต้นระบบสำหรับทีม' : 'ยินดีต้อนรับกลับ'}</p><h1>{setup ? 'สร้างผู้ดูแลคนแรก' : 'เข้าสู่ระบบ'}</h1><p className="auth-description">{setup ? 'ตั้งค่าบัญชีของคุณ แล้วเริ่มเพิ่มพนักงานให้ทีม' : 'ใช้รหัสพนักงานและรหัสผ่านของคุณ'}</p>{notice && <Alert kind="success">{notice}</Alert>}<form onSubmit={handle}>{setup && <Field label="ชื่อผู้ดูแล" name="name" required maxLength={100} autoComplete="name" />}<Field label="รหัสพนักงาน" name="employee_code" required maxLength={50} autoComplete="username" autoCapitalize="none" spellCheck={false} /><Field label="รหัสผ่าน" name="password" type="password" required minLength={setup ? 8 : undefined} maxLength={128} autoComplete={setup ? 'new-password' : 'current-password'} />{setup && <p className="field-hint">ใช้รหัสผ่านอย่างน้อย 8 ตัวอักษร</p>}{error && <Alert>{error}</Alert>}<button className="button primary full" disabled={busy}>{busy ? 'กำลังดำเนินการ…' : setup ? 'สร้างบัญชีและเริ่มใช้งาน' : 'เข้าสู่ระบบ'}<ArrowRight size={19} /></button></form><div className="auth-help"><ShieldCheck size={18} /><p>{setup ? 'บัญชีนี้จะจัดการพนักงานและการลงทะเบียนใบหน้าของทีม' : 'หากเข้าสู่ระบบไม่ได้ กรุณาติดต่อผู้ดูแลทีมของคุณ'}</p></div></div><footer className="auth-footer">ระบบบันทึกเวลาตามเวลาประเทศไทย</footer></main></div>;
}

function EnrollmentChoice({ employee, onCamera, onUpload, onCancel }: { employee: Employee; onCamera: () => void; onUpload: (image: Blob) => void; onCancel: () => void }) {
  const [error, setError] = useState('');
  return <><PageHeading eyebrow="ลงทะเบียนใบหน้าให้พนักงาน" title={employee.name}><button className="icon-button" onClick={onCancel} aria-label="กลับ"><ArrowLeft /></button></PageHeading><section className="choice-panel"><div className="motion-icon"><CameraIcon size={30} /></div><h2>เลือกภาพใบหน้าที่ชัดเจน</h2><p>ให้มีเพียงใบหน้าของพนักงานคนนี้ มองตรง อยู่ในที่สว่าง และไม่ใส่แว่นกันแดด</p><button className="button primary full" onClick={onCamera}><CameraIcon size={19} />เปิดกล้องถ่ายภาพ</button><label className="button outline full upload-button"><ImagePlus size={19} />เลือกภาพ JPEG หรือ PNG<input type="file" accept="image/jpeg,image/png" onChange={e => { const file = e.target.files?.[0]; e.target.value = ''; if (!file) return; if (!['image/jpeg', 'image/png'].includes(file.type) || file.size > 10 * 1024 * 1024) { setError('ใช้ภาพ JPEG หรือ PNG ขนาดไม่เกิน 10 MB'); return; } onUpload(file); }} /></label>{error && <Alert>{error}</Alert>}<p className="privacy-note">ลงทะเบียนด้วยความยินยอมของพนักงาน และตรวจสอบว่าภาพตรงกับบุคคลนี้</p><button className="button text full" onClick={onCancel}>ยกเลิก</button></section></>;
}

function CaptureReview({ submission, employee, busy, error, onSubmit, onRetake, onCancel }: { submission: Submission; employee: Employee; busy: boolean; error: string; onSubmit: () => void; onRetake: () => void; onCancel: () => void }) {
  const [url, setUrl] = useState('');
  useEffect(() => { const next = URL.createObjectURL(submission.image); setUrl(next); return () => URL.revokeObjectURL(next); }, [submission.image]);
  return <><PageHeading eyebrow={submission.mode === 'attendance' ? 'พร้อมบันทึกเวลา' : 'ลงทะเบียนใบหน้า'} title={submission.mode === 'attendance' ? `ยืนยัน${actionLabel(submission.action!)}` : 'ตรวจสอบภาพของคุณ'} /><section className="review-panel"><div className="review-photo">{url && <img src={url} alt={`ภาพใบหน้าของ ${employee.name} ที่จะส่งตรวจสอบ`} />}</div><div className="review-content"><div className="review-check"><CheckCircle2 size={24} /><span>{submission.mode === 'attendance' ? 'ทำท่าทางครบ 3 ขั้นตอนแล้ว' : 'ภาพสำหรับลงทะเบียน'}</span></div><h2>{employee.name}</h2><p>{submission.mode === 'attendance' ? 'ตรวจสอบภาพ แล้วกดยืนยัน ระบบจะเปรียบเทียบใบหน้าและบันทึกเวลาปัจจุบันให้คุณ' : 'ตรวจสอบว่าใบหน้าชัดเจน เมื่อยืนยัน ภาพนี้จะใช้สำหรับเปรียบเทียบในการลงเวลา'}</p>{error && <Alert>{error}</Alert>}<button className="button primary full" onClick={onSubmit} disabled={busy}>{busy ? 'กำลังตรวจสอบและบันทึก…' : submission.mode === 'attendance' ? `ยืนยัน${actionLabel(submission.action!)}` : 'ยืนยันลงทะเบียน'}<ArrowRight size={18} /></button><button className="button outline full" disabled={busy} onClick={onRetake}><CameraIcon size={18} />ถ่ายใหม่</button><button className="button text full" disabled={busy} onClick={onCancel}>ยกเลิก</button><p className="privacy-note">ภาพถูกส่งเพื่อเปรียบเทียบใบหน้า เวลาที่บันทึกเป็นเวลาจากระบบ</p></div></section></>;
}
