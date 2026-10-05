import { useEffect, useId, useState, type FormEvent, type ReactNode } from 'react';
import { ArrowDownLeft, ArrowUpRight, Camera, Check, ChevronRight, Clock3, Fingerprint, LogOut, Plus, RefreshCw, Search, ShieldCheck, UserRound, Users, X } from 'lucide-react';
import { actionLabel, api, dateOf, errorMessage, json, timeOf, type AttendanceEvent, type AttendanceStatus, type Employee } from './api';

export function Alert({ children, kind = 'error' }: { children: ReactNode; kind?: 'error' | 'success' }) { return <div className={`${kind}-banner`} role={kind === 'error' ? 'alert' : 'status'}>{kind === 'success' && <Check size={18} />}{children}</div>; }
export function Field({ label, ...props }: React.InputHTMLAttributes<HTMLInputElement> & { label: string }) { const id = useId(); return <label className="field" htmlFor={id}><span>{label}</span><input id={id} {...props} /></label>; }
export function Loading() { return <div className="page-loading" role="status"><span className="spinner" /><span>กำลังโหลดข้อมูล…</span></div>; }
export function PageHeading({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) { return <div className="section-heading"><div><p className="eyebrow">{eyebrow}</p><h1>{title}</h1></div>{children}</div>; }
export function EventRow({ event }: { event: AttendanceEvent }) { const checkIn = event.action === 'check_in'; return <li className="event-row"><span className={`event-icon ${checkIn ? 'in' : 'out'}`}>{checkIn ? <ArrowDownLeft size={22} /> : <ArrowUpRight size={22} />}</span><div><strong>{actionLabel(event.action)}</strong><span>{dateOf(event.occurred_at)}</span></div><div className="event-time"><strong>{timeOf(event.occurred_at)}</strong><span>เวลาประเทศไทย</span></div></li>; }

export function Home({ employee, refresh, onCamera, onEnroll, onHistory, success }: { employee: Employee; refresh: number; onCamera: (action: 'check_in' | 'check_out') => void; onEnroll: () => void; onHistory: () => void; success: AttendanceEvent | null }) {
  const [status, setStatus] = useState<AttendanceStatus | null>(null);
  const [events, setEvents] = useState<AttendanceEvent[]>([]);
  const [error, setError] = useState('');
  const [reload, setReload] = useState(0);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    const controller = new AbortController(); setLoading(true); setError(''); setStatus(null);
    Promise.all([api<AttendanceStatus>('/attendance/status', { signal: controller.signal }), api<{ events: AttendanceEvent[] }>('/attendance?limit=3', { signal: controller.signal })])
      .then(([s, e]) => { if (!controller.signal.aborted) { setStatus(s); setEvents(e.events); } })
      .catch(e => { if (!controller.signal.aborted) setError(errorMessage(e)); })
      .finally(() => { if (!controller.signal.aborted) setLoading(false); });
    return () => controller.abort();
  }, [employee.id, refresh, reload]);
  const today = new Intl.DateTimeFormat('th-TH', { timeZone: 'Asia/Bangkok', weekday: 'long', day: 'numeric', month: 'long' }).format(new Date());
  return <>
    <PageHeading eyebrow={today} title={`สวัสดี ${employee.name.split(' ')[0]}`}><span className="employee-chip"><UserRound size={15} />{employee.employee_code}</span></PageHeading>
    {success && <Alert kind="success">บันทึก{actionLabel(success.action)}แล้ว เวลา {timeOf(success.occurred_at)} น.</Alert>}
    <div className="home-grid"><section className="attendance-hero">
      <div className="hero-top"><span className="pill"><span className="live-dot" />ลงเวลาประจำวัน</span><Fingerprint size={30} aria-hidden="true" /></div>
      <p className="hero-subtitle">{!employee.face_enrolled ? 'เริ่มต้นใช้งานครั้งแรก' : status?.next_action === 'check_out' ? 'พร้อมสำหรับเวลาพักของคุณ' : 'พร้อมเริ่มวันใหม่ไปด้วยกัน'}</p>
      <h2>{!employee.face_enrolled ? 'ลงทะเบียนใบหน้า' : status?.next_action === 'check_out' ? 'วันนี้คุณเข้างานแล้ว' : 'เริ่มต้นวันทำงาน'}</h2>
      <p>{!employee.face_enrolled ? 'ถ่ายภาพใบหน้าที่ชัดเจนหนึ่งภาพ เพื่อใช้ยืนยันเมื่อคุณลงเวลา' : 'เปิดกล้อง ทำตามท่าทาง 3 ขั้นตอน แล้วให้ระบบบันทึกเวลาให้คุณ'}</p>
      {error ? <><Alert>{error}</Alert><button className="button light full" onClick={() => setReload(n => n + 1)}><RefreshCw size={18} />โหลดสถานะใหม่</button></> : !employee.face_enrolled ? <button className="button light full" onClick={onEnroll}><Camera size={20} />ลงทะเบียนใบหน้า<ChevronRight size={18} /></button> : <button className="button light full" disabled={loading || !status} onClick={() => status && onCamera(status.next_action)}><Camera size={20} />{loading ? 'กำลังโหลดสถานะ…' : `ลงเวลา${actionLabel(status!.next_action)}`}<ChevronRight size={18} /></button>}
      <div className="hero-footnote"><ShieldCheck size={15} />ตรวจสอบใบหน้าก่อนบันทึกทุกครั้ง</div>
    </section><section className="status-panel"><p className="eyebrow">สถานะของคุณ</p><div className="status-mark"><Clock3 size={24} /></div><h2>{loading ? 'กำลังโหลด…' : status?.next_action === 'check_out' ? 'อยู่ระหว่างงาน' : 'ยังไม่ได้เข้างาน'}</h2><p>{status?.last_event ? `รายการล่าสุด · ${actionLabel(status.last_event.action)}` : 'เริ่มลงเวลาเมื่อพร้อมทำงาน'}</p>{status?.last_event && <div className="last-time"><strong>{timeOf(status.last_event.occurred_at)}</strong><span>{dateOf(status.last_event.occurred_at)} · เวลาประเทศไทย</span></div>}<div className={`enrollment-status ${employee.face_enrolled ? 'enrolled' : ''}`}><span className="live-dot" />{employee.face_enrolled ? 'ลงทะเบียนใบหน้าแล้ว' : 'ยังไม่ได้ลงทะเบียนใบหน้า'}</div></section></div>
    <section className="recent-section"><div className="section-title"><h2>รายการล่าสุด</h2><button className="text-link" onClick={onHistory}>ดูทั้งหมด<ChevronRight size={16} /></button></div>{loading ? <Loading /> : events.length ? <ul className="event-list">{events.map(event => <EventRow key={event.id} event={event} />)}</ul> : <div className="empty-state compact"><Clock3 size={28} /><h3>รายการแรกของคุณเริ่มได้วันนี้</h3><p>เมื่อคุณลงเวลา รายการจะแสดงที่นี่</p></div>}</section>
  </>;
}

export function History({ employee }: { employee: Employee }) {
  const [events, setEvents] = useState<AttendanceEvent[] | null>(null); const [error, setError] = useState(''); const [reload, setReload] = useState(0);
  useEffect(() => { const controller = new AbortController(); setEvents(null); setError(''); api<{ events: AttendanceEvent[] }>('/attendance?limit=100', { signal: controller.signal }).then(r => { if (!controller.signal.aborted) setEvents(r.events); }).catch(e => { if (!controller.signal.aborted) setError(errorMessage(e)); }); return () => controller.abort(); }, [employee.id, reload]);
  return <><PageHeading eyebrow="บันทึกของคุณ" title="ประวัติลงเวลา"><button className="icon-button" onClick={() => setReload(r => r + 1)} aria-label="โหลดประวัติใหม่"><RefreshCw size={20} /></button></PageHeading><div className="history-note"><Clock3 size={18} />ทุกเวลาแสดงตามเวลาประเทศไทย</div>{error ? <Alert>{error}</Alert> : events === null ? <Loading /> : events.length ? <section className="history-panel"><div className="section-title"><h2>รายการล่าสุด</h2><span className="muted">{events.length} รายการ</span></div><ul className="event-list">{events.map(event => <EventRow key={event.id} event={event} />)}</ul><p className="list-footnote">แสดงสูงสุด 100 รายการล่าสุด</p></section> : <div className="empty-state"><Clock3 size={36} /><h2>ยังไม่มีประวัติลงเวลา</h2><p>รายการเข้าและออกงานของคุณจะแสดงที่นี่</p></div>}</>;
}

export function Account({ employee, onEnroll, onLogout, logoutBusy, error }: { employee: Employee; onEnroll: () => void; onLogout: () => void; logoutBusy: boolean; error: string }) {
  return <><PageHeading eyebrow="ข้อมูลส่วนตัว" title="บัญชีของคุณ" /><section className="profile-panel"><div className="avatar large">{employee.name.slice(0,1)}</div><h2>{employee.name}</h2><p>{employee.role === 'admin' ? 'ผู้ดูแลระบบ' : 'พนักงาน'}</p><dl className="profile-details"><div><dt>รหัสพนักงาน</dt><dd>{employee.employee_code}</dd></div><div><dt>สถานะบัญชี</dt><dd className="teal">ใช้งานปกติ</dd></div><div><dt>ใบหน้า</dt><dd>{employee.face_enrolled ? 'ลงทะเบียนแล้ว' : 'ยังไม่ได้ลงทะเบียน'}</dd></div></dl>{!employee.face_enrolled ? <button className="button primary full" onClick={onEnroll}><Camera size={18} />ลงทะเบียนใบหน้า</button> : <p className="account-note">หากต้องการเปลี่ยนภาพใบหน้า กรุณาติดต่อผู้ดูแลระบบ</p>}{error && <Alert>{error}</Alert>}<button className="button outline full logout-button" disabled={logoutBusy} onClick={onLogout}><LogOut size={18} />{logoutBusy ? 'กำลังออกจากระบบ…' : 'ออกจากระบบ'}</button></section></>;
}

export function Employees({ onEnroll, onSelfChange, selfId }: { onEnroll: (employee: Employee) => void; onSelfChange: (employee: Employee) => void; selfId: string }) {
  const [employees, setEmployees] = useState<Employee[] | null>(null); const [error, setError] = useState(''); const [success, setSuccess] = useState(''); const [reload, setReload] = useState(0); const [search, setSearch] = useState('');
  const [creating, setCreating] = useState(false); const [busy, setBusy] = useState(false); const [confirm, setConfirm] = useState<{ id: string; action: 'reset' | 'disable' } | null>(null);
  useEffect(() => { const controller = new AbortController(); setError(''); api<{ employees: Employee[] }>('/employees', { signal: controller.signal }).then(r => { if (!controller.signal.aborted) setEmployees(r.employees); }).catch(e => { if (!controller.signal.aborted) setError(errorMessage(e)); }); return () => controller.abort(); }, [reload]);
  async function create(event: FormEvent<HTMLFormElement>) {
    event.preventDefault(); setBusy(true); setError(''); setSuccess(''); const form = event.currentTarget, values = new FormData(form);
    try { const r = await api<{ employee: Employee }>('/employees', json({ employee_code: String(values.get('employee_code')).trim(), name: String(values.get('name')).trim(), password: values.get('password') })); setEmployees(list => [...(list ?? []), r.employee]); form.reset(); setCreating(false); setSuccess(`เพิ่ม ${r.employee.name} แล้ว ให้พนักงานเข้าสู่ระบบและลงทะเบียนใบหน้า`); }
    catch(e) { setError(errorMessage(e)); } finally { setBusy(false); }
  }
  async function update(employee: Employee, action: 'reset' | 'disable' | 'enable') {
    setBusy(true); setError(''); setSuccess('');
    try {
      let updated: Employee;
      if (action === 'reset') { await api<void>(`/employees/${encodeURIComponent(employee.id)}/face`, { method: 'DELETE' }); updated = { ...employee, face_enrolled: false }; }
      else { const r = await api<{ employee: Employee }>(`/employees/${encodeURIComponent(employee.id)}`, { method: 'PATCH', body: JSON.stringify({ active: action === 'enable' }) }); updated = r.employee; }
      setEmployees(list => list!.map(e => e.id === employee.id ? updated : e)); if (employee.id === selfId) onSelfChange(updated);
      setConfirm(null); setSuccess(action === 'reset' ? `ล้างใบหน้าของ ${employee.name} แล้ว สามารถลงทะเบียนใหม่ได้` : `อัปเดตบัญชี ${employee.name} แล้ว`);
    } catch(e) { setError(errorMessage(e)); } finally { setBusy(false); }
  }
  const visible = employees?.filter(e => `${e.name} ${e.employee_code}`.toLocaleLowerCase().includes(search.toLocaleLowerCase()));
  return <><PageHeading eyebrow="จัดการทีม" title="พนักงาน"><button className="button primary small" onClick={() => setCreating(!creating)}>{creating ? <X size={17} /> : <Plus size={17} />}{creating ? 'ปิดฟอร์ม' : 'เพิ่มพนักงาน'}</button></PageHeading>
    {creating && <section className="create-panel"><div className="section-title"><h2>บัญชีพนักงานใหม่</h2><Users size={22} /></div><form onSubmit={create}><div className="form-grid"><Field label="ชื่อพนักงาน" name="name" required maxLength={100} autoComplete="off" /><Field label="รหัสพนักงาน" name="employee_code" required maxLength={50} autoComplete="off" /><Field label="รหัสผ่านเริ่มต้น" name="password" type="password" required minLength={8} maxLength={128} autoComplete="new-password" /></div><p className="field-hint">รหัสผ่านอย่างน้อย 8 ตัวอักษร ส่งให้พนักงานผ่านช่องทางที่ปลอดภัย</p><button className="button primary" disabled={busy}>{busy ? 'กำลังบันทึก…' : 'สร้างบัญชี'}</button></form></section>}
    {error && <Alert>{error}</Alert>}{success && <Alert kind="success">{success}</Alert>}
    <div className="employee-toolbar"><label className="search-field"><Search size={19} /><input value={search} onChange={e => setSearch(e.target.value)} placeholder="ค้นหาชื่อหรือรหัสพนักงาน" aria-label="ค้นหาพนักงาน" /></label><button className="icon-button" onClick={() => setReload(r => r + 1)} aria-label="โหลดพนักงานใหม่"><RefreshCw size={19} /></button></div>
    {employees === null ? error ? null : <Loading /> : visible?.length ? <div className="employee-list">{visible.map(employee => <article className={`employee-row ${!employee.active ? 'inactive' : ''}`} key={employee.id}><div className="employee-identity"><div className="avatar">{employee.name.slice(0,1)}</div><div><h3>{employee.name}{employee.id === selfId && <span className="self-tag">คุณ</span>}</h3><p>{employee.employee_code} · {employee.role === 'admin' ? 'ผู้ดูแลระบบ' : 'พนักงาน'}</p></div></div><div className="employee-state"><span className={`pill ${employee.active ? 'positive' : 'neutral'}`}>{employee.active ? 'ใช้งานปกติ' : 'ระงับบัญชี'}</span><span className="face-state"><Fingerprint size={15} />{employee.face_enrolled ? 'มีใบหน้าแล้ว' : 'ยังไม่มีใบหน้า'}</span></div><div className="employee-actions">{confirm?.id === employee.id ? <div className="inline-confirm"><p>{confirm.action === 'reset' ? 'ล้างใบหน้าเพื่อให้ลงทะเบียนใหม่?' : 'ระงับการเข้าสู่ระบบบัญชีนี้?'}</p><button className="button danger small" disabled={busy} onClick={() => void update(employee, confirm.action)}>ยืนยัน</button><button className="button outline small" disabled={busy} onClick={() => setConfirm(null)}>ยกเลิก</button></div> : <>{employee.active && <button className="button outline small" disabled={busy} onClick={() => onEnroll(employee)}><Camera size={16} />{employee.face_enrolled ? 'เปลี่ยนใบหน้า' : 'เพิ่มใบหน้า'}</button>}{employee.face_enrolled && <button className="button text small" disabled={busy} onClick={() => setConfirm({ id: employee.id, action: 'reset' })}>ล้างใบหน้า</button>}{employee.id !== selfId && <button className={`button text small ${employee.active ? 'danger-text' : ''}`} disabled={busy} onClick={() => employee.active ? setConfirm({ id: employee.id, action: 'disable' }) : void update(employee, 'enable')}>{employee.active ? 'ระงับบัญชี' : 'เปิดใช้งาน'}</button>}</>}</div></article>)}</div> : <div className="empty-state"><Users size={34} /><h2>ไม่พบพนักงาน</h2><p>ลองค้นหาด้วยชื่อหรือรหัสอื่น</p></div>}
  </>;
}
