export type Employee = { id: string; employee_code: string; name: string; role: 'admin' | 'employee'; active: boolean; face_enrolled: boolean };
export type AttendanceAction = 'check_in' | 'check_out';
export type AttendanceEvent = { id: string; employee_id: string; action: AttendanceAction; occurred_at: string; similarity: number; request_id: string; client_challenge_passed: boolean };
export type AttendanceStatus = { next_action: AttendanceAction; last_event: AttendanceEvent | null };

const messages: Record<string, string> = {
  INVALID_LOGIN: 'รหัสพนักงานหรือรหัสผ่านไม่ถูกต้อง ลองใหม่อีกครั้ง',
  ACCOUNT_INACTIVE: 'บัญชีนี้ถูกระงับ กรุณาติดต่อผู้ดูแล',
  SETUP_COMPLETED: 'มีผู้ดูแลระบบแล้ว กรุณาเข้าสู่ระบบ',
  INVALID_FIELDS: 'ข้อมูลไม่ถูกต้อง กรุณาตรวจสอบแล้วลองอีกครั้ง',
  REQUEST_ID_CONFLICT: 'รายการนี้ไม่สามารถส่งซ้ำได้ กรุณากลับไปดูสถานะล่าสุดแล้วเริ่มใหม่',
  ADMIN_SELF_DISABLE: 'ไม่สามารถระงับบัญชีผู้ดูแลที่กำลังใช้งานอยู่',
  INVALID_ORIGIN: 'ไม่สามารถทำรายการจากหน้านี้ได้ กรุณาเปิดระบบผ่านที่อยู่ของทีมแล้วลองใหม่',
  INVALID_CREDENTIALS: 'รหัสพนักงานหรือรหัสผ่านไม่ถูกต้อง ลองใหม่อีกครั้ง',
  UNAUTHENTICATED: 'กรุณาเข้าสู่ระบบอีกครั้ง', AUTH_REQUIRED: 'กรุณาเข้าสู่ระบบอีกครั้ง',
  ACCOUNT_DISABLED: 'บัญชีนี้ถูกระงับ กรุณาติดต่อผู้ดูแล', EMPLOYEE_INACTIVE: 'บัญชีนี้ถูกระงับ กรุณาติดต่อผู้ดูแล',
  FORBIDDEN: 'คุณไม่มีสิทธิ์ทำรายการนี้', FACE_MISMATCH: 'ใบหน้าไม่ตรงกับที่ลงทะเบียน ลองใหม่ในที่มีแสงสว่าง',
  FACE_NOT_FOUND: 'ไม่พบใบหน้าที่ชัดเจน มองตรงและให้ใบหน้าอยู่ในกรอบ',
  MULTIPLE_FACES: 'พบมากกว่าหนึ่งใบหน้า กรุณาให้มีเฉพาะคุณอยู่ในภาพ',
  FACE_NOT_ENROLLED: 'กรุณาลงทะเบียนใบหน้าก่อนลงเวลา', MODELS_UNAVAILABLE: 'ระบบตรวจสอบใบหน้ายังไม่พร้อม กรุณาลองใหม่ภายหลัง',
  INVALID_TRANSITION: 'สถานะลงเวลาเปลี่ยนแล้ว กรุณากลับไปดูสถานะล่าสุด',
  ENROLLMENT_EXISTS: 'ลงทะเบียนใบหน้าแล้ว หากต้องการเปลี่ยน กรุณาติดต่อผู้ดูแล',
  FACE_ALREADY_ENROLLED: 'ลงทะเบียนใบหน้าแล้ว หากต้องการเปลี่ยน กรุณาติดต่อผู้ดูแล',
  DUPLICATE_EMPLOYEE_CODE: 'รหัสพนักงานนี้มีอยู่แล้ว กรุณาใช้รหัสอื่น',
  SETUP_COMPLETE: 'มีผู้ดูแลระบบแล้ว กรุณาเข้าสู่ระบบ', SETUP_ALREADY_COMPLETED: 'มีผู้ดูแลระบบแล้ว กรุณาเข้าสู่ระบบ',
  INVALID_IMAGE: 'ไม่สามารถอ่านภาพนี้ได้ กรุณาใช้ภาพ JPEG หรือ PNG ที่ชัดเจน',
  INVALID_REQUEST: 'ข้อมูลไม่ถูกต้อง กรุณาตรวจสอบแล้วลองอีกครั้ง',
  INVALID_CHALLENGE: 'ขั้นตอนตรวจสอบไม่สมบูรณ์ กรุณาเริ่มใหม่',
  EMPLOYEE_NOT_FOUND: 'ไม่พบบัญชีพนักงานนี้ กรุณาโหลดข้อมูลใหม่',
};

export class ApiError extends Error {
  constructor(public status: number, public code: string) {
    super(messages[code] ?? (status === 401 ? 'กรุณาเข้าสู่ระบบอีกครั้ง' : status === 403 ? 'บัญชีนี้ไม่สามารถทำรายการได้ กรุณาติดต่อผู้ดูแล' : status === 422 ? 'ข้อมูลหรือภาพไม่ถูกต้อง กรุณาตรวจสอบแล้วลองใหม่' : status === 503 ? 'ระบบยังไม่พร้อม กรุณาลองใหม่ภายหลัง' : 'ทำรายการไม่สำเร็จ กรุณาลองอีกครั้ง'));
  }
}
export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api${path}`, { ...init, credentials: 'same-origin', headers: { ...(init.body instanceof FormData ? {} : init.body ? { 'Content-Type': 'application/json' } : {}), ...init.headers } });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') throw error;
    throw new Error('เชื่อมต่อไม่ได้ ตรวจสอบอินเทอร์เน็ตแล้วลองใหม่');
  }
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    if (response.status === 401 && path !== '/auth/login' && path !== '/auth/me') window.dispatchEvent(new Event('attendance:session-expired'));
    throw new ApiError(response.status, body.error?.code ?? 'UNKNOWN');
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}
export const json = (body: unknown): RequestInit => ({ method: 'POST', body: JSON.stringify(body) });
export const errorMessage = (error: unknown) => error instanceof Error ? error.message : 'เกิดข้อผิดพลาด กรุณาลองใหม่';
export const timeOf = (value: string) => new Intl.DateTimeFormat('th-TH', { timeZone: 'Asia/Bangkok', hour: '2-digit', minute: '2-digit' }).format(new Date(value));
export const dateOf = (value: string) => new Intl.DateTimeFormat('th-TH', { timeZone: 'Asia/Bangkok', day: 'numeric', month: 'short', year: 'numeric' }).format(new Date(value));
export const actionLabel = (action: AttendanceAction) => action === 'check_in' ? 'เข้างาน' : 'ออกงาน';
