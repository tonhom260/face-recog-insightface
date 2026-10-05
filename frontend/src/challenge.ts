/** Pure, monotonic state machine. Measurements use unmirrored camera coordinates.
 * A positive yaw points the face toward image-right, the person's own left.
 * CSS mirrors only the preview: it never reverses measurements or the saved image.
 */
export const ACTIONS = ['turn_left', 'turn_right', 'look_down', 'look_up', 'smile', 'blink'] as const;
export type ChallengeAction = typeof ACTIONS[number];
export const ACTION_COPY: Record<ChallengeAction, { title: string; hint: string }> = {
  turn_left: { title: 'หันหน้าไปทางซ้ายของคุณ', hint: 'หันช้า ๆ แล้วกลับมามองตรง' },
  turn_right: { title: 'หันหน้าไปทางขวาของคุณ', hint: 'หันช้า ๆ แล้วกลับมามองตรง' },
  look_down: { title: 'ก้มหน้าเล็กน้อย', hint: 'ก้มช้า ๆ แล้วกลับมามองตรง' },
  look_up: { title: 'เงยหน้าเล็กน้อย', hint: 'เงยช้า ๆ แล้วกลับมามองตรง' },
  smile: { title: 'ยิ้มให้กล้อง', hint: 'ยิ้มให้เห็นชัด แล้วกลับมาทำหน้าปกติ' },
  blink: { title: 'หลับตาสั้น ๆ แล้วลืมตา', hint: 'หลับตาค้างสักครู่ แล้วมองตรง' },
};
export type Measurement = { at: number; faces: number; yaw: number; pitch: number; smile: number; blink: number; centered: boolean };
type Baseline = { yaw: number; pitch: number; smile: number; blink: number };
export type ChallengeState = {
  actions: ChallengeAction[]; index: number; phase: 'calibrate' | 'neutral' | 'action' | 'return' | 'passed' | 'timed_out';
  startedAt: number; lastAt: number; holdAt: number | null; frames: number; baseline: Baseline | null;
  samples: Baseline[]; notice: 'face_lost' | 'multiple_faces' | 'center' | null;
};
const DEG = 180 / Math.PI;
export function rotationFromMatrix(data: number[]): { yaw: number; pitch: number } {
  if (data.length !== 16 || data.some(n => !Number.isFinite(n))) return { yaw: NaN, pitch: NaN };
  // MediaPipe matrices are column-major. Normalize out scale before decomposition.
  const length = Math.hypot(data[8], data[9], data[10]);
  if (!length) return { yaw: NaN, pitch: NaN };
  return { yaw: Math.atan2(data[8], data[10]) * DEG, pitch: Math.atan2(-data[9], Math.hypot(data[8], data[10])) * DEG };
}
export function chooseActions(random: () => number = secureRandom): ChallengeAction[] {
  const list = [...ACTIONS];
  for (let i = list.length - 1; i > 0; i--) {
    const value = random();
    if (value < 0 || value >= 1 || !Number.isFinite(value)) throw new Error('Invalid randomness');
    const j = Math.floor(value * (i + 1));
    [list[i], list[j]] = [list[j], list[i]];
  }
  return list.slice(0, 3);
}
function secureRandom() { return crypto.getRandomValues(new Uint32Array(1))[0] / 4294967296; }
export function newChallenge(actions: ChallengeAction[] = chooseActions()): ChallengeState {
  if (actions.length !== 3 || new Set(actions).size !== 3 || actions.some(a => !ACTIONS.includes(a))) throw new Error('Three distinct actions required');
  return { actions: [...actions], index: 0, phase: 'calibrate', startedAt: -1, lastAt: -1, holdAt: null, frames: 0, baseline: null, samples: [], notice: null };
}
export function isNeutral(m: Measurement, baseline: Baseline | null): boolean {
  return m.faces === 1 && m.centered && Number.isFinite(m.yaw) && Number.isFinite(m.pitch)
    && Math.abs(m.yaw - (baseline?.yaw ?? 0)) < 8 && Math.abs(m.pitch - (baseline?.pitch ?? 0)) < 8
    && m.smile < Math.max(.25, (baseline?.smile ?? 0) + .12) && m.blink < .25;
}
function performs(action: ChallengeAction, m: Measurement, base: Baseline): boolean {
  if (!m.centered) return false;
  const yaw = m.yaw - base.yaw, pitch = m.pitch - base.pitch;
  switch (action) {
    case 'turn_left': return yaw > 18 && Math.abs(pitch) < 18;
    case 'turn_right': return yaw < -18 && Math.abs(pitch) < 18;
    case 'look_down': return pitch > 13 && Math.abs(yaw) < 18;
    case 'look_up': return pitch < -13 && Math.abs(yaw) < 18;
    case 'smile': return m.smile > Math.max(.5, base.smile + .3) && Math.abs(yaw) < 12 && Math.abs(pitch) < 12;
    case 'blink': return m.blink > .65 && Math.abs(yaw) < 12 && Math.abs(pitch) < 12;
  }
}
function holding(s: ChallengeState, m: Measurement, predicate: boolean, duration: number, count: number): [ChallengeState, boolean] {
  if (!predicate) return [{ ...s, holdAt: null, frames: 0, samples: [] }, false];
  const next = { ...s, holdAt: s.holdAt ?? m.at, frames: s.frames + 1 };
  return [next, m.at - next.holdAt >= duration && next.frames >= count];
}
export function advanceChallenge(state: ChallengeState, m: Measurement): ChallengeState {
  if (state.phase === 'timed_out' || m.at <= state.lastAt || !Number.isFinite(m.at)) return state;
  let s = { ...state, lastAt: m.at, startedAt: state.startedAt < 0 ? m.at : state.startedAt };
  if (m.faces !== 1 || !Number.isFinite(m.yaw) || !Number.isFinite(m.pitch)) {
    return { ...newChallenge(s.actions), startedAt: m.at, lastAt: m.at, notice: m.faces > 1 ? 'multiple_faces' : 'face_lost' };
  }
  // Discontinuous inference must not count as a sustained pose.
  if (state.lastAt >= 0 && m.at - state.lastAt > 500) return { ...newChallenge(s.actions), startedAt: m.at, lastAt: m.at, notice: null };
  if (state.phase === 'passed') return { ...s, notice: m.centered ? null : 'center' };
  if (m.at - s.startedAt > 15000) return { ...s, phase: 'timed_out', holdAt: null, frames: 0 };
  s.notice = m.centered ? null : 'center';
  if (s.phase === 'calibrate') {
    const [held, done] = holding(s, m, isNeutral(m, null), 600, 8);
    held.samples = held.frames ? [...held.samples, { yaw: m.yaw, pitch: m.pitch, smile: m.smile, blink: m.blink }] : [];
    if (!done) return held;
    const avg = (key: keyof Baseline) => held.samples.reduce((total, v) => total + v[key], 0) / held.samples.length;
    return { ...held, baseline: { yaw: avg('yaw'), pitch: avg('pitch'), smile: avg('smile'), blink: avg('blink') }, phase: 'neutral', startedAt: m.at, holdAt: null, frames: 0, samples: [] };
  }
  if (s.phase === 'neutral' || s.phase === 'return') {
    const [held, done] = holding(s, m, isNeutral(m, s.baseline), 300, 5);
    if (!done) return held;
    if (s.phase === 'neutral') return { ...held, phase: 'action', holdAt: null, frames: 0 };
    const index = s.index + 1;
    return { ...held, index, phase: index === 3 ? 'passed' : 'neutral', startedAt: m.at, holdAt: null, frames: 0 };
  }
  const action = s.actions[s.index];
  const [held, done] = holding(s, m, performs(action, m, s.baseline!), action === 'blink' ? 60 : 240, action === 'blink' ? 2 : 4);
  return done ? { ...held, phase: 'return', holdAt: null, frames: 0 } : held;
}
