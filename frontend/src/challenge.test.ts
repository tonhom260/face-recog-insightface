import { describe, it, expect } from 'vitest';
import { advanceChallenge, chooseActions, newChallenge, rotationFromMatrix, type ChallengeState, type Measurement } from './challenge';

const actions = ['turn_left', 'smile', 'blink'] as const;
function frame(at: number, values: Partial<Measurement> = {}): Measurement { return { at, faces: 1, yaw: 0, pitch: 0, smile: .05, blink: .05, centered: true, ...values }; }
function feed(s: ChallengeState, start: number, duration: number, values: Partial<Measurement> = {}) {
  for (let at = start; at <= start + duration; at += 50) s = advanceChallenge(s, frame(at, values));
  return s;
}
function ready() { return feed(newChallenge([...actions]), 0, 1050); }
describe('local motion challenge', () => {
  it('chooses exactly three unique supported actions', () => {
    for (const r of [0, .2, .5, .9999]) expect(new Set(chooseActions(() => r)).size).toBe(3);
    expect(() => chooseActions(() => 1)).toThrow();
    expect(() => newChallenge(['blink', 'blink', 'smile'])).toThrow();
  });
  it('requires neutral calibration and sustained neutral → action → neutral', () => {
    let s = ready(); expect(s.phase).toBe('action');
    s = feed(s, 1100, 350, { yaw: 24 }); expect(s.phase).toBe('return'); expect(s.index).toBe(0);
    s = feed(s, 1500, 350); expect(s.index).toBe(1); expect(s.phase).toBe('neutral');
    s = feed(s, 1900, 350); s = feed(s, 2300, 300, { smile: .8 }); expect(s.phase).toBe('return');
    s = feed(s, 2650, 350); s = feed(s, 3050, 350); s = feed(s, 3450, 100, { blink: .85 });
    expect(s.phase).toBe('return'); s = feed(s, 3600, 350); expect(s.phase).toBe('passed'); expect(s.index).toBe(3);
  });
  it('cannot skip neutral, accept the wrong direction, or count a single frame', () => {
    let s = feed(newChallenge([...actions]), 0, 1000, { yaw: 25 }); expect(s.phase).toBe('calibrate');
    s = ready(); s = feed(s, 1100, 500, { yaw: -25 }); expect(s.phase).toBe('action');
    s = advanceChallenge(s, frame(1650, { yaw: 25 })); s = advanceChallenge(s, frame(1700)); expect(s.phase).toBe('action');
  });
  it.each([
    ['turn_left', { yaw: 24 }], ['turn_right', { yaw: -24 }],
    ['look_down', { pitch: 18 }], ['look_up', { pitch: -18 }],
    ['smile', { smile: .8 }], ['blink', { blink: .85 }],
  ] as const)('recognizes %s only after a sustained pose and a return to neutral', (action, values) => {
    const other = actions.filter(a => a !== action);
    let s = feed(newChallenge([action, ...other.slice(0, 2)]), 0, 1050);
    s = feed(s, 1100, 300, values);
    expect(s.phase).toBe('return'); expect(s.index).toBe(0);
    s = feed(s, 1450, 350); expect(s.index).toBe(1);
  });
  it('resets all progress on face loss, multiple faces, or long frame gap', () => {
    let s = feed(ready(), 1100, 350, { yaw: 24 });
    for (const faces of [0, 2]) { const reset = advanceChallenge(s, frame(1500, { faces })); expect(reset.phase).toBe('calibrate'); expect(reset.index).toBe(0); expect(reset.baseline).toBeNull(); }
    s = advanceChallenge(s, frame(2500)); expect(s.phase).toBe('calibrate');
  });
  it('ignores stale frames and times out each step', () => {
    let s = ready(); expect(advanceChallenge(s, frame(1000, { yaw: 25 }))).toBe(s);
    s = feed(s, 1100, 16000); expect(s.phase).toBe('timed_out');
    expect(advanceChallenge(s, frame(18000, { yaw: 25 }))).toBe(s);
  });
  it('requires a centered single face and rejects invalid measurements', () => {
    expect(feed(newChallenge([...actions]), 0, 1000, { centered: false }).phase).toBe('calibrate');
    expect(advanceChallenge(ready(), frame(1100, { yaw: NaN })).phase).toBe('calibrate');
  });
  it('resets a passed challenge if the face disappears before the final photograph', () => {
    const passed = { ...ready(), phase: 'passed' as const, index: 3 };
    expect(advanceChallenge(passed, frame(1100, { faces: 0 })).phase).toBe('calibrate');
    expect(advanceChallenge(passed, frame(1100, { faces: 2 })).index).toBe(0);
    expect(advanceChallenge(passed, frame(2000)).phase).toBe('calibrate');
    expect(advanceChallenge(passed, frame(1100)).phase).toBe('passed');
  });
  it('uses an unmirrored matrix: own-left is positive yaw; down is positive pitch', () => {
    const theta = Math.PI / 6, c = Math.cos(theta), s = Math.sin(theta);
    expect(rotationFromMatrix([c,0,-s,0, 0,1,0,0, s,0,c,0, 0,0,0,1]).yaw).toBeCloseTo(30);
    expect(rotationFromMatrix([1,0,0,0, 0,c,s,0, 0,-s,c,0, 0,0,0,1]).pitch).toBeCloseTo(30);
    expect(rotationFromMatrix([]).yaw).toBeNaN();
  });
});
