import { describe, expect, it } from 'vitest';
import { advanceLightGate, estimateLuma, faceRegion, minimumFaceLuma, newLightGate, resumeLightGate } from './lighting';
const pixels = (luma: number, width = 10, height = 10) => Uint8ClampedArray.from({ length: width * height * 4 }, (_, i) => i % 4 === 3 ? 255 : luma);
describe('client low-light gate', () => {
  it('measures dark and bright images without relying on their background', () => {
    expect(estimateLuma(pixels(5), 10, 10)).toBeCloseTo(5);
    expect(estimateLuma(pixels(200), 10, 10)).toBeCloseTo(200);
    const image = pixels(230);
    for (let y = 2; y < 8; y++) for (let x = 3; x < 7; x++) for (let c = 0; c < 3; c++) image[(y * 10 + x) * 4 + c] = 20;
    expect(estimateLuma(image, 10, 10, { x: .3, y: .2, width: .4, height: .6 })).toBeCloseTo(20);
    expect(estimateLuma(image, 10, 10, { x: 0, y: 0, width: 1, height: 1 })).toBeGreaterThan(150);
  });
  it('stops immediately in darkness and requires 500ms stable brighter light to recover', () => {
    let state = newLightGate();
    for (const at of [0,150,300,450]) { state = advanceLightGate(state, 120, at); expect(state.blocked).toBe(true); }
    state = advanceLightGate(state, 120, 600); expect(state.blocked).toBe(false);
    state = advanceLightGate(state, 44, 700); expect(state.blocked).toBe(true); expect(state.goodSince).toBeNull();
    for (const at of [850,1000,1150,1300]) { state = advanceLightGate(state, 60, at); expect(state.blocked).toBe(true); }
    expect(advanceLightGate(state, 60, 1450).blocked).toBe(false);
  });
  it('uses hysteresis and resets recovery when brightness flickers or frames stop', () => {
    let state = advanceLightGate(newLightGate(), 60, 0);
    state = advanceLightGate(state, 54, 150); expect(state.goodSince).toBeNull();
    state = advanceLightGate(state, 60, 300); state = advanceLightGate(state, 60, 1500);
    expect(state.blocked).toBe(true); expect(state.goodSince).toBe(1500);
    expect(advanceLightGate(state, 60, 1490)).toBe(state);
    expect(advanceLightGate(state, NaN, 1600).blocked).toBe(true);
  });
  it('starts normally in bright light but never resumes automatically after any actual darkness', () => {
    let state = newLightGate();
    for (const at of [0,150,300,450,600]) state = advanceLightGate(state, 100, at);
    expect(state.blocked).toBe(false); expect(state.needsResume).toBe(false);
    state = advanceLightGate(state, 10, 750); expect(state.needsResume).toBe(true);
    expect(resumeLightGate(state)).toBe(state);
    for (const at of [900,1050,1200,1350,1500]) state = advanceLightGate(state, 100, at);
    expect(state.blocked).toBe(false); expect(state.needsResume).toBe(true);
    state = advanceLightGate(state, 100, 1650); expect(state.needsResume).toBe(true);
    expect(resumeLightGate(state).needsResume).toBe(false);
    const darkAgain = advanceLightGate(state, 20, 1800); expect(resumeLightGate(darkAgain).needsResume).toBe(true);
  });
  it('requires explicit resume after an initially dark or invalid camera reading', () => {
    for (const luma of [0, NaN]) {
      let state = advanceLightGate(newLightGate(), luma, 0);
      for (const at of [150,300,450,600,750]) state = advanceLightGate(state, 100, at);
      expect(state.blocked).toBe(false); expect(state.needsResume).toBe(true);
    }
  });
  it('validates configuration and malformed pixel inputs safely', () => {
    for (const value of [undefined,'','oops','NaN','Infinity','9','181']) expect(minimumFaceLuma(value)).toBe(45);
    expect(minimumFaceLuma('60')).toBe(60);
    expect(estimateLuma([], 10, 10)).toBeNaN(); expect(estimateLuma(pixels(5), -10, 10)).toBeNaN();
    expect(estimateLuma(new Uint8ClampedArray(400), 10, 10)).toBeNaN();
  });
  it('uses a bounded inner face region and falls back when no usable landmarks exist', () => {
    expect(faceRegion(undefined)).toBeUndefined(); expect(faceRegion([])).toBeUndefined();
    const points = Array.from({ length: 478 }, () => ({ x: .5, y: .5 }));
    points[234] = { x: .2, y: .5 }; points[454] = { x: .8, y: .5 }; points[10] = { x: .5, y: .1 }; points[152] = { x: .5, y: .9 };
    const region = faceRegion(points)!; expect(region.x).toBeCloseTo(.29); expect(region.width).toBeCloseTo(.42); expect(region.height).toBeCloseTo(.544);
  });
});
