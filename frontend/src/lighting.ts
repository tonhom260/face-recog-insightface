/** Local image brightness gate. This is a low-light usability check, not a
 * guarantee of face quality. Tune the minimum using real employee phones.
 */
export const DEFAULT_MIN_FACE_LUMA = 45;
export const LIGHT_RECOVERY_MARGIN = 10;
export const LIGHT_RECOVERY_MS = 500;
export type Region = { x: number; y: number; width: number; height: number };
export type LightGate = { blocked: boolean; needsResume: boolean; goodSince: number | null; lastAt: number; luma: number };
export function minimumFaceLuma(value: string | undefined): number {
  if (value === undefined || value.trim() === '') return DEFAULT_MIN_FACE_LUMA;
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 10 && parsed <= 180 ? parsed : DEFAULT_MIN_FACE_LUMA;
}
export function newLightGate(): LightGate { return { blocked: true, needsResume: false, goodSince: null, lastAt: -1, luma: 0 }; }
export function advanceLightGate(state: LightGate, luma: number, at: number, minimum = DEFAULT_MIN_FACE_LUMA): LightGate {
  if (!Number.isFinite(at) || at <= state.lastAt) return state;
  if (!Number.isFinite(luma) || luma < 0 || luma > 255) return { blocked: true, needsResume: true, goodSince: null, lastAt: at, luma: 0 };
  const needsResume = state.needsResume || luma < minimum;
  if (!state.blocked) return { blocked: luma < minimum, needsResume, goodSince: null, lastAt: at, luma };
  const threshold = minimum + LIGHT_RECOVERY_MARGIN;
  const goodSince = luma >= threshold ? state.goodSince !== null && at - state.lastAt <= 350 ? state.goodSince : at : null;
  return { blocked: goodSince === null || at - goodSince < LIGHT_RECOVERY_MS, needsResume, goodSince, lastAt: at, luma };
}
/** Call only after sampling the current camera frame immediately on the click. */
export function resumeLightGate(state: LightGate, minimum = DEFAULT_MIN_FACE_LUMA): LightGate {
  return !state.blocked && Number.isFinite(state.luma) && state.luma >= minimum ? { ...state, needsResume: false } : state;
}
export function estimateLuma(pixels: ArrayLike<number>, width: number, height: number, region: Region = { x: .25, y: .2, width: .5, height: .6 }): number {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || pixels.length !== width * height * 4) return NaN;
  if (Object.values(region).some(v => !Number.isFinite(v)) || region.width <= 0 || region.height <= 0) return NaN;
  const x0 = Math.max(0, Math.min(width - 1, Math.floor(region.x * width)));
  const y0 = Math.max(0, Math.min(height - 1, Math.floor(region.y * height)));
  const x1 = Math.max(x0 + 1, Math.min(width, Math.ceil((region.x + region.width) * width)));
  const y1 = Math.max(y0 + 1, Math.min(height, Math.ceil((region.y + region.height) * height)));
  let total = 0, count = 0;
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) {
    const i = (y * width + x) * 4;
    if (pixels[i + 3] === 0) continue;
    total += .2126 * pixels[i] + .7152 * pixels[i + 1] + .0722 * pixels[i + 2]; count++;
  }
  return count ? total / count : NaN;
}
export function faceRegion(landmarks: Array<{ x: number; y: number }> | undefined): Region | undefined {
  if (!landmarks || landmarks.length < 455) return undefined;
  const left = landmarks[234], right = landmarks[454], top = landmarks[10], bottom = landmarks[152];
  const x0 = Math.min(left.x, right.x), x1 = Math.max(left.x, right.x), y0 = Math.min(top.y, bottom.y), y1 = Math.max(top.y, bottom.y);
  if (![x0,x1,y0,y1].every(Number.isFinite) || x1 - x0 < .04 || y1 - y0 < .04) return undefined;
  // Sample cheeks and central face, keeping hair and bright background out.
  return { x: x0 + (x1 - x0) * .15, y: y0 + (y1 - y0) * .18, width: (x1 - x0) * .7, height: (y1 - y0) * .68 };
}
