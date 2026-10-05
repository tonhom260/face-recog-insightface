import { useEffect, useRef, useState } from 'react';
import { FaceLandmarker, FilesetResolver, type FaceLandmarkerResult } from '@mediapipe/tasks-vision';
import { Camera as CameraIcon, Check, RotateCcw, X, ArrowLeft, ArrowRight, MoveDown, MoveUp, Smile, Eye } from 'lucide-react';
import { ACTION_COPY, advanceChallenge, isNeutral, newChallenge, rotationFromMatrix, type ChallengeAction, type ChallengeState, type Measurement } from './challenge';
import { advanceLightGate, estimateLuma, faceRegion, minimumFaceLuma, newLightGate, resumeLightGate, type Region } from './lighting';

export type Capture = { image: Blob; actions: ChallengeAction[] };
type Props = { mode: 'attendance' | 'enroll'; onCapture: (capture: Capture) => void; onCancel: () => void };
const icons = { turn_left: ArrowLeft, turn_right: ArrowRight, look_down: MoveDown, look_up: MoveUp, smile: Smile, blink: Eye };
const MIN_FACE_LUMA = minimumFaceLuma(import.meta.env.VITE_MIN_FACE_LUMA);

export function measurement(result: FaceLandmarkerResult, at: number): Measurement {
  const face = result.faceLandmarks[0];
  const rotation = rotationFromMatrix(result.facialTransformationMatrixes[0]?.data ?? []);
  const scores = result.faceBlendshapes[0]?.categories ?? [];
  const score = (name: string) => scores.find(s => s.categoryName === name)?.score ?? 0;
  const points = face ? [face[1], face[33], face[263], face[152], face[10]] : [];
  // Keep the whole face in a generous central area; allow head turns inside it.
  const centered = points.length > 0 && points.every(p => p.x > .1 && p.x < .9 && p.y > .05 && p.y < .95)
    && face[1].x > .25 && face[1].x < .75 && face[1].y > .2 && face[1].y < .8;
  return { at, faces: result.faceLandmarks.length, ...rotation, centered, smile: Math.min(score('mouthSmileLeft'), score('mouthSmileRight')), blink: Math.min(score('eyeBlinkLeft'), score('eyeBlinkRight')) };
}
const cameraError = (error: unknown) => {
  if (error instanceof DOMException) {
    if (error.name === 'NotAllowedError' || error.name === 'SecurityError') return 'กรุณาอนุญาตให้ใช้กล้องในตั้งค่าเบราว์เซอร์ แล้วเปิดกล้องอีกครั้ง';
    if (error.name === 'NotFoundError') return 'ไม่พบกล้อง กรุณาใช้อุปกรณ์ที่มีกล้องหน้า';
    if (error.name === 'NotReadableError') return 'เปิดกล้องไม่ได้ ปิดแอปอื่นที่ใช้กล้องแล้วลองใหม่';
  }
  return 'เปิดระบบกล้องไม่สำเร็จ ตรวจสอบการเชื่อมต่อแล้วลองอีกครั้ง';
};

export default function Camera({ mode, onCapture, onCancel }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const callbackRef = useRef(onCapture);
  callbackRef.current = onCapture;
  const [attempt, setAttempt] = useState(0);
  const [state, setState] = useState<ChallengeState>(() => newChallenge());
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [ready, setReady] = useState(false);
  const [notice, setNotice] = useState('ให้ใบหน้าอยู่ในกรอบและมองตรง');
  const [lightState, setLightState] = useState(newLightGate);
  const shootRef = useRef<(() => void) | null>(null);
  const resumeRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    let cancelled = false, stream: MediaStream | null = null, landmarker: FaceLandmarker | null = null, raf = 0;
    let machine = newChallenge(), lastInference = -1, lastVideo = -1, lastRender = -1, frontalAt: number | null = null, captured = false;
    let light = newLightGate(), lastLightAt = -1, region: Region | undefined;
    const lightCanvas = document.createElement('canvas');
    lightCanvas.width = 64; lightCanvas.height = 48;
    const lightContext = lightCanvas.getContext('2d', { willReadFrequently: true });
    setLightState(light);
    setState(machine); setLoading(true); setError(''); setReady(false);
    setNotice('ให้ใบหน้าอยู่ในกรอบและมองตรง');
    const video = videoRef.current!;
    const stop = () => {
      cancelAnimationFrame(raf); stream?.getTracks().forEach(track => track.stop());
      video.pause(); video.srcObject = null; landmarker?.close(); landmarker = null; shootRef.current = null; resumeRef.current = null;
    };
    const sampleLight = (at: number) => {
      if (!video.videoWidth || !lightContext) return false;
      lightContext.drawImage(video, 0, 0, lightCanvas.width, lightCanvas.height);
      const pixels = lightContext.getImageData(0, 0, lightCanvas.width, lightCanvas.height);
      const luma = estimateLuma(pixels.data, pixels.width, pixels.height, region);
      const wasPaused = light.needsResume;
      light = advanceLightGate(light, luma, at, MIN_FACE_LUMA); lastLightAt = at;
      setLightState(light);
      if (light.blocked || light.needsResume) {
        // Keep the camera open to monitor recovery, but discard every motion
        // result and neutral/capture hold. Darkness starts a fresh attempt.
        if (light.needsResume && !wasPaused) machine = newChallenge();
        // A face may move while inference is paused. Recover from a current
        // central sample, then verify the newly detected face ROI immediately.
        region = undefined;
        frontalAt = null; shootRef.current = null; setReady(false); setState({ ...machine });
      }
      return !light.blocked && !light.needsResume;
    };
    resumeRef.current = () => {
      if (cancelled || captured || !video.videoWidth) return;
      sampleLight(performance.now());
      light = resumeLightGate(light, MIN_FACE_LUMA);
      if (light.blocked || light.needsResume) return;
      machine = newChallenge(); frontalAt = null; region = undefined;
      setState(machine); setLightState(light); setReady(false); shootRef.current = null;
    };
    const capture = () => {
      if (cancelled || captured || !video.videoWidth) return;
      // Recheck immediately: a previously enabled shutter cannot use an old
      // bright sample after the current frame becomes dark.
      if (!sampleLight(performance.now())) return;
      captured = true;
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth; canvas.height = video.videoHeight;
      const context = canvas.getContext('2d');
      if (!context) { captured = false; setError('บันทึกภาพไม่ได้ กรุณาลองเปิดกล้องใหม่'); return; }
      // Capture raw pixels; the preview alone is mirrored.
      context.drawImage(video, 0, 0);
      canvas.toBlob(blob => {
        if (cancelled) return;
        if (!blob) { captured = false; setError('บันทึกภาพไม่ได้ กรุณาลองเปิดกล้องใหม่'); return; }
        stop(); callbackRef.current({ image: blob, actions: mode === 'attendance' ? machine.actions : [] });
      }, 'image/jpeg', .92);
    };
    const tick = (at: number) => {
      if (cancelled || captured) return;
      try {
        if (video.readyState >= 2 && at - lastLightAt >= 150) sampleLight(at);
        if (light.blocked || light.needsResume) { raf = requestAnimationFrame(tick); return; }
        if (landmarker && video.readyState >= 2 && at - lastInference >= 40 && video.currentTime !== lastVideo) {
          if (lastInference >= 0 && at - lastInference > 500) frontalAt = null;
          lastInference = at; lastVideo = video.currentTime;
          const result = landmarker.detectForVideo(video, at);
          const m = measurement(result, at);
          const newFaceRegion = m.faces === 1 && !region;
          region = m.faces === 1 ? faceRegion(result.faceLandmarks[0]) : undefined;
          if (newFaceRegion && region && !sampleLight(performance.now())) { raf = requestAnimationFrame(tick); return; }
          if (mode === 'attendance') machine = advanceChallenge(machine, m);
          const neutral = isNeutral(m, mode === 'attendance' ? machine.baseline : null);
          if (neutral) frontalAt ??= at; else frontalAt = null;
          const canCapture = frontalAt !== null && at - frontalAt >= 500;
          if (mode === 'attendance' && machine.phase === 'passed' && canCapture) capture();
          if (mode === 'enroll') shootRef.current = canCapture ? capture : null;
          if (at - lastRender > 90) {
            lastRender = at; setState({ ...machine }); setReady(canCapture);
            setNotice(m.faces > 1 ? 'ให้มีเฉพาะคุณอยู่ในภาพ' : m.faces === 0 ? 'ยังไม่พบใบหน้า ให้ใบหน้าอยู่ในกรอบ' : !m.centered ? 'ขยับให้ใบหน้าอยู่กลางกรอบ' : '');
          }
        }
        raf = requestAnimationFrame(tick);
      } catch {
        stop(); setError('ระบบอ่านใบหน้าหยุดทำงาน กรุณาเปิดกล้องใหม่');
      }
    };
    async function start() {
      try {
        if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) { setError('กล้องต้องใช้การเชื่อมต่อ HTTPS และเบราว์เซอร์ที่รองรับ กรุณาเปิดด้วย Safari หรือ Chrome รุ่นล่าสุด'); setLoading(false); return; }
        stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user', width: { ideal: 640 }, height: { ideal: 480 } }, audio: false });
        if (cancelled) { stream.getTracks().forEach(track => track.stop()); return; }
        video.srcObject = stream; await video.play();
        const fileset = await FilesetResolver.forVisionTasks('/wasm');
        const options = { baseOptions: { modelAssetPath: '/models/face_landmarker.task' }, runningMode: 'VIDEO' as const, numFaces: 2, outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true, minFaceDetectionConfidence: .6, minFacePresenceConfidence: .6, minTrackingConfidence: .6 };
        try { landmarker = await FaceLandmarker.createFromOptions(fileset, { ...options, baseOptions: { ...options.baseOptions, delegate: 'GPU' } }); }
        catch { if (!cancelled) landmarker = await FaceLandmarker.createFromOptions(fileset, { ...options, baseOptions: { ...options.baseOptions, delegate: 'CPU' } }); }
        if (cancelled) { stop(); return; }
        setLoading(false); raf = requestAnimationFrame(tick);
      } catch (err) {
        if (!cancelled) { stop(); setError(cameraError(err)); setLoading(false); }
      }
    }
    void start();
    return () => { cancelled = true; stop(); };
  }, [attempt, mode]);

  const currentAction = state.actions[Math.min(state.index, 2)];
  const Icon = icons[currentAction];
  const phase = state.phase;
  const lightBlocked = !loading && !error && (lightState.blocked || lightState.needsResume);
  const tooDark = lightBlocked && lightState.luma < MIN_FACE_LUMA;
  let title = mode === 'enroll' ? 'มองตรงเข้าหากล้อง' : phase === 'calibrate' || phase === 'neutral' ? 'มองตรงและทำหน้าปกติ' : phase === 'return' || phase === 'passed' ? 'กลับมามองตรง' : ACTION_COPY[currentAction].title;
  let hint = mode === 'enroll' ? 'เห็นใบหน้าชัดเจน ถอดแว่นกันแดด และอยู่ในที่สว่าง' : phase === 'action' ? ACTION_COPY[currentAction].hint : 'ค้างไว้สักครู่ ให้ใบหน้าอยู่ในกรอบ';
  if (phase === 'timed_out') { title = 'ใช้เวลานานเกินไป'; hint = 'ลองใหม่อีกครั้ง ขยับช้า ๆ ในที่มีแสงสว่าง'; }
  if (lightBlocked) {
    title = tooDark ? 'แสงน้อยเกินไป' : lightState.blocked ? 'ค้างไว้สักครู่ กำลังตรวจแสง' : 'แสงพร้อมแล้ว';
    hint = tooDark ? 'ขยับไปที่สว่าง หรือเปิดไฟให้ส่องใบหน้า' : lightState.blocked ? 'รอให้แสงบนใบหน้าคงที่ก่อนเริ่ม' : 'กดเริ่มตรวจต่อ เพื่อเริ่มขั้นตอนใหม่';
  }
  return <section className="camera-screen" aria-labelledby="camera-title">
    <div className="section-heading"><div><p className="eyebrow">{mode === 'enroll' ? 'ลงทะเบียนใบหน้า' : 'ตรวจสอบก่อนลงเวลา'}</p><h1 id="camera-title">{mode === 'enroll' ? 'ภาพเดียว เริ่มต้นได้เลย' : 'ทำตามท่าทางทีละขั้น'}</h1></div><button className="icon-button" onClick={onCancel} aria-label="ปิดกล้อง"><X /></button></div>
    <div className="camera-layout"><div className={`camera-preview ${ready && !lightBlocked ? 'ready' : ''} ${tooDark ? 'too-dark' : ''}`}>
      <video ref={videoRef} autoPlay muted playsInline aria-label="ภาพจากกล้องหน้า" />
      <div className="face-guide" aria-hidden="true"><span /><span /><span /><span /></div>
      <div className="camera-top-tag"><span className="live-dot" />กล้องหน้า</div>
      <div className="camera-bottom-tag">{loading ? 'กำลังเตรียมกล้อง…' : tooDark ? 'แสงน้อยเกินไป กรุณาเพิ่มแสง' : lightState.needsResume && !lightState.blocked ? 'แสงพร้อมแล้ว กดเริ่มตรวจต่อ' : lightBlocked ? 'กำลังตรวจแสง…' : notice || (mode === 'enroll' && ready ? 'พร้อมถ่ายภาพ' : 'ใบหน้าอยู่ในกรอบ')}</div>
      {loading && <div className="camera-loading"><span className="spinner" /><p>กำลังเตรียมกล้อง</p></div>}
    </div><div className="camera-instructions">
      {mode === 'attendance' && <div className="step-track" aria-label={`ขั้นตอน ${Math.min(state.index + 1, 3)} จาก 3`}>{[0,1,2].map(i => <span key={i} className={i < state.index ? 'complete' : i === state.index ? 'current' : ''}>{i < state.index ? <Check size={16} /> : i + 1}</span>)}</div>}
      <div className="motion-icon" aria-hidden="true">{mode === 'enroll' || phase !== 'action' ? <CameraIcon size={32} /> : <Icon size={32} />}</div>
      <p className="eyebrow">{mode === 'attendance' ? `ขั้นตอน ${Math.min(state.index + 1, 3)} / 3` : 'ภาพสำหรับยืนยันตัวตน'}</p>
      <h2 aria-live="polite">{title}</h2><p>{hint}</p>
      {tooDark && <div className="lighting-warning" role="status">{mode === 'attendance' ? 'หยุดตรวจท่าทางชั่วคราว ปรับแสงแล้วกดเริ่มตรวจต่อ เพื่อเริ่มใหม่ทั้ง 3 ขั้นตอน' : 'ยังถ่ายภาพไม่ได้ ปรับแสงแล้วกดเริ่มตรวจต่อ'}</div>}
      {error && <div className="error-banner" role="alert">{error}</div>}
      {(error || phase === 'timed_out') ? <button className="button primary full" onClick={() => setAttempt(a => a + 1)}><RotateCcw size={18} />เปิดกล้องใหม่</button> : <>{lightState.needsResume && <button className="button primary full" disabled={lightState.blocked} onClick={() => resumeRef.current?.()}><RotateCcw size={18} />เริ่มตรวจต่อ</button>}{mode === 'enroll' ? <button className={`button ${lightState.needsResume ? 'outline' : 'primary'} full`} disabled={!ready || loading || lightBlocked} onClick={() => shootRef.current?.()}><CameraIcon size={18} />ถ่ายภาพใบหน้า</button> : !tooDark && <div className="camera-tip">{lightState.needsResume ? 'จะเริ่มตรวจใหม่จากขั้นตอนที่ 1 เมื่อคุณกดเริ่มตรวจต่อ' : phase === 'passed' ? 'กำลังเตรียมภาพ กรุณามองตรง' : 'ระบบจะถ่ายภาพเมื่อทำครบทั้ง 3 ขั้นตอน'}</div>}</>}
      <button className="button text full" onClick={onCancel}>ยกเลิก</button>
      <p className="privacy-note">กล้องใช้ตรวจท่าทางบนอุปกรณ์ของคุณ ระบบส่งเฉพาะภาพถ่ายเพื่อเปรียบเทียบใบหน้า</p>
    </div></div>
  </section>;
}
