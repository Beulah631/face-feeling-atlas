import {
  EMOTIONS,
  type Emotion,
  type BoundingBox,
  type FaceDetectionResult,
  type EmotionPredictionResult,
  type PreprocessResult,
  type MonitoringAlert,
  type Observation,
} from "./types";

/**
 * emotionService — the single integration point between the UI and any real
 * inference backend.
 *
 * Demo mode produces clearly-labelled SIMULATED values so the prototype can be
 * demonstrated without trained weights. When `backendUrl` is configured the
 * same functions call a real Python/FastAPI service:
 *
 *   POST {backendUrl}/api/detect-face     -> FaceDetectionResult-like payload
 *   POST {backendUrl}/api/predict-emotion -> { emotion, confidence, probabilities? }
 *
 * Nothing in this file claims that YOLOv8 or Swin Transformer inference is
 * running: `source` is always reported honestly as "demo" or "backend".
 */

export interface CameraStartOptions {
  deviceId?: string | undefined;
}

export class CameraPermissionError extends Error {}

let activeStream: MediaStream | null = null;

function randomBetween(min: number, max: number) {
  return min + Math.random() * (max - min);
}

async function postJson<T>(url: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal: signal ?? null,
  });
  if (!res.ok) throw new Error(`Request failed (${res.status}) for ${url}`);
  return (await res.json()) as T;
}

/* ------------------------------------------------------------------ camera */

async function startCamera(options: CameraStartOptions = {}): Promise<MediaStream> {
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
    throw new CameraPermissionError("Camera API is not available in this browser.");
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: options.deviceId
        ? { deviceId: { exact: options.deviceId } }
        : { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    activeStream = stream;
    return stream;
  } catch (err) {
    const name = (err as DOMException)?.name;
    if (name === "NotAllowedError" || name === "SecurityError") {
      throw new CameraPermissionError(
        "Camera permission is required for live facial analysis.",
      );
    }
    if (name === "NotFoundError" || name === "OverconstrainedError") {
      throw new CameraPermissionError("No usable camera device was found.");
    }
    throw new CameraPermissionError(
      "Camera could not be started. Please check your device and try again.",
    );
  }
}

function stopCamera(stream?: MediaStream | null) {
  const target = stream ?? activeStream;
  target?.getTracks().forEach((track) => track.stop());
  if (target === activeStream) activeStream = null;
}

async function listCameras(): Promise<MediaDeviceInfo[]> {
  if (typeof navigator === "undefined" || !navigator.mediaDevices?.enumerateDevices) {
    return [];
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((d) => d.kind === "videoinput");
}

/* ------------------------------------------------------------------- frames */

/** Draws the current video frame to a canvas and returns a JPEG data URL. */
function captureFrame(
  video: HTMLVideoElement,
  maxWidth = 640,
): { dataUrl: string; width: number; height: number } | null {
  if (!video.videoWidth || !video.videoHeight) return null;
  const scale = Math.min(1, maxWidth / video.videoWidth);
  const width = Math.round(video.videoWidth * scale);
  const height = Math.round(video.videoHeight * scale);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(video, 0, 0, width, height);
  return { dataUrl: canvas.toDataURL("image/jpeg", 0.85), width, height };
}

/* ---------------------------------------------------------- face detection */

/** Simulated, centre-weighted bounding box in normalized (0..1) coordinates. */
function simulateBox(): BoundingBox {
  const size = randomBetween(0.3, 0.4);
  return {
    x: Number((0.5 - size / 2 + randomBetween(-0.05, 0.05)).toFixed(4)),
    y: Number((0.42 - size / 2 + randomBetween(-0.04, 0.04)).toFixed(4)),
    width: Number(size.toFixed(4)),
    height: Number((size * 1.18).toFixed(4)),
  };
}

interface DetectFaceArgs {
  frameDataUrl?: string | null | undefined;
  backendUrl?: string | undefined;
  demoMode: boolean;
  signal?: AbortSignal | undefined;
}

async function detectFace({
  frameDataUrl,
  backendUrl,
  demoMode,
  signal,
}: DetectFaceArgs): Promise<FaceDetectionResult> {
  if (!demoMode && backendUrl) {
    const raw = await postJson<{
      detected: boolean;
      confidence?: number;
      faceCount?: number;
      box?: BoundingBox | null;
      boxes?: BoundingBox[];
    }>(`${backendUrl.replace(/\/$/, "")}/api/detect-face`, { image: frameDataUrl }, signal);
    const boxes = raw.boxes ?? (raw.box ? [raw.box] : []);
    return {
      detected: raw.detected,
      faceCount: raw.faceCount ?? boxes.length,
      confidence: typeof raw.confidence === "number" ? raw.confidence : null,
      box: boxes[0] ?? null,
      source: "backend",
    };
  }

  // Demo mode: simulated detection. No real detector runs, so confidence is null.
  const roll = Math.random();
  const faceCount = roll < 0.08 ? 0 : roll > 0.94 ? 2 : 1;
  return {
    detected: faceCount > 0,
    faceCount,
    confidence: null,
    box: faceCount > 0 ? simulateBox() : null,
    source: "demo",
  };
}

/* ------------------------------------------------------------ preprocessing */

interface PreprocessArgs {
  video?: HTMLVideoElement | null | undefined;
  box: BoundingBox | null;
  /** Target square size expected by the selected model. */
  targetSize?: number | undefined;
}

/**
 * Crops the detected face region and resizes it to the model input size.
 * Normalization to [0,1] happens on the tensor side of a real backend; here we
 * only report that the step was prepared.
 */
async function preprocessFace({
  video,
  box,
  targetSize = 224,
}: PreprocessArgs): Promise<PreprocessResult> {
  const base: PreprocessResult = {
    dataUrl: null,
    width: targetSize,
    height: targetSize,
    normalized: false,
  };
  if (!video || !box || !video.videoWidth) return base;

  const canvas = document.createElement("canvas");
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext("2d");
  if (!ctx) return base;

  const sx = box.x * video.videoWidth;
  const sy = box.y * video.videoHeight;
  const sw = box.width * video.videoWidth;
  const sh = box.height * video.videoHeight;
  ctx.drawImage(video, sx, sy, sw, sh, 0, 0, targetSize, targetSize);

  return {
    dataUrl: canvas.toDataURL("image/jpeg", 0.9),
    width: targetSize,
    height: targetSize,
    normalized: true,
  };
}

/* ------------------------------------------------------------- prediction */

function simulateProbabilities(dominant: Emotion): Record<Emotion, number> {
  const raw: Record<string, number> = {};
  EMOTIONS.forEach((e) => {
    raw[e] = randomBetween(0.5, 6);
  });
  raw[dominant] = randomBetween(55, 88);
  const total = EMOTIONS.reduce((sum, e) => sum + raw[e]!, 0);
  const probabilities = {} as Record<Emotion, number>;
  let running = 0;
  EMOTIONS.forEach((e, i) => {
    if (i === EMOTIONS.length - 1) {
      probabilities[e] = Number((100 - running).toFixed(1));
    } else {
      const v = Number(((raw[e]! / total) * 100).toFixed(1));
      probabilities[e] = v;
      running = Number((running + v).toFixed(1));
    }
  });
  return probabilities;
}

interface PredictArgs {
  faceDataUrl?: string | null | undefined;
  backendUrl?: string | undefined;
  demoMode: boolean;
  /** Bias the simulation so demo emotions persist for a few cycles. */
  preferredEmotion?: Emotion | null | undefined;
  signal?: AbortSignal | undefined;
}

async function predictEmotion({
  faceDataUrl,
  backendUrl,
  demoMode,
  preferredEmotion,
  signal,
}: PredictArgs): Promise<EmotionPredictionResult> {
  if (!demoMode && backendUrl) {
    const raw = await postJson<{
      emotion: Emotion;
      confidence: number;
      probabilities?: Record<Emotion, number>;
    }>(
      `${backendUrl.replace(/\/$/, "")}/api/predict-emotion`,
      { image: faceDataUrl },
      signal,
    );
    const probabilities =
      raw.probabilities ??
      (EMOTIONS.reduce((acc, e) => {
        acc[e] = e === raw.emotion ? Number((raw.confidence * 100).toFixed(1)) : 0;
        return acc;
      }, {} as Record<Emotion, number>));
    return {
      emotion: raw.emotion,
      confidence: raw.confidence,
      probabilities,
      source: "backend",
    };
  }

  const dominant =
    preferredEmotion ?? EMOTIONS[Math.floor(Math.random() * EMOTIONS.length)]!;
  const probabilities = simulateProbabilities(dominant);
  return {
    emotion: dominant,
    confidence: Number((probabilities[dominant] / 100).toFixed(3)),
    probabilities,
    source: "demo",
  };
}

/* ------------------------------------------------------------------ history */

const HISTORY_LIMIT = 20;

function appendObservation(history: Observation[], observation: Observation): Observation[] {
  return [observation, ...history].slice(0, HISTORY_LIMIT);
}

function getEmotionHistory(history: Observation[]): Observation[] {
  return history.slice(0, HISTORY_LIMIT);
}

/* -------------------------------------------------------------------- alert */

interface AlertArgs {
  history: Observation[];
  thresholdPct: number;
  consecutive: number;
}

/**
 * Returns an alert when the most recent `consecutive` observations share the
 * same emotion above the confidence threshold. The message only ever describes
 * the observed expression — never intent, honesty or misconduct.
 */
function generateAlert({
  history,
  thresholdPct,
  consecutive,
}: AlertArgs): MonitoringAlert | null {
  if (history.length < consecutive) return null;
  const window = history.slice(0, consecutive);
  const first = window[0]!;
  const qualifies = window.every(
    (o) => o.emotion === first.emotion && o.confidence * 100 >= thresholdPct,
  );
  if (!qualifies) return null;
  if (first.emotion === "Neutral" || first.emotion === "Happy") return null;

  return {
    id: `${first.emotion}-${first.at}`,
    emotion: first.emotion,
    message: `${first.emotion}-like expression detected consistently.`,
    at: first.at,
    count: consecutive,
    threshold: thresholdPct,
  };
}

export const emotionService = {
  startCamera,
  stopCamera,
  listCameras,
  captureFrame,
  detectFace,
  preprocessFace,
  predictEmotion,
  appendObservation,
  getEmotionHistory,
  generateAlert,
  HISTORY_LIMIT,
};

export type EmotionService = typeof emotionService;
