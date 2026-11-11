/**
 * Shared domain types for the Facial Emotion Recognition prototype.
 */

export const EMOTIONS = [
  "Happy",
  "Sad",
  "Angry",
  "Fear",
  "Surprise",
  "Disgust",
  "Neutral",
] as const;

export type Emotion = (typeof EMOTIONS)[number];

export type InferenceSource = "demo" | "backend";

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Shape returned by POST /api/detect-face */
export interface FaceDetectionResult {
  detected: boolean;
  /** Number of faces found in the frame (0, 1 or more). */
  faceCount: number;
  /** null when no real detector produced a score (demo mode). */
  confidence: number | null;
  box: BoundingBox | null;
  source: InferenceSource;
}

/** Shape returned by POST /api/predict-emotion */
export interface EmotionPredictionResult {
  emotion: Emotion;
  /** Prediction confidence in [0,1]. NOT model accuracy. */
  confidence: number;
  probabilities: Record<Emotion, number>;
  source: InferenceSource;
}

export interface PreprocessResult {
  /** data URL of the cropped + resized face patch, when available. */
  dataUrl: string | null;
  width: number;
  height: number;
  normalized: boolean;
}

export type PipelineStage =
  | "capture"
  | "detect"
  | "crop"
  | "resize"
  | "normalize"
  | "inputPrep"
  | "features"
  | "classify";

export type StageStatus = "waiting" | "processing" | "completed";

export type FaceStatus = "none" | "single" | "multiple";

export interface Observation {
  id: string;
  /** epoch ms */
  at: number;
  /** HH:MM:SS */
  time: string;
  emotion: Emotion;
  confidence: number;
  status: "Detected" | "Simulated";
  source: InferenceSource;
}

export interface MonitoringAlert {
  id: string;
  emotion: Emotion;
  message: string;
  at: number;
  count: number;
  threshold: number;
}

export interface Settings {
  demoMode: boolean;
  alertsEnabled: boolean;
  /** percentage 50..99 */
  confidenceThreshold: number;
  /** consecutive matching observations required */
  consecutiveObservations: number;
  /** deviceId of the selected camera, "" = default */
  cameraDeviceId: string;
  /** base URL of a future FastAPI inference backend */
  backendUrl: string;
  /** how often (ms) an analysis cycle runs */
  intervalMs: number;
  voiceAnnouncementsEnabled: boolean;
  /** URI of a selected English browser voice; empty uses the browser default. */
  voiceName: string;
  voiceCooldownSeconds: number;
  /** Live announcements below this model score are treated as uncertain. */
  voiceConfidenceThreshold: number;
}

const configuredBackendUrl = import.meta.env["VITE_BACKEND_URL"]?.trim();

export const DEFAULT_SETTINGS: Settings = {
  demoMode: false,
  alertsEnabled: true,
  confidenceThreshold: 80,
  consecutiveObservations: 3,
  cameraDeviceId: "",
  // Vercel Services routes the API through the same origin. Keep local
  // development on the standalone FastAPI port unless explicitly overridden.
  backendUrl: configuredBackendUrl || (import.meta.env.PROD ? "/" : "http://127.0.0.1:8000"),
  intervalMs: 2500,
  voiceAnnouncementsEnabled: true,
  voiceName: "",
  voiceCooldownSeconds: 15,
  voiceConfidenceThreshold: 50,
};
