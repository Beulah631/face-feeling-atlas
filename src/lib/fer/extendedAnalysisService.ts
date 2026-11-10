import type { Emotion } from "./types";

export type AnalysisModel = "swin" | "cnn" | "compare";

export interface FaceBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface ModelPrediction {
  emotion: Emotion;
  confidence: number;
  probabilities: Record<Emotion, number>;
  source: "backend";
  model_name: string;
}

export interface FaceAnalysis {
  face_id: string;
  box: FaceBox;
  detector_confidence: number;
  predictions: Partial<Record<"swin" | "cnn", ModelPrediction>>;
  model_errors: Partial<Record<"swin" | "cnn", string>>;
}

export interface MultiFaceResponse {
  source: "backend";
  model: AnalysisModel;
  face_count: number;
  faces: FaceAnalysis[];
  model_errors: Record<string, string>;
}

export interface VideoEvent {
  timestamp_seconds: number;
  sample_index: number;
  face_id: string;
  box: FaceBox;
  detector_confidence: number;
  predictions: Partial<Record<"swin" | "cnn", ModelPrediction>>;
  quality?: FaceQualityMeasurements;
}

export interface FaceQualityMeasurements {
  face_visibility: number | null;
  face_visibility_note: string;
  face_crop_width_px: number | null;
  face_crop_height_px: number | null;
  blur_laplacian_variance: number | null;
  brightness_mean: number | null;
  detection_confidence: number | null;
  quality_flags: string[];
  measurement_error?: string | null;
}

export interface ReliabilityReport {
  status: "reliable" | "uncertain" | "insufficient_visual_evidence";
  reasons: string[];
  model_agreement: {
    agreeing_frames: number;
    disagreeing_frames: number;
    paired_prediction_frames: number;
    unavailable_prediction_frames: number;
    agreement_percent: number | null;
    timeline: Array<{
      sample_index: number;
      timestamp_seconds: number;
      face_id: string;
      cnn_emotion: Emotion | null;
      swin_emotion: Emotion | null;
      agree: boolean | null;
    }>;
  };
  temporal_stability: {
    primary_model: "swin" | "cnn";
    most_frequent_expression: Emotion | null;
    expression_counts: Partial<Record<Emotion, number>>;
    expression_changes: number;
    consecutive_prediction_pairs: number;
    stable_sequences: ReliabilitySequence[];
    unstable_sequences: ReliabilitySequence[];
    temporal_consistency_percent: number | null;
    note: string;
  };
  face_quality: {
    insufficient_quality_frames: number;
    frame_count: number;
    low_quality_frame_percent: number;
    usable_frames: number;
    measurements: Array<
      FaceQualityMeasurements & { sample_index: number; timestamp_seconds: number; face_id: string }
    >;
    no_face_frames: Array<{ sample_index: number; timestamp_seconds: number; face_count: number }>;
    visibility_note: string;
  };
  rules: Record<string, number | string>;
  confidence_note: string;
}

export interface ReliabilitySequence {
  face_id: string;
  emotion: Emotion;
  start_sample_index: number;
  end_sample_index: number;
  start_timestamp_seconds: number;
  end_timestamp_seconds: number;
  frames: number;
}

export interface VideoJob {
  job_id: string;
  status: "queued" | "processing" | "cancelling" | "cancelled" | "completed" | "failed";
  file_name: string;
  duration_seconds: number;
  model: AnalysisModel;
  interval_ms: number;
  sampled_frames_total: number;
  frames_processed: number;
  face_detections: number;
  current_timestamp_seconds: number | null;
  processing_seconds: number | null;
  events: VideoEvent[];
  frame_records?: Array<{ sample_index: number; timestamp_seconds: number; face_count: number }>;
  reliability?: ReliabilityReport | null;
  error: string | null;
}

function apiBase(backendUrl: string) {
  const base = backendUrl.trim().replace(/\/$/, "");
  if (!base) throw new Error("Backend URL is not configured in Settings.");
  return base;
}

async function readError(response: Response) {
  try {
    const payload = (await response.json()) as { detail?: unknown };
    return typeof payload.detail === "string" ? payload.detail : `HTTP ${response.status}`;
  } catch {
    return `HTTP ${response.status}`;
  }
}

export async function analyzeMultipleFaces(
  imageDataUrl: string,
  model: AnalysisModel,
  backendUrl: string,
): Promise<MultiFaceResponse> {
  const response = await fetch(`${apiBase(backendUrl)}/api/analyze-faces`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: imageDataUrl, model }),
  });
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as MultiFaceResponse;
}

export function uploadVideoJob(
  file: File,
  model: AnalysisModel,
  intervalMs: number,
  backendUrl: string,
  onProgress: (progress: number) => void,
): {
  promise: Promise<{
    job_id: string;
    status: string;
    duration_seconds: number;
    sampled_frames_total: number;
  }>;
  abort: () => void;
} {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<{
    job_id: string;
    status: string;
    duration_seconds: number;
    sampled_frames_total: number;
  }>((resolve, reject) => {
    xhr.open("POST", `${apiBase(backendUrl)}/api/video-jobs`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100));
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) resolve(xhr.response);
      else reject(new Error(xhr.response?.detail ?? `Video upload failed (HTTP ${xhr.status}).`));
    };
    xhr.onerror = () => reject(new Error("Network error while uploading the video."));
    xhr.onabort = () => reject(new DOMException("Video upload cancelled.", "AbortError"));
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("model", model);
    form.append("interval_ms", String(intervalMs));
    xhr.send(form);
  });
  return { promise, abort: () => xhr.abort() };
}

export async function getVideoJob(jobId: string, backendUrl: string): Promise<VideoJob> {
  const response = await fetch(
    `${apiBase(backendUrl)}/api/video-jobs/${encodeURIComponent(jobId)}`,
  );
  if (!response.ok) throw new Error(await readError(response));
  return (await response.json()) as VideoJob;
}

export async function cancelVideoJob(jobId: string, backendUrl: string): Promise<void> {
  const response = await fetch(
    `${apiBase(backendUrl)}/api/video-jobs/${encodeURIComponent(jobId)}`,
    {
      method: "DELETE",
    },
  );
  if (!response.ok) throw new Error(await readError(response));
}
