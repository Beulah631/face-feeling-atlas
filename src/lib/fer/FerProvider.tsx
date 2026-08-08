import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { emotionService, CameraPermissionError } from "./emotionService";
import {
  DEFAULT_SETTINGS,
  EMOTIONS,
  type BoundingBox,
  type Emotion,
  type FaceStatus,
  type MonitoringAlert,
  type Observation,
  type PipelineStage,
  type Settings,
  type StageStatus,
} from "./types";
import { timeLabel } from "./emotionMeta";

const STORAGE_KEY = "fer.settings.v1";

const EMPTY_STAGES: Record<PipelineStage, StageStatus> = {
  capture: "waiting",
  detect: "waiting",
  crop: "waiting",
  resize: "waiting",
  normalize: "waiting",
  inputPrep: "waiting",
  features: "waiting",
  classify: "waiting",
};

export interface FerState {
  settings: Settings;
  updateSettings: (patch: Partial<Settings>) => void;
  resetSettings: () => void;

  videoRef: React.RefObject<HTMLVideoElement | null>;
  cameraActive: boolean;
  cameraMessage: string | null;
  cameraError: string | null;
  startCamera: () => Promise<void>;
  stopCamera: () => void;
  cameras: MediaDeviceInfo[];
  refreshCameras: () => Promise<void>;

  analyzing: boolean;
  toggleAnalyzing: () => void;
  captureOnce: () => Promise<void>;

  faceStatus: FaceStatus;
  faceCount: number;
  box: BoundingBox | null;
  detectionConfidence: number | null;
  facePreview: string | null;

  stages: Record<PipelineStage, StageStatus>;
  currentEmotion: Emotion | null;
  confidence: number | null;
  probabilities: Record<Emotion, number> | null;
  inferenceSource: "demo" | "backend";

  history: Observation[];
  clearHistory: () => void;
  alert: MonitoringAlert | null;
  dismissAlert: () => void;
  sessionStart: number | null;
}

const FerContext = createContext<FerState | null>(null);

function loadSettings(): Settings {
  if (typeof window === "undefined") return DEFAULT_SETTINGS;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_SETTINGS;
    return { ...DEFAULT_SETTINGS, ...(JSON.parse(raw) as Partial<Settings>) };
  } catch {
    return DEFAULT_SETTINGS;
  }
}

export function FerProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<Settings>(DEFAULT_SETTINGS);
  const [hydrated, setHydrated] = useState(false);

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraMessage, setCameraMessage] = useState<string | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([]);

  const [analyzing, setAnalyzing] = useState(false);
  const [faceStatus, setFaceStatus] = useState<FaceStatus>("none");
  const [faceCount, setFaceCount] = useState(0);
  const [box, setBox] = useState<BoundingBox | null>(null);
  const [detectionConfidence, setDetectionConfidence] = useState<number | null>(null);
  const [facePreview, setFacePreview] = useState<string | null>(null);
  const [stages, setStages] = useState<Record<PipelineStage, StageStatus>>(EMPTY_STAGES);
  const [currentEmotion, setCurrentEmotion] = useState<Emotion | null>(null);
  const [confidence, setConfidence] = useState<number | null>(null);
  const [probabilities, setProbabilities] = useState<Record<Emotion, number> | null>(null);
  const [history, setHistory] = useState<Observation[]>([]);
  const [alert, setAlert] = useState<MonitoringAlert | null>(null);
  const [sessionStart, setSessionStart] = useState<number | null>(null);

  const settingsRef = useRef(settings);
  settingsRef.current = settings;
  const runningRef = useRef(false);
  const holdRef = useRef<{ emotion: Emotion; left: number } | null>(null);

  /* --------------------------------------------------------- persistence */
  useEffect(() => {
    setSettings(loadSettings());
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    } catch {
      /* storage unavailable — settings stay in memory */
    }
  }, [settings, hydrated]);

  const updateSettings = useCallback((patch: Partial<Settings>) => {
    setSettings((prev) => ({ ...prev, ...patch }));
  }, []);

  const resetSettings = useCallback(() => setSettings(DEFAULT_SETTINGS), []);

  /* -------------------------------------------------------------- camera */
  const refreshCameras = useCallback(async () => {
    try {
      setCameras(await emotionService.listCameras());
    } catch {
      setCameras([]);
    }
  }, []);

  const stopCamera = useCallback(() => {
    emotionService.stopCamera(streamRef.current);
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraActive(false);
    setAnalyzing(false);
    setCameraMessage("Camera Stopped");
    setFaceStatus("none");
    setFaceCount(0);
    setBox(null);
    setDetectionConfidence(null);
    setStages(EMPTY_STAGES);
  }, []);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      const stream = await emotionService.startCamera({
        deviceId: settingsRef.current.cameraDeviceId || undefined,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => undefined);
      }
      setCameraActive(true);
      setCameraMessage("Camera Active");
      setSessionStart((prev) => prev ?? Date.now());
      void refreshCameras();
    } catch (err) {
      const message =
        err instanceof CameraPermissionError
          ? err.message
          : "Camera could not be started on this device.";
      setCameraError(message);
      setCameraActive(false);
      setCameraMessage(null);
    }
  }, [refreshCameras]);

  useEffect(() => () => emotionService.stopCamera(streamRef.current), []);

  /* ------------------------------------------------------ analysis cycle */
  const runCycle = useCallback(async () => {
    if (runningRef.current) return;
    runningRef.current = true;
    const s = settingsRef.current;
    const setStage = (stage: PipelineStage, status: StageStatus) =>
      setStages((prev) => ({ ...prev, [stage]: status }));

    try {
      setStages(EMPTY_STAGES);
      setStage("capture", "processing");
      const video = videoRef.current;
      const frame = video ? emotionService.captureFrame(video) : null;
      setStage("capture", "completed");

      setStage("detect", "processing");
      const detection = await emotionService.detectFace({
        frameDataUrl: frame?.dataUrl ?? null,
        backendUrl: s.backendUrl,
        demoMode: s.demoMode,
      });
      setStage("detect", "completed");

      setFaceCount(detection.faceCount);
      setBox(detection.box);
      setDetectionConfidence(detection.confidence);
      setFaceStatus(
        detection.faceCount === 0 ? "none" : detection.faceCount > 1 ? "multiple" : "single",
      );

      if (!detection.detected) {
        setStages((prev) => ({ ...prev, crop: "waiting", classify: "waiting" }));
        return;
      }

      setStage("crop", "processing");
      const face = await emotionService.preprocessFace({ video, box: detection.box });
      setStage("crop", "completed");
      setStage("resize", "completed");
      setStage("normalize", "completed");
      setStage("inputPrep", "completed");
      setFacePreview(face.dataUrl);

      setStage("features", "processing");
      // Hold a simulated emotion for a few cycles so demo output looks plausible.
      let preferred: Emotion | null = null;
      if (s.demoMode) {
        if (holdRef.current && holdRef.current.left > 0) {
          holdRef.current.left -= 1;
          preferred = holdRef.current.emotion;
        } else {
          const next = EMOTIONS[Math.floor(Math.random() * EMOTIONS.length)]!;
          holdRef.current = { emotion: next, left: 2 + Math.floor(Math.random() * 3) };
          preferred = next;
        }
      }
      const prediction = await emotionService.predictEmotion({
        faceDataUrl: face.dataUrl,
        backendUrl: s.backendUrl,
        demoMode: s.demoMode,
        preferredEmotion: preferred,
      });
      setStage("features", "completed");
      setStage("classify", "completed");

      setCurrentEmotion(prediction.emotion);
      setConfidence(prediction.confidence);
      setProbabilities(prediction.probabilities);

      const at = Date.now();
      const observation: Observation = {
        id: `${at}-${Math.random().toString(36).slice(2, 7)}`,
        at,
        time: timeLabel(at),
        emotion: prediction.emotion,
        confidence: prediction.confidence,
        status: prediction.source === "demo" ? "Simulated" : "Detected",
        source: prediction.source,
      };

      setHistory((prev) => {
        const next = emotionService.appendObservation(prev, observation);
        if (s.alertsEnabled) {
          const generated = emotionService.generateAlert({
            history: next,
            thresholdPct: s.confidenceThreshold,
            consecutive: s.consecutiveObservations,
          });
          if (generated) setAlert((current) => (current?.id === generated.id ? current : generated));
        }
        return next;
      });
    } catch {
      setStages(EMPTY_STAGES);
    } finally {
      runningRef.current = false;
    }
  }, []);

  useEffect(() => {
    if (!analyzing || !cameraActive) return;
    void runCycle();
    const id = window.setInterval(() => void runCycle(), settings.intervalMs);
    return () => window.clearInterval(id);
  }, [analyzing, cameraActive, runCycle, settings.intervalMs]);

  const toggleAnalyzing = useCallback(() => {
    setAnalyzing((prev) => {
      if (!prev) setSessionStart((s) => s ?? Date.now());
      return !prev;
    });
  }, []);

  const captureOnce = useCallback(async () => {
    setSessionStart((s) => s ?? Date.now());
    await runCycle();
  }, [runCycle]);

  const clearHistory = useCallback(() => {
    setHistory([]);
    setAlert(null);
  }, []);

  const value = useMemo<FerState>(
    () => ({
      settings,
      updateSettings,
      resetSettings,
      videoRef,
      cameraActive,
      cameraMessage,
      cameraError,
      startCamera,
      stopCamera,
      cameras,
      refreshCameras,
      analyzing,
      toggleAnalyzing,
      captureOnce,
      faceStatus,
      faceCount,
      box,
      detectionConfidence,
      facePreview,
      stages,
      currentEmotion,
      confidence,
      probabilities,
      inferenceSource: settings.demoMode || !settings.backendUrl ? "demo" : "backend",
      history,
      clearHistory,
      alert,
      dismissAlert: () => setAlert(null),
      sessionStart,
    }),
    [
      settings,
      updateSettings,
      resetSettings,
      cameraActive,
      cameraMessage,
      cameraError,
      startCamera,
      stopCamera,
      cameras,
      refreshCameras,
      analyzing,
      toggleAnalyzing,
      captureOnce,
      faceStatus,
      faceCount,
      box,
      detectionConfidence,
      facePreview,
      stages,
      currentEmotion,
      confidence,
      probabilities,
      history,
      clearHistory,
      alert,
      sessionStart,
    ],
  );

  return <FerContext.Provider value={value}>{children}</FerContext.Provider>;
}

export function useFer(): FerState {
  const ctx = useContext(FerContext);
  if (!ctx) throw new Error("useFer must be used inside <FerProvider>");
  return ctx;
}
