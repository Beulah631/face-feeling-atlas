import { useCallback, useEffect, useRef, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import {
  Camera,
  CheckCircle2,
  Cpu,
  Server,
  Settings as SettingsIcon,
  Volume2,
  XCircle,
} from "lucide-react";
import { PageHeader, Panel, Disclaimer, Chip } from "@/components/fer/ui";
import { useFer } from "@/lib/fer/FerProvider";
import { getEnglishVoices } from "@/lib/fer/voiceAnnouncements";
import {
  checkBackendHealth,
  type BackendConnectionStatus,
  type BackendHealthResult,
} from "@/lib/fer/backendHealth";

export const Route = createFileRoute("/settings")({
  head: () => ({
    meta: [
      { title: "Settings — Camera, Alerts and Demo Mode" },
      {
        name: "description",
        content:
          "Configure camera selection, confidence threshold, consecutive observation count, alert toggle, demo mode and the future inference backend URL.",
      },
      { property: "og:title", content: "Settings — Facial Emotion Recognition Prototype" },
      { property: "og:description", content: "Preferences persist locally in your browser." },
    ],
  }),
  component: SettingsPage,
});

function SettingsPage() {
  const { settings, updateSettings, resetSettings, cameras, refreshCameras, voiceAvailable } =
    useFer();
  const [englishVoices, setEnglishVoices] = useState<SpeechSynthesisVoice[]>([]);

  useEffect(() => {
    if (!voiceAvailable) return;
    const refreshVoices = () => setEnglishVoices(getEnglishVoices());
    refreshVoices();
    window.speechSynthesis.addEventListener("voiceschanged", refreshVoices);
    return () => window.speechSynthesis.removeEventListener("voiceschanged", refreshVoices);
  }, [voiceAvailable]);

  const [backendHealth, setBackendHealth] = useState<
    Omit<BackendHealthResult, "status"> & { status: BackendConnectionStatus }
  >({ status: "checking", detail: "Checking the configured backend health endpoint…" });
  const [checkingBackend, setCheckingBackend] = useState(false);
  const healthRequestId = useRef(0);
  const healthAbortController = useRef<AbortController | null>(null);

  const checkHealth = useCallback(async (url: string) => {
    const requestId = ++healthRequestId.current;
    healthAbortController.current?.abort();
    const controller = new AbortController();
    healthAbortController.current = controller;
    setBackendHealth({
      status: "checking",
      detail: "Checking the configured backend health endpoint…",
    });
    setCheckingBackend(true);
    try {
      const result = await checkBackendHealth(url, { signal: controller.signal, timeoutMs: 5000 });
      if (requestId === healthRequestId.current) setBackendHealth(result);
    } finally {
      if (requestId === healthRequestId.current) setCheckingBackend(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void checkHealth(settings.backendUrl), 250);
    return () => {
      window.clearTimeout(timer);
      healthRequestId.current += 1;
      healthAbortController.current?.abort();
    };
  }, [checkHealth, settings.backendUrl]);

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Configuration"
        title="Settings"
        description="All settings persist in your browser's local storage."
        actions={
          <button
            type="button"
            onClick={resetSettings}
            className="rounded-lg border border-border bg-surface-2 px-3.5 py-2 text-[10px] font-semibold tracking-widest uppercase hover:bg-secondary"
          >
            Reset defaults
          </button>
        }
      />

      <div className="grid gap-4 lg:grid-cols-2">
        <Panel title="Camera" icon={Camera}>
          <label className="block text-sm">
            <span className="text-[11px] tracking-widest text-muted-foreground uppercase">
              Camera selection
            </span>
            <select
              value={settings.cameraDeviceId}
              onChange={(e) => updateSettings({ cameraDeviceId: e.target.value })}
              className="mt-2 w-full rounded-lg border border-border bg-background/60 px-3 py-2 text-sm"
            >
              <option value="">Default camera</option>
              {cameras.map((device, i) => (
                <option key={device.deviceId || i} value={device.deviceId}>
                  {device.label || `Camera ${i + 1}`}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            onClick={() => void refreshCameras()}
            className="mt-3 rounded-lg border border-border bg-surface-2 px-3 py-2 text-[10px] font-semibold tracking-widest uppercase hover:bg-secondary"
          >
            Refresh device list
          </button>
          <p className="mt-3 text-[11px] text-muted-foreground">
            Device labels are only available after camera permission has been granted once.
          </p>
        </Panel>

        <Panel title="Analysis & alerts" icon={SettingsIcon}>
          <label className="block">
            <span className="text-[11px] tracking-widest text-muted-foreground uppercase">
              Confidence threshold: {settings.confidenceThreshold}%
            </span>
            <input
              type="range"
              min={50}
              max={99}
              value={settings.confidenceThreshold}
              onChange={(e) => updateSettings({ confidenceThreshold: Number(e.target.value) })}
              className="mt-2 w-full accent-[var(--color-primary)]"
            />
          </label>
          <label className="mt-4 block">
            <span className="text-[11px] tracking-widest text-muted-foreground uppercase">
              Consecutive observations: {settings.consecutiveObservations}
            </span>
            <input
              type="range"
              min={2}
              max={10}
              value={settings.consecutiveObservations}
              onChange={(e) => updateSettings({ consecutiveObservations: Number(e.target.value) })}
              className="mt-2 w-full accent-[var(--color-primary)]"
            />
          </label>
          <label className="mt-4 block">
            <span className="text-[11px] tracking-widest text-muted-foreground uppercase">
              Analysis interval: {(settings.intervalMs / 1000).toFixed(1)}s
            </span>
            <input
              type="range"
              min={1000}
              max={6000}
              step={500}
              value={settings.intervalMs}
              onChange={(e) => updateSettings({ intervalMs: Number(e.target.value) })}
              className="mt-2 w-full accent-[var(--color-primary)]"
            />
          </label>
          <label className="mt-4 flex items-center justify-between gap-3 rounded-lg border border-border bg-background/40 px-3 py-2.5 text-sm">
            <span>Alerts enabled</span>
            <input
              type="checkbox"
              checked={settings.alertsEnabled}
              onChange={(e) => updateSettings({ alertsEnabled: e.target.checked })}
              className="h-4 w-8 accent-[var(--color-primary)]"
            />
          </label>
          <label className="mt-3 flex items-center justify-between gap-3 rounded-lg border border-warning/30 bg-warning/5 px-3 py-2.5 text-sm">
            <span>Demo mode (simulated output)</span>
            <input
              type="checkbox"
              checked={settings.demoMode}
              onChange={(e) => updateSettings({ demoMode: e.target.checked })}
              className="h-4 w-8 accent-[var(--color-warning)]"
            />
          </label>
        </Panel>

        <Panel
          title="Voice announcements"
          icon={Volume2}
          footer="Live announcements require the same emotion to remain stable for the configured consecutive observations. Predictions below the voice confidence threshold are not spoken."
        >
          <label className="flex items-center justify-between gap-3 rounded-lg border border-border bg-background/40 px-3 py-2.5 text-sm">
            <span>Voice announcements enabled</span>
            <input
              type="checkbox"
              checked={settings.voiceAnnouncementsEnabled}
              onChange={(e) => updateSettings({ voiceAnnouncementsEnabled: e.target.checked })}
              className="h-4 w-8 accent-[var(--color-primary)]"
              disabled={!voiceAvailable}
            />
          </label>
          {!voiceAvailable ? (
            <p className="mt-3 text-xs text-warning">
              This browser does not provide speech synthesis. Voice announcements are unavailable.
            </p>
          ) : null}
          <label className="mt-4 block text-sm">
            <span className="text-[11px] tracking-widest text-muted-foreground uppercase">
              English browser voice
            </span>
            <select
              value={settings.voiceName}
              onChange={(e) => updateSettings({ voiceName: e.target.value })}
              disabled={!voiceAvailable || !settings.voiceAnnouncementsEnabled}
              className="mt-2 w-full rounded-lg border border-border bg-background/60 px-3 py-2 text-sm disabled:opacity-50"
            >
              <option value="">Browser default (English)</option>
              {englishVoices.map((voice) => (
                <option key={voice.voiceURI} value={voice.voiceURI}>
                  {voice.name} ({voice.lang})
                </option>
              ))}
            </select>
            {voiceAvailable && englishVoices.length === 0 ? (
              <span className="mt-1 block text-[11px] text-muted-foreground">
                No English voices are currently listed; the browser default will be used.
              </span>
            ) : null}
          </label>
          <label className="mt-4 block">
            <span className="text-[11px] tracking-widest text-muted-foreground uppercase">
              Live announcement cooldown: {settings.voiceCooldownSeconds}s
            </span>
            <input
              type="range"
              min={5}
              max={60}
              step={5}
              value={settings.voiceCooldownSeconds}
              onChange={(e) => updateSettings({ voiceCooldownSeconds: Number(e.target.value) })}
              disabled={!settings.voiceAnnouncementsEnabled}
              className="mt-2 w-full accent-[var(--color-primary)] disabled:opacity-50"
            />
          </label>
          <label className="mt-4 block">
            <span className="text-[11px] tracking-widest text-muted-foreground uppercase">
              Speak only at or above: {settings.voiceConfidenceThreshold}% model score
            </span>
            <input
              type="range"
              min={30}
              max={90}
              step={5}
              value={settings.voiceConfidenceThreshold}
              onChange={(e) => updateSettings({ voiceConfidenceThreshold: Number(e.target.value) })}
              disabled={!settings.voiceAnnouncementsEnabled}
              className="mt-2 w-full accent-[var(--color-primary)] disabled:opacity-50"
            />
          </label>
        </Panel>

        <Panel
          title="Model mode / inference backend"
          icon={Server}
          className="lg:col-span-2"
          badge={
            backendHealth?.status === "online" ? (
              <Chip tone="success">Backend Online</Chip>
            ) : backendHealth?.status === "checking" || checkingBackend ? (
              <Chip tone="muted">Checking…</Chip>
            ) : backendHealth?.status === "error" ? (
              <Chip tone="destructive">Backend Error</Chip>
            ) : (
              <Chip tone="warning">Backend Offline</Chip>
            )
          }
          footer="Use the local FastAPI URL during development. Vercel Services defaults to the same-origin API route; set VITE_BACKEND_URL only when using a separately hosted backend. Turn Demo Mode off to use real model inference."
        >
          <label className="block text-sm">
            <span className="text-[11px] tracking-widest text-muted-foreground uppercase">
              Backend base URL
            </span>
            <div className="mt-2 flex gap-2">
              <input
                type="url"
                value={settings.backendUrl}
                onChange={(e) => {
                  updateSettings({ backendUrl: e.target.value });
                  setBackendHealth({
                    status: "checking",
                    detail: "Backend URL changed; checking the saved URL…",
                  });
                }}
                placeholder="http://127.0.0.1:8000"
                className="w-full rounded-lg border border-border bg-background/60 px-3 py-2 font-mono text-sm"
              />
              <button
                type="button"
                onClick={() => void checkHealth(settings.backendUrl)}
                className="shrink-0 rounded-lg border border-border bg-surface-2 px-3.5 py-2 text-[10px] font-semibold tracking-widest uppercase hover:bg-secondary"
              >
                {checkingBackend ? "Check Again" : "Test Connection"}
              </button>
            </div>
          </label>

          {backendHealth ? (
            <p
              role={
                backendHealth.status === "error" || backendHealth.status === "offline"
                  ? "alert"
                  : "status"
              }
              className={`mt-3 text-xs ${backendHealth.status === "online" ? "text-success" : backendHealth.status === "error" ? "text-destructive" : "text-muted-foreground"}`}
            >
              {backendHealth.detail}
            </p>
          ) : null}

          {/* Model Status Cards */}
          {backendHealth?.status === "online" && backendHealth.data?.models && (
            <div className="mt-4 grid gap-3 sm:grid-cols-3">
              {Object.entries(backendHealth.data.models).map(([key, info]) => (
                <div key={key} className="rounded-lg border border-border bg-background/50 p-3">
                  <div className="flex items-center justify-between">
                    <p className="font-medium text-xs text-foreground">{info.name}</p>
                    {info.loaded ? (
                      <CheckCircle2 className="h-4 w-4 text-success" />
                    ) : (
                      <XCircle className="h-4 w-4 text-warning" />
                    )}
                  </div>
                  <p className="mt-1 text-[11px] text-muted-foreground">
                    Status:{" "}
                    <span
                      className={
                        info.loaded ? "text-success font-semibold" : "text-warning font-semibold"
                      }
                    >
                      {info.loaded
                        ? "Loaded & Ready"
                        : info.weights_exist
                          ? "Weights Found"
                          : "Weights Pending"}
                    </span>
                  </p>
                  {info.load_error && (
                    <p className="mt-1 text-[10px] text-destructive">{info.load_error}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </Panel>
      </div>

      <Disclaimer>
        With Demo Mode on, or with no backend URL configured, every prediction shown in this
        application is simulated and labelled as such.
      </Disclaimer>
    </div>
  );
}
