import { useEffect, useMemo, useRef, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { AlertTriangle, BarChart3, Download, FileVideo, Play, Square, Video } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { PageHeader, Panel, StatCard } from "@/components/fer/ui";
import { ExpressionReliabilityPanel } from "@/components/fer/ExpressionReliabilityPanel";
import { useFer } from "@/lib/fer/FerProvider";
import { EMOTIONS } from "@/lib/fer/types";
import {
  cancelVideoJob,
  getVideoJob,
  uploadVideoJob,
  type AnalysisModel,
  type VideoJob,
} from "@/lib/fer/extendedAnalysisService";

export const Route = createFileRoute("/video-analysis")({ component: VideoAnalysisPage });

const MAX_BYTES = 200 * 1024 * 1024;
const MAX_SECONDS = 600;
const ACCEPTED = new Set(["video/mp4", "video/quicktime", "video/webm"]);
const formatTime = (seconds: number) =>
  `${Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0")}:${Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0")}.${Math.floor((seconds % 1) * 10)}`;

function VideoAnalysisPage() {
  const { settings, inferenceSource } = useFer();
  const inputRef = useRef<HTMLInputElement>(null);
  const playerRef = useRef<HTMLVideoElement>(null);
  const pollRef = useRef<number | null>(null);
  const uploadRef = useRef<{ abort: () => void } | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [duration, setDuration] = useState<number | null>(null);
  const [intervalMs, setIntervalMs] = useState(1000);
  const [model, setModel] = useState<AnalysisModel>("swin");
  const [job, setJob] = useState<VideoJob | null>(null);
  const [jobId, setJobId] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  useEffect(() => {
    if (!jobId || !settings.backendUrl || !busy) return;
    let active = true;
    const poll = async () => {
      try {
        const current = await getVideoJob(jobId, settings.backendUrl);
        if (!active) return;
        setJob(current);
        if (["completed", "failed", "cancelled"].includes(current.status)) {
          setBusy(false);
          if (pollRef.current !== null) window.clearInterval(pollRef.current);
          pollRef.current = null;
        }
      } catch (cause) {
        if (!active) return;
        setError(cause instanceof Error ? cause.message : "Could not read video job status.");
        setBusy(false);
      }
    };
    void poll();
    pollRef.current = window.setInterval(() => void poll(), 1000);
    return () => {
      active = false;
      if (pollRef.current !== null) window.clearInterval(pollRef.current);
      pollRef.current = null;
    };
  }, [jobId, settings.backendUrl, busy]);

  useEffect(
    () => () => {
      uploadRef.current?.abort();
    },
    [],
  );

  function selectVideo(next?: File) {
    if (!next) return;
    setError(null);
    setJob(null);
    setJobId(null);
    setDuration(null);
    if (!ACCEPTED.has(next.type) && !/\.(mp4|mov|webm)$/i.test(next.name)) {
      setFile(null);
      setError("Choose an MP4, MOV, or WebM video.");
      return;
    }
    if (next.size > MAX_BYTES) {
      setFile(null);
      setError("Video must be 200 MB or smaller.");
      return;
    }
    setFile(next);
  }

  function metadataReady() {
    const mediaDuration = playerRef.current?.duration;
    if (!mediaDuration || !Number.isFinite(mediaDuration)) {
      setError("The browser could not read the video duration.");
      setDuration(null);
      return;
    }
    setDuration(mediaDuration);
    setError(
      mediaDuration > MAX_SECONDS
        ? "Video duration exceeds the backend limit of 10 minutes."
        : null,
    );
  }

  async function startAnalysis() {
    if (!file || duration === null || duration > MAX_SECONDS) return;
    if (inferenceSource !== "backend" || settings.demoMode) {
      setError(
        "Video analysis requires Model Mode. Set the backend URL and turn off Demo Mode in Settings.",
      );
      return;
    }
    setBusy(true);
    setError(null);
    setJob(null);
    setJobId(null);
    setUploadProgress(0);
    try {
      const task = uploadVideoJob(file, model, intervalMs, settings.backendUrl, setUploadProgress);
      uploadRef.current = task;
      const created = await task.promise;
      uploadRef.current = null;
      setJobId(created.job_id);
      setUploadProgress(100);
    } catch (cause) {
      uploadRef.current = null;
      if (cause instanceof DOMException && cause.name === "AbortError") {
        setBusy(false);
        setError("Upload cancelled.");
        return;
      }
      setBusy(false);
      setError(cause instanceof Error ? cause.message : "Video upload failed.");
    }
  }

  async function stopAnalysis() {
    uploadRef.current?.abort();
    uploadRef.current = null;
    if (jobId) {
      setJob((current) => (current ? { ...current, status: "cancelling" } : current));
      try {
        await cancelVideoJob(jobId, settings.backendUrl);
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Unable to cancel the analysis job.");
      }
      return;
    }
    setBusy(false);
  }

  const selectedModel = model === "compare" ? "swin" : model;
  const distribution = useMemo(() => {
    const counts = Object.fromEntries(EMOTIONS.map((emotion) => [emotion, 0])) as Record<
      (typeof EMOTIONS)[number],
      number
    >;
    for (const event of job?.events ?? []) {
      const prediction = event.predictions[selectedModel];
      if (prediction) counts[prediction.emotion] += 1;
    }
    const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
    return EMOTIONS.map((emotion) => ({
      emotion,
      count: counts[emotion],
      percent: total ? (counts[emotion] / total) * 100 : 0,
    }));
  }, [job?.events, selectedModel]);

  const timelineData = useMemo(() => {
    const grouped = new Map<
      number,
      { timestamp: number; count: number } & Record<string, number>
    >();
    for (const event of job?.events ?? []) {
      const prediction = event.predictions[selectedModel];
      if (!prediction) continue;
      const timestamp = Number(event.timestamp_seconds.toFixed(2));
      const point = grouped.get(timestamp) ?? { timestamp, count: 0 };
      point.count += 1;
      for (const emotion of EMOTIONS)
        point[emotion] = (point[emotion] ?? 0) + prediction.probabilities[emotion];
      grouped.set(timestamp, point);
    }
    return [...grouped.values()]
      .sort((a, b) => a.timestamp - b.timestamp)
      .map((point) => {
        const result: Record<string, number> = { timestamp: point.timestamp };
        for (const emotion of EMOTIONS) result[emotion] = (point[emotion] ?? 0) / point.count;
        return result;
      });
  }, [job?.events, selectedModel]);

  function exportCsv() {
    if (!job) return;
    const rows = [
      [
        "timestamp_seconds",
        "timestamp",
        "face_id",
        "x",
        "y",
        "width",
        "height",
        "detector_confidence",
        "model",
        "emotion",
        "confidence",
        ...EMOTIONS.map((emotion) => `${emotion}_probability`),
      ],
    ];
    for (const event of job.events)
      for (const modelKey of ["swin", "cnn"] as const) {
        const prediction = event.predictions[modelKey];
        if (!prediction || (job.model !== "compare" && job.model !== modelKey)) continue;
        rows.push([
          String(event.timestamp_seconds),
          formatTime(event.timestamp_seconds),
          event.face_id,
          String(event.box.x),
          String(event.box.y),
          String(event.box.width),
          String(event.box.height),
          String(event.detector_confidence),
          modelKey,
          prediction.emotion,
          String(prediction.confidence),
          ...EMOTIONS.map((emotion) => String(prediction.probabilities[emotion])),
        ]);
      }
    const csv = rows
      .map((row) => row.map((cell) => `"${cell.replaceAll('"', '""')}"`).join(","))
      .join("\r\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv;charset=utf-8" }));
    const link = document.createElement("a");
    link.href = url;
    link.download = `${job.file_name.replace(/\.[^.]+$/, "")}-emotion-analysis.csv`;
    link.click();
    URL.revokeObjectURL(url);
  }

  function exportPdf() {
    if (!job) return;
    const reportWindow = window.open("", "_blank");
    if (!reportWindow) {
      setError("The browser blocked the report window. Allow pop-ups for this site and try again.");
      return;
    }
    reportWindow.opener = null;
    const escapeHtml = (value: string) =>
      value.replace(
        /[&<>"]/g,
        (character) =>
          ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[character] ?? character,
      );
    const rows = job.events
      .flatMap((event) =>
        (["swin", "cnn"] as const).flatMap((modelKey) => {
          const prediction = event.predictions[modelKey];
          if (!prediction || (job.model !== "compare" && job.model !== modelKey)) return [];
          return [
            `<tr><td>${formatTime(event.timestamp_seconds)}</td><td>${escapeHtml(event.face_id)}</td><td>${modelKey === "swin" ? "Swin Transformer" : "CNN"}</td><td>${escapeHtml(prediction.emotion)}</td><td>${(prediction.confidence * 100).toFixed(1)}%</td></tr>`,
          ];
        }),
      )
      .join("");
    reportWindow.document.write(
      `<!doctype html><html><head><title>Video Emotion Analysis Report</title><style>body{font:14px Arial,sans-serif;margin:32px;color:#17202a}h1{font-size:22px}p{color:#445}table{border-collapse:collapse;width:100%;margin-top:20px}th,td{border:1px solid #ccd;padding:8px;text-align:left}th{background:#eef2f5}@media print{button{display:none}}</style></head><body><h1>Video Emotion Analysis Report</h1><p>File: ${escapeHtml(job.file_name)} · Duration: ${job.duration_seconds.toFixed(2)} sec · Status: ${escapeHtml(job.status)}</p><p>Frames analyzed: ${job.frames_processed}/${job.sampled_frames_total} · Face detections: ${job.face_detections} · Processing: ${job.processing_seconds?.toFixed(2) ?? "in progress"} sec</p><p>Rows below contain only predictions returned by the backend model.</p><button onclick="window.print()">Print / Save as PDF</button><table><thead><tr><th>Timestamp</th><th>Temporary Face ID</th><th>Model</th><th>Predicted emotion</th><th>Confidence</th></tr></thead><tbody>${rows || '<tr><td colspan="5">No face predictions were returned.</td></tr>'}</tbody></table><script>window.addEventListener('load',()=>window.print())</script></body></html>`,
    );
    reportWindow.document.close();
  }

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Sampled frame inference"
        title="Video Analysis"
        description="Upload a clip for timestamped YOLOv8 face detection and real model emotion classification. Frames are sampled at your chosen interval and processed by a FastAPI background job."
      />
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(300px,0.68fr)]">
        <Panel title="Video input" icon={FileVideo}>
          <div
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              selectVideo(event.dataTransfer.files[0]);
            }}
            className={`rounded-xl border border-dashed p-5 text-center ${dragging ? "border-primary bg-primary/5" : "border-border bg-surface-2/40"}`}
          >
            <input
              ref={inputRef}
              type="file"
              accept="video/mp4,video/quicktime,video/webm,.mp4,.mov,.webm"
              className="hidden"
              onChange={(event) => selectVideo(event.target.files?.[0])}
            />
            <Video className="mx-auto h-6 w-6 text-primary" />
            <p className="mt-2 text-sm font-semibold">Drop a video or choose a file</p>
            <p className="mt-1 text-xs text-muted-foreground">
              MP4, MOV, WebM · up to 200 MB · up to 10 minutes
            </p>
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="mt-3 rounded-lg border border-border px-3 py-2 text-xs font-semibold hover:bg-surface-2"
            >
              Choose video
            </button>
          </div>
          {previewUrl ? (
            <video
              ref={playerRef}
              src={previewUrl}
              controls
              onLoadedMetadata={metadataReady}
              onError={() =>
                setError(
                  "The browser cannot preview this video format. Backend codec support may also be limited.",
                )
              }
              className="mt-4 max-h-[440px] w-full rounded-xl border border-border bg-black"
            />
          ) : null}
          <div className="mt-4 grid gap-3 sm:grid-cols-2">
            <div>
              <label
                className="mb-1 block text-[10px] font-semibold tracking-widest text-muted-foreground uppercase"
                htmlFor="frame-interval"
              >
                Frame sampling interval
              </label>
              <select
                id="frame-interval"
                value={intervalMs}
                disabled={busy}
                onChange={(event) => setIntervalMs(Number(event.target.value))}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs"
              >
                <option value={500}>Every 0.5 second</option>
                <option value={1000}>Every 1 second</option>
                <option value={2000}>Every 2 seconds</option>
                <option value={5000}>Every 5 seconds</option>
                <option value={10000}>Every 10 seconds</option>
              </select>
            </div>
            <div>
              <label
                className="mb-1 block text-[10px] font-semibold tracking-widest text-muted-foreground uppercase"
                htmlFor="video-model"
              >
                Emotion model
              </label>
              <select
                id="video-model"
                value={model}
                disabled={busy}
                onChange={(event) => setModel(event.target.value as AnalysisModel)}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 text-xs"
              >
                <option value="swin">Swin Transformer</option>
                <option value="cnn">CNN</option>
                <option value="compare">Compare Swin + CNN</option>
              </select>
            </div>
          </div>
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
            <p className="text-xs text-muted-foreground">
              {file
                ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(1)} MB · ${duration ? `${duration.toFixed(1)} sec` : "reading duration…"}`
                : "No video selected"}
            </p>
            <button
              type="button"
              disabled={!file || !duration || duration > MAX_SECONDS || busy}
              onClick={() => void startAnalysis()}
              className="rounded-lg bg-primary px-4 py-2 text-xs font-semibold text-primary-foreground disabled:opacity-50"
            >
              {busy ? "Processing…" : "Start analysis"}
            </button>
          </div>
          {uploadProgress !== null ? (
            <div className="mt-3">
              <div className="mb-1 flex justify-between text-[11px] text-muted-foreground">
                <span>{jobId ? "Analysis job" : "Upload progress"}</span>
                <span>
                  {job
                    ? `${job.frames_processed}/${job.sampled_frames_total} frames`
                    : `${uploadProgress}%`}
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-surface-2">
                <div
                  className="h-full rounded-full bg-primary transition-[width]"
                  style={{
                    width: `${job ? (job.frames_processed / Math.max(1, job.sampled_frames_total)) * 100 : uploadProgress}%`,
                  }}
                />
              </div>
            </div>
          ) : null}
          {busy ? (
            <button
              type="button"
              onClick={() => void stopAnalysis()}
              className="mt-3 inline-flex items-center gap-2 rounded-lg border border-destructive/30 px-3 py-2 text-xs text-destructive"
            >
              <Square className="h-3.5 w-3.5" />
              Cancel analysis
            </button>
          ) : null}
          {job ? (
            <p className="mt-3 text-xs text-muted-foreground">
              Status: <span className="font-semibold capitalize text-foreground">{job.status}</span>
              {job.current_timestamp_seconds !== null
                ? ` · at ${formatTime(job.current_timestamp_seconds)}`
                : ""}
            </p>
          ) : null}
          {error || job?.error ? (
            <p
              role="alert"
              className="mt-3 rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"
            >
              {error ?? job?.error}
            </p>
          ) : null}
        </Panel>
        <Panel title="Actual prediction distribution" icon={BarChart3}>
          <p className="mb-4 text-xs text-muted-foreground">
            Percentages are calculated from successful model predictions returned by this job.
            Frames without detected faces or successful predictions are excluded.
          </p>
          <div className="h-72 w-full">
            {job?.events.length ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={distribution} margin={{ top: 8, right: 8, bottom: 4, left: -18 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="emotion" tick={{ fontSize: 10 }} interval={0} />
                  <YAxis tick={{ fontSize: 10 }} allowDecimals={false} />
                  <Tooltip
                    formatter={(value, name, item) => [
                      `${value} (${item.payload.percent.toFixed(1)}%)`,
                      "Predictions",
                    ]}
                  />
                  <Bar dataKey="count" fill="hsl(var(--primary))" radius={[5, 5, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="grid h-full place-items-center rounded-xl border border-dashed border-border text-xs text-muted-foreground">
                Charts populate from completed frame predictions.
              </div>
            )}
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2">
            {distribution.map(({ emotion, count, percent }) => (
              <div key={emotion} className="rounded-lg bg-surface-2/60 px-3 py-2 text-xs">
                <span className="font-medium">{emotion}</span>
                <span className="float-right text-muted-foreground">
                  {percent.toFixed(1)}% ({count})
                </span>
              </div>
            ))}
          </div>
        </Panel>
      </div>

      {job ? (
        <>
          <div className="grid gap-3 sm:grid-cols-3">
            <StatCard
              label="Sampled frames analyzed"
              value={`${job.frames_processed} / ${job.sampled_frames_total}`}
              hint={`Sampling interval: ${job.interval_ms} ms`}
              icon={Play}
            />
            <StatCard
              label="Face detections"
              value={String(job.face_detections)}
              hint={`${new Set(job.events.map((event) => event.face_id)).size} temporary track IDs`}
              icon={Video}
              tone="accent"
            />
            <StatCard
              label="Processing time"
              value={
                job.processing_seconds === null
                  ? "In progress"
                  : `${job.processing_seconds.toFixed(2)} sec`
              }
              hint={`${job.events.length} timestamped face predictions`}
              icon={BarChart3}
              tone="success"
            />
          </div>
          <Panel
            title="Timestamped emotion timeline"
            icon={Video}
            badge={
              <span className="text-[10px] text-muted-foreground">
                {job.events.length} face events
              </span>
            }
          >
            {job.events.length ? (
              <>
                <div className="mb-3 h-72 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart
                      data={timelineData}
                      margin={{ top: 8, right: 10, bottom: 4, left: -15 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis
                        dataKey="timestamp"
                        type="number"
                        domain={[0, job.duration_seconds]}
                        tickFormatter={(value) => formatTime(Number(value))}
                        tick={{ fontSize: 10 }}
                      />
                      <YAxis domain={[0, 100]} tick={{ fontSize: 10 }} />
                      <Tooltip
                        labelFormatter={(value) => formatTime(Number(value))}
                        formatter={(value) => [`${Number(value).toFixed(1)}%`, "Class probability"]}
                      />
                      <Legend wrapperStyle={{ fontSize: 10 }} />
                      {EMOTIONS.map((emotion, index) => (
                        <Line
                          key={emotion}
                          type="monotone"
                          dataKey={emotion}
                          stroke={`hsl(${(index * 49 + 175) % 360} 68% 52%)`}
                          dot={{ r: 2 }}
                          connectNulls
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                <p className="mb-3 text-[11px] text-muted-foreground">
                  Average model class probabilities per sampled timestamp across detected faces.
                </p>
                <div className="relative mb-5 h-10 rounded-lg border border-border bg-surface-2/50">
                  {job.events.map((event, index) => (
                    <button
                      key={`${event.sample_index}-${event.face_id}-${index}`}
                      type="button"
                      title={`${formatTime(event.timestamp_seconds)} · ${event.face_id} · ${event.predictions[selectedModel]?.emotion ?? "unavailable"}`}
                      onClick={() => {
                        if (playerRef.current)
                          playerRef.current.currentTime = event.timestamp_seconds;
                      }}
                      className="absolute top-0 h-full w-1.5 rounded bg-primary hover:bg-accent"
                      style={{
                        left: `${Math.min(99.5, (event.timestamp_seconds / Math.max(job.duration_seconds, 0.01)) * 100)}%`,
                      }}
                    />
                  ))}
                </div>
                <div className="grid gap-2 md:grid-cols-2 xl:grid-cols-3">
                  {job.events.map((event, index) => (
                    <article
                      key={`${event.sample_index}-${event.face_id}-${index}`}
                      className="rounded-lg border border-border p-3"
                    >
                      <p className="text-xs font-semibold">
                        {formatTime(event.timestamp_seconds)}{" "}
                        <span className="font-normal text-muted-foreground">· {event.face_id}</span>
                      </p>
                      {(["swin", "cnn"] as const).map((modelKey) => {
                        const prediction = event.predictions[modelKey];
                        if (!prediction || (job.model !== "compare" && job.model !== modelKey))
                          return null;
                        return (
                          <p key={modelKey} className="mt-1 text-xs text-muted-foreground">
                            {modelKey === "swin" ? "Swin" : "CNN"}:{" "}
                            <span className="font-semibold text-foreground">
                              {prediction.emotion}
                            </span>{" "}
                            · {(prediction.confidence * 100).toFixed(1)}%
                          </p>
                        );
                      })}
                    </article>
                  ))}
                </div>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">
                No face detections or successful predictions have been returned yet.
              </p>
            )}
          </Panel>
          {job.status === "completed" ? (
            <Panel title="Data exports" icon={Download}>
              <div className="flex flex-wrap gap-3">
                <button
                  type="button"
                  onClick={exportCsv}
                  disabled={!job.events.length}
                  className="inline-flex items-center gap-2 rounded-lg border border-border px-4 py-2 text-xs font-semibold disabled:opacity-50"
                >
                  <Download className="h-4 w-4" />
                  Download CSV
                </button>
                <button
                  type="button"
                  onClick={exportPdf}
                  className="inline-flex items-center gap-2 rounded-lg border border-border px-4 py-2 text-xs font-semibold"
                >
                  <Download className="h-4 w-4" />
                  Print / Save PDF
                </button>
              </div>
              <p className="mt-3 text-xs text-muted-foreground">
                The summary and exports contain only actual backend detections and predictions.
              </p>
            </Panel>
          ) : null}
        </>
      ) : null}
      {job && ["queued", "processing", "cancelling"].includes(job.status) ? (
        <Panel title="Expression Reliability Analysis" icon={BarChart3}>
          <p className="text-sm text-muted-foreground">
            Reliability metrics are calculated from the completed sampled-frame outputs. No
            reliability status is assigned while inference is still running.
          </p>
        </Panel>
      ) : null}
      {job?.status === "completed" && job.reliability ? (
        <ExpressionReliabilityPanel report={job.reliability} />
      ) : null}
      {job?.status === "completed" && !job.reliability ? (
        <p
          role="alert"
          className="rounded-lg border border-warning/30 bg-warning/5 p-3 text-sm text-warning"
        >
          The backend completed this analysis but did not include a reliability report. Confirm that
          the backend is running the updated video-analysis code.
        </p>
      ) : null}
      {inferenceSource !== "backend" || settings.demoMode ? (
        <p className="flex gap-2 rounded-xl border border-warning/30 bg-warning/5 p-4 text-xs text-warning">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          Real video analysis needs Model Mode and a reachable FastAPI backend. This page does not
          show simulated predictions.
        </p>
      ) : null}
    </div>
  );
}
