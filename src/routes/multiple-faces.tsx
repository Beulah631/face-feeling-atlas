import { useRef, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { AlertTriangle, ScanFace, Upload } from "lucide-react";
import { useFer } from "@/lib/fer/FerProvider";
import {
  analyzeMultipleFaces,
  type AnalysisModel,
  type MultiFaceResponse,
} from "@/lib/fer/extendedAnalysisService";
import { EMOTIONS } from "@/lib/fer/types";
import { prepareImagePayload } from "@/lib/fer/imagePayload";
import { Chip, PageHeader, Panel } from "@/components/fer/ui";

export const Route = createFileRoute("/multiple-faces")({ component: MultipleFacesPage });

const MAX_IMAGE_BYTES = 10 * 1024 * 1024;

function readDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () =>
      typeof reader.result === "string"
        ? resolve(reader.result)
        : reject(new Error("Could not read the selected image."));
    reader.onerror = () => reject(new Error("Could not read the selected image."));
    reader.readAsDataURL(file);
  });
}

function MultipleFacesPage() {
  const { settings, inferenceSource } = useFer();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [model, setModel] = useState<AnalysisModel>("swin");
  const [result, setResult] = useState<MultiFaceResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [dragging, setDragging] = useState(false);

  async function chooseImage(next: File | undefined) {
    if (!next) return;
    setResult(null);
    setError(null);
    if (!/^image\/(jpeg|png|webp)$/.test(next.type)) {
      setFile(null);
      setPreview(null);
      setError("Choose a JPG, PNG, or WEBP image.");
      return;
    }
    if (next.size > MAX_IMAGE_BYTES) {
      setFile(null);
      setPreview(null);
      setError("Image must be 10 MB or smaller.");
      return;
    }
    try {
      setFile(next);
      setPreview(await readDataUrl(next));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Unable to read this image.");
    }
  }

  async function analyze() {
    if (!preview) return;
    if (inferenceSource !== "backend" || settings.demoMode) {
      setError(
        "Multiple-face inference requires Model Mode. Configure the backend URL and turn off Demo Mode in Settings.",
      );
      return;
    }
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const inferenceImage = await prepareImagePayload(preview);
      setResult(await analyzeMultipleFaces(inferenceImage, model, settings.backendUrl));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Multiple-face analysis failed.");
    } finally {
      setBusy(false);
    }
  }

  const scaleBox = (value: number) => `${Math.max(0, Math.min(100, value * 100))}%`;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Independent image workflow"
        title="Multiple Face Detection"
        description="Detect and classify every visible face independently using the connected YOLOv8 and emotion models. Face IDs exist only for this uploaded image; no identity recognition is performed."
      />
      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.25fr)_minmax(320px,0.75fr)]">
        <Panel title="Image and detections" icon={ScanFace}>
          <div
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setDragging(false);
              void chooseImage(event.dataTransfer.files[0]);
            }}
            className={`rounded-xl border border-dashed p-5 text-center transition-colors ${dragging ? "border-primary bg-primary/5" : "border-border bg-surface-2/40"}`}
          >
            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              className="hidden"
              onChange={(event) => void chooseImage(event.target.files?.[0])}
            />
            <Upload className="mx-auto h-6 w-6 text-primary" />
            <p className="mt-2 text-sm font-semibold">Drop an image here or choose a file</p>
            <p className="mt-1 text-xs text-muted-foreground">JPG, PNG or WEBP · up to 10 MB</p>
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="mt-3 rounded-lg border border-border px-3 py-2 text-xs font-semibold hover:bg-surface-2"
            >
              Choose image
            </button>
          </div>
          <div className="mt-4 flex flex-wrap items-center justify-between gap-3">
            <p className="min-w-0 truncate text-xs text-muted-foreground">
              {file
                ? `${file.name} · ${(file.size / 1024 / 1024).toFixed(2)} MB`
                : "No image selected"}
            </p>
            <div className="flex items-center gap-2">
              <label htmlFor="multi-model" className="sr-only">
                Classification model
              </label>
              <select
                id="multi-model"
                value={model}
                onChange={(event) => setModel(event.target.value as AnalysisModel)}
                className="rounded-lg border border-border bg-background px-2.5 py-2 text-xs"
              >
                <option value="swin">Swin Transformer</option>
                <option value="cnn">CNN</option>
                <option value="compare">Compare both</option>
              </select>
              <button
                type="button"
                disabled={!preview || busy}
                onClick={() => void analyze()}
                className="rounded-lg bg-primary px-4 py-2 text-xs font-semibold text-primary-foreground disabled:opacity-50"
              >
                {busy ? "Analyzing…" : "Analyze faces"}
              </button>
            </div>
          </div>
          {error ? (
            <p
              role="alert"
              className="mt-4 rounded-lg border border-destructive/30 bg-destructive/5 p-3 text-sm text-destructive"
            >
              {error}
            </p>
          ) : null}
          {preview ? (
            <div className="relative mx-auto mt-5 w-fit max-w-full overflow-hidden rounded-xl border border-border bg-black/20">
              <img
                src={preview}
                alt="Selected image for multiple-face analysis"
                className="block max-h-[70vh] max-w-full object-contain"
              />
              {result?.faces.map((face) => {
                const pred = face.predictions.swin ?? face.predictions.cnn;
                return (
                  <div
                    key={face.face_id}
                    className="absolute border-2 border-primary"
                    style={{
                      left: scaleBox(face.box.x),
                      top: scaleBox(face.box.y),
                      width: scaleBox(face.box.width),
                      height: scaleBox(face.box.height),
                    }}
                  >
                    <span
                      className="absolute left-0 top-0 -translate-y-full whitespace-nowrap rounded-t px-1.5 py-1 text-[10px] font-bold text-primary-foreground"
                      style={{ backgroundColor: "hsl(var(--primary))" }}
                    >
                      {face.face_id}
                      {pred
                        ? ` · ${pred.emotion} ${Math.round(pred.confidence * 100)}%`
                        : " · unavailable"}
                    </span>
                  </div>
                );
              })}
            </div>
          ) : null}
        </Panel>

        <Panel
          title="Per-face results"
          icon={ScanFace}
          badge={
            result ? (
              <Chip tone="primary">
                {result.face_count} face{result.face_count === 1 ? "" : "s"}
              </Chip>
            ) : undefined
          }
        >
          {inferenceSource !== "backend" || settings.demoMode ? (
            <div className="flex gap-2 rounded-lg border border-warning/30 bg-warning/5 p-3 text-xs text-warning">
              <AlertTriangle className="h-4 w-4 shrink-0" />
              This page requires real backend Model Mode; it never substitutes simulated
              predictions.
            </div>
          ) : null}
          {!result ? (
            <p className="text-sm text-muted-foreground">
              Face boxes and actual classifier results will appear here after analysis.
            </p>
          ) : null}
          {result && result.face_count === 0 ? (
            <div className="rounded-lg border border-warning/30 bg-warning/5 p-4 text-sm text-warning">
              No face was detected in this image.
            </div>
          ) : null}
          {result?.faces.length ? (
            <div className="space-y-3">
              {result.faces.map((face) => (
                <article
                  key={face.face_id}
                  className="rounded-xl border border-border bg-surface-2/35 p-4"
                >
                  <div className="flex items-center justify-between gap-2">
                    <h3 className="font-semibold">{face.face_id}</h3>
                    <Chip tone="muted">YOLO {Math.round(face.detector_confidence * 100)}%</Chip>
                  </div>
                  {(["swin", "cnn"] as const).map((modelKey) => {
                    const prediction = face.predictions[modelKey];
                    const modelError = face.model_errors[modelKey];
                    if (model !== "compare" && model !== modelKey) return null;
                    return (
                      <div key={modelKey} className="mt-3 border-t border-border pt-3">
                        <p className="text-[10px] font-semibold tracking-widest text-muted-foreground uppercase">
                          {modelKey === "swin" ? "Swin Transformer" : "CNN"}
                        </p>
                        {prediction ? (
                          <>
                            <p className="mt-1 text-sm font-bold">
                              {prediction.emotion}{" "}
                              <span className="font-normal text-muted-foreground">
                                · {Math.round(prediction.confidence * 100)}% model confidence
                              </span>
                            </p>
                            <div className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
                              {EMOTIONS.map((emotion) => (
                                <span key={emotion} className="flex justify-between gap-2">
                                  <span>{emotion}</span>
                                  <span>{prediction.probabilities[emotion]?.toFixed(1)}%</span>
                                </span>
                              ))}
                            </div>
                          </>
                        ) : (
                          <p className="mt-1 text-xs text-destructive">
                            {modelError ?? "This model did not return a prediction."}
                          </p>
                        )}
                      </div>
                    );
                  })}
                </article>
              ))}
            </div>
          ) : null}
          {result && Object.entries(result.model_errors).length > 0 ? (
            <p className="mt-3 text-xs text-destructive">
              {Object.entries(result.model_errors)
                .map(([name, message]) => `${name}: ${message}`)
                .join(" · ")}
            </p>
          ) : null}
        </Panel>
      </div>
    </div>
  );
}
