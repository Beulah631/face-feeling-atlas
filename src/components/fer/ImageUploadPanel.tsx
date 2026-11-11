import { useState, useRef, useCallback, useEffect } from "react";
import {
  AlertCircle,
  CheckCircle2,
  FileImage,
  GitCompare,
  ImageIcon,
  Loader2,
  RefreshCw,
  ScanFace,
  Trash2,
  Upload,
  UploadCloud,
  Cpu,
  Volume2,
  VolumeX,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useFer } from "@/lib/fer/FerProvider";
import { emotionService } from "@/lib/fer/emotionService";
import type { ModelComparisonResult } from "@/lib/fer/emotionService";
import { EMOTION_META } from "@/lib/fer/emotionMeta";
import { EMOTIONS, type BoundingBox, type Emotion } from "@/lib/fer/types";
import { announceEmotion, cancelVoiceAnnouncements } from "@/lib/fer/voiceAnnouncements";
import { prepareImagePayload } from "@/lib/fer/imagePayload";
import { Chip, Panel } from "./ui";

interface AnalysisResult {
  detected: boolean;
  faceCount: number;
  box: BoundingBox | null;
  confidence: number | null;
  emotion: Emotion | null;
  emotionConfidence: number | null;
  probabilities: Record<Emotion, number> | null;
  croppedFaceUrl: string | null;
  comparison: ModelComparisonResult | null;
  comparisonError: string | null;
}

const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/jpg", "image/webp"];
const MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024; // 10MB

function cropSquareFace(image: HTMLImageElement, box: BoundingBox, targetSize = 224) {
  const width = image.naturalWidth;
  const height = image.naturalHeight;
  const left = Math.max(0, Math.min(width, box.x * width));
  const top = Math.max(0, Math.min(height, box.y * height));
  const right = Math.max(left, Math.min(width, (box.x + box.width) * width));
  const bottom = Math.max(top, Math.min(height, (box.y + box.height) * height));
  const side = Math.min(Math.max(right - left, bottom - top), width, height);
  if (side <= 0) return null;

  const x = Math.max(0, Math.min(width - side, (left + right - side) / 2));
  const y = Math.max(0, Math.min(height - side, (top + bottom - side) / 2));
  const canvas = document.createElement("canvas");
  canvas.width = targetSize;
  canvas.height = targetSize;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  ctx.drawImage(image, x, y, side, side, 0, 0, targetSize, targetSize);
  return canvas.toDataURL("image/jpeg", 0.9);
}

export function ImageUploadPanel() {
  const { settings, inferenceSource, voiceAvailable } = useFer();

  const [imageFile, setImageFile] = useState<File | null>(null);
  const [imageDataUrl, setImageDataUrl] = useState<string | null>(null);
  const [imageDimensions, setImageDimensions] = useState<{ width: number; height: number } | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const [modelNotice, setModelNotice] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const imgElementRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => () => cancelVoiceAnnouncements(), []);

  /* ----------------------------------------------------------- file handling */

  const processFile = useCallback((file: File) => {
    setValidationError(null);
    setModelNotice(null);
    setResult(null);

    // Validate type
    const isAllowedType =
      ALLOWED_TYPES.includes(file.type) ||
      /\.(jpe?g|png|webp)$/i.test(file.name);

    if (!isAllowedType) {
      setValidationError("Please select a valid image file (JPG, JPEG, or PNG).");
      return;
    }

    // Validate size
    if (file.size > MAX_FILE_SIZE_BYTES) {
      setValidationError("Image file size exceeds the 10MB limit. Please choose a smaller image.");
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      setImageFile(file);
      setImageDataUrl(dataUrl);

      // Extract dimensions
      const img = new Image();
      img.onload = () => {
        setImageDimensions({ width: img.naturalWidth, height: img.naturalHeight });
      };
      img.onerror = () => {
        setImageFile(null);
        setImageDataUrl(null);
        setImageDimensions(null);
        setValidationError("This image could not be decoded. Please try another JPG, PNG, or WEBP file.");
      };
      img.src = dataUrl;
    };
    reader.onerror = () => {
      setValidationError("Failed to read the selected image file. Please try again.");
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      setIsDragging(false);
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        processFile(files[0]!);
      }
    },
    [processFile],
  );

  const handleFileInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files;
      if (files && files.length > 0) {
        processFile(files[0]!);
      }
    },
    [processFile],
  );

  const handleRemoveImage = useCallback(() => {
    setImageFile(null);
    setImageDataUrl(null);
    setImageDimensions(null);
    setResult(null);
    setValidationError(null);
    setModelNotice(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  const handleReplaceClick = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  /* ---------------------------------------------------------- image analysis */

  const handleAnalyzeImage = async () => {
    if (!imageDataUrl) return;

    setValidationError(null);
    setModelNotice(null);
    setIsAnalyzing(true);

    // Rule: Do not fabricate AI results.
    // If backend is not connected or in demoMode, honestly display model status.
    const backendUrl = settings.backendUrl.trim();
    if (settings.demoMode) {
      setIsAnalyzing(false);
      setModelNotice("Demo Mode is enabled for this app session. Turn it off in Settings to run real backend inference on uploaded images.");
      return;
    }
    if (!backendUrl) {
      setIsAnalyzing(false);
      setModelNotice("The Backend URL is empty in this app session. Set it in Settings, then retry this upload.");
      return;
    }

    try {
      const inferenceImage = await prepareImagePayload(imageDataUrl);
      // Step 1: Real YOLOv8 Face Detection
      const detection = await emotionService.detectFace({
        frameDataUrl: inferenceImage,
        backendUrl,
        demoMode: false,
      });

      if (!detection.detected || !detection.box) {
        setIsAnalyzing(false);
        setResult({
          detected: false,
          faceCount: 0,
          box: null,
          confidence: null,
          emotion: null,
          emotionConfidence: null,
          probabilities: null,
          croppedFaceUrl: null,
          comparison: null,
          comparisonError: null,
        });
        setModelNotice("No face detected in this image. Please upload a clear frontal face image.");
        return;
      }

      // Step 2: Crop Face Patch via offscreen canvas
      let croppedUrl: string | null = null;
      if (imgElementRef.current && detection.box) {
        croppedUrl = cropSquareFace(imgElementRef.current, detection.box);
      }
      if (!croppedUrl) throw new Error("The detected face could not be cropped from the uploaded image.");

      // Run the classifier endpoint directly so the upload flow is observable
      // as detect-face -> predict-emotion. Comparison is an additional request.
      const prediction = await emotionService.predictEmotion({
        faceDataUrl: croppedUrl,
        backendUrl,
        demoMode: false,
      });
      if (prediction.source === "backend" && settings.voiceAnnouncementsEnabled) {
        announceEmotion(prediction.emotion, settings.voiceName);
      }
      let comparison: ModelComparisonResult | null = null;
      let comparisonError: string | null = null;
      try {
        comparison = await emotionService.compareModels(croppedUrl, backendUrl);
        comparisonError = comparison.model_errors.cnn_baseline ?? null;
      } catch (err) {
        comparisonError = err instanceof Error ? err.message : String(err);
      }

      setResult({
        detected: true,
        faceCount: detection.faceCount,
        box: detection.box,
        confidence: detection.confidence,
        emotion: prediction.emotion,
        emotionConfidence: prediction.confidence,
        probabilities: prediction.probabilities,
        croppedFaceUrl: croppedUrl,
        comparison,
        comparisonError,
      });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setModelNotice(`Inference request failed: ${msg}`);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  const topPredictions = result?.probabilities
    ? EMOTIONS.map((emotion) => ({ emotion, probability: result.probabilities![emotion] }))
        .sort((a, b) => b.probability - a.probability)
        .slice(0, 3)
    : [];

  return (
    <div className="space-y-4">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        onChange={handleFileInputChange}
        className="hidden"
        aria-label="Upload Image File"
      />

      {/* Main Panel */}
      <div className="panel overflow-hidden">
        {/* Panel Header */}
        <div className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-3 border-b border-border px-4 py-3">
          <div className="flex min-w-0 items-center gap-2">
            <Upload className="h-4 w-4 shrink-0 text-primary" />
            <h3 className="truncate text-xs font-semibold tracking-widest text-muted-foreground uppercase">
              Upload Image Analysis
            </h3>
          </div>
          <Chip tone={imageDataUrl ? "success" : "muted"}>
            <FileImage className="h-3 w-3" />
            {imageDataUrl ? "Image Loaded" : "No File Selected"}
          </Chip>
        </div>

        {/* Upload Dropzone or Image Preview */}
        {!imageDataUrl ? (
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={cn(
              "relative flex flex-col items-center justify-center p-8 sm:p-12 text-center cursor-pointer transition-colors bg-background/50 grid-lines",
              "border-2 border-dashed m-4 rounded-xl",
              isDragging
                ? "border-primary bg-primary/10"
                : "border-border hover:border-primary/50 hover:bg-surface-2/40",
            )}
          >
            <div className="grid h-14 w-14 place-items-center rounded-2xl bg-primary/10 text-primary ring-1 ring-primary/25">
              <UploadCloud className="h-7 w-7" />
            </div>
            <h4 className="mt-4 font-display text-sm font-semibold text-foreground">
              Drag & drop your face image here
            </h4>
            <p className="mt-1 text-xs text-muted-foreground max-w-sm">
              Supports JPEG, JPG, PNG, and WEBP files up to 10MB.
            </p>
            <button
              type="button"
              className="mt-4 inline-flex items-center gap-2 rounded-lg bg-primary px-4 py-2 text-xs font-semibold tracking-wide text-primary-foreground uppercase shadow transition-opacity hover:opacity-90"
            >
              <Upload className="h-3.5 w-3.5" /> Browse Files
            </button>
          </div>
        ) : (
          <div className="p-4 space-y-4">
            {/* Image Preview with Bounding Box Overlay */}
            <div className="relative w-full overflow-hidden rounded-xl bg-background/80 grid-lines border border-border flex items-center justify-center min-h-[260px] max-h-[420px]">
              <div className="relative inline-block max-h-[400px] max-w-full">
                <img
                  ref={imgElementRef}
                  src={imageDataUrl}
                  alt="Uploaded face preview"
                  className="block max-h-[400px] max-w-full rounded-lg object-contain"
                />

                {/* Bounding box is positioned against the displayed image bounds. */}
                {result?.box && (
                  <div
                    className="pointer-events-none absolute rounded-md border-2 border-primary/95 shadow-[0_0_15px_rgba(34,211,238,0.35)]"
                    style={{
                      left: `${result.box.x * 100}%`,
                      top: `${result.box.y * 100}%`,
                      width: `${result.box.width * 100}%`,
                      height: `${result.box.height * 100}%`,
                    }}
                  >
                    <span className="absolute -top-6 left-0 whitespace-nowrap rounded bg-primary px-1.5 py-0.5 text-[10px] font-semibold tracking-wider text-primary-foreground uppercase">
                      YOLOv8 • Face {result.confidence ? `${(result.confidence * 100).toFixed(0)}%` : ""}
                    </span>
                  </div>
                )}
              </div>

              {/* High-tech corner indicators */}
              <div className="pointer-events-none absolute inset-3">
                {["left-0 top-0 border-l-2 border-t-2", "right-0 top-0 border-r-2 border-t-2", "left-0 bottom-0 border-l-2 border-b-2", "right-0 bottom-0 border-r-2 border-b-2"].map(
                  (pos) => (
                    <span
                      key={pos}
                      className={cn("absolute h-5 w-5 border-primary/70 rounded-[2px]", pos)}
                    />
                  ),
                )}
              </div>

              {/* Scanning Laser Beam during analysis */}
              {isAnalyzing && <span className="scanline" />}
            </div>

            {/* Metadata Bar */}
            <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border bg-surface-2/60 px-3 py-2 text-xs text-muted-foreground">
              <span className="truncate max-w-[200px] font-medium text-foreground">
                {imageFile?.name}
              </span>
              <div className="flex items-center gap-3 font-mono text-[11px]">
                {imageDimensions && (
                  <span>{imageDimensions.width} × {imageDimensions.height} px</span>
                )}
                {imageFile && <span>{formatFileSize(imageFile.size)}</span>}
              </div>
            </div>
          </div>
        )}

        {/* Validation Error Alert */}
        {validationError && (
          <div className="mx-4 mb-4 flex items-center gap-2 rounded-lg border border-destructive/30 bg-destructive/10 px-3.5 py-2.5 text-xs text-destructive">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{validationError}</span>
          </div>
        )}

        {/* Model Notice / Honest Limitation Message */}
        {modelNotice && (
          <div className="mx-4 mb-4 rounded-xl border border-warning/40 bg-warning/10 p-4 text-xs text-warning leading-relaxed">
            <div className="flex items-start gap-2.5">
              <AlertCircle className="h-4 w-4 shrink-0 mt-0.5" />
              <div>
                <p className="font-semibold">Model Status Notice</p>
                <p className="mt-1 text-muted-foreground">{modelNotice}</p>
              </div>
            </div>
          </div>
        )}

        {/* Action Controls Toolbar */}
        {imageDataUrl && (
          <div className="flex flex-wrap items-center gap-2 border-t border-border px-4 py-3 bg-surface/40">
            <button
              type="button"
              onClick={handleAnalyzeImage}
              disabled={isAnalyzing}
              className="inline-flex items-center gap-2 rounded-lg bg-primary px-3.5 py-2 text-xs font-semibold tracking-wide text-primary-foreground uppercase transition-opacity hover:opacity-90 disabled:opacity-50"
            >
              {isAnalyzing ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Analyzing Image...
                </>
              ) : (
                <>
                  <ScanFace className="h-3.5 w-3.5" />
                  Analyze Image
                </>
              )}
            </button>

            <button
              type="button"
              onClick={handleReplaceClick}
              disabled={isAnalyzing}
              className="inline-flex items-center gap-2 rounded-lg border border-border bg-surface-2 px-3.5 py-2 text-xs font-semibold tracking-wide uppercase transition-colors hover:bg-secondary disabled:opacity-50"
            >
              <RefreshCw className="h-3.5 w-3.5" /> Replace Image
            </button>

            <button
              type="button"
              onClick={handleRemoveImage}
              disabled={isAnalyzing}
              className="inline-flex items-center gap-2 rounded-lg border border-destructive/30 bg-surface-2 px-3.5 py-2 text-xs font-semibold tracking-wide text-destructive uppercase transition-colors hover:bg-destructive/10 disabled:opacity-50"
            >
              <Trash2 className="h-3.5 w-3.5" /> Remove Image
            </button>
          </div>
        )}
      </div>

      {/* Real Model Prediction Results Section (Shown only upon actual analysis) */}
      {result && result.detected && result.emotion && (
        <div className="grid gap-4 md:grid-cols-2">
          {/* Dominant Emotion Summary */}
          <Panel
            title="Image Emotion Result"
            icon={Cpu}
            badge={<Chip tone="success">Real Backend Inference</Chip>}
            footer="Generated from YOLOv8 cropped face patch using Swin Transformer feature extraction."
          >
            <div className="flex items-center gap-4">
              {result.croppedFaceUrl && (
                <img
                  src={result.croppedFaceUrl}
                  alt="Cropped face patch"
                  className="h-16 w-16 shrink-0 rounded-xl border border-border object-cover"
                />
              )}
              <div className="min-w-0">
                <p className="font-display text-2xl font-bold text-foreground">
                  {result.emotion.toUpperCase()}
                  {voiceAvailable ? (
                    settings.voiceAnnouncementsEnabled ? (
                      <Volume2
                        className="ml-2 inline h-4 w-4 text-primary"
                        aria-label="Voice announcements enabled"
                        title="Voice announcements enabled"
                      />
                    ) : (
                      <VolumeX
                        className="ml-2 inline h-4 w-4 text-muted-foreground"
                        aria-label="Voice announcements off"
                        title="Voice announcements off"
                      />
                    )
                  ) : null}
                </p>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Classifier Confidence:{" "}
                  <span className="font-semibold text-foreground">
                    {result.emotionConfidence
                      ? `${(result.emotionConfidence * 100).toFixed(1)}%`
                      : "—"}
                  </span>
                </p>
              </div>
            </div>
          </Panel>

          {/* 7-Class Probability Distribution */}
          <Panel
            title="Probability Breakdown"
            icon={ScanFace}
            footer="Normalized percentage distribution over the 7 emotion classes."
          >
            <div className="mb-4 grid grid-cols-3 gap-2" aria-label="Top 3 model predictions">
              {topPredictions.map(({ emotion, probability }, index) => (
                <div
                  key={emotion}
                  className="rounded-lg border border-border bg-surface-2/60 px-2.5 py-2"
                >
                  <p className="text-[10px] tracking-wider text-muted-foreground uppercase">
                    #{index + 1}
                  </p>
                  <p className="mt-0.5 truncate text-xs font-semibold text-foreground">{emotion}</p>
                  <p className="font-mono text-[11px] text-muted-foreground">
                    {probability.toFixed(1)}%
                  </p>
                </div>
              ))}
            </div>
            <div className="space-y-2 text-xs">
              {EMOTIONS.map((emotion) => {
                const val = result.probabilities?.[emotion] ?? 0;
                const meta = EMOTION_META[emotion];
                return (
                  <div key={emotion}>
                    <div className="mb-1 flex items-center justify-between text-[11px]">
                      <span className="flex items-center gap-1.5 font-medium">
                        <meta.icon className={cn("h-3 w-3", meta.tone)} />
                        {emotion}
                      </span>
                      <span className="font-mono text-muted-foreground">{val.toFixed(1)}%</span>
                    </div>
                    <div className="h-1.5 overflow-hidden rounded-full bg-surface-2">
                      <div
                        className={cn("h-full rounded-full transition-all duration-500", meta.bar)}
                        style={{ width: `${val}%` }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </Panel>

          <Panel
            title="Swin vs CNN Comparison"
            icon={GitCompare}
            className="md:col-span-2"
            footer={result.comparison?.comparison_summary ?? result.comparisonError ?? "Comparison results are unavailable."}
          >
            <div className="grid gap-3 sm:grid-cols-2">
              {([
                ["Swin Transformer", result.comparison?.swin_transformer, "swin_transformer"],
                ["CNN Baseline", result.comparison?.cnn_baseline, "cnn_baseline"],
              ] as const).map(([name, output, key]) => (
                <div key={key} className="rounded-lg border border-border bg-background/40 p-3">
                  <p className="text-[11px] font-semibold tracking-wider text-muted-foreground uppercase">{name}</p>
                  {output ? (
                    <div className="mt-2 flex items-baseline justify-between gap-3">
                      <span className="font-display text-lg font-bold">{output.emotion}</span>
                      <span className="font-mono text-xs text-muted-foreground">{(output.confidence * 100).toFixed(1)}% confidence</span>
                    </div>
                  ) : (
                    <p className="mt-2 text-xs text-warning">
                      {result.comparison?.model_errors[key] ?? result.comparisonError ?? "Model output unavailable."}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </Panel>
        </div>
      )}
    </div>
  );
}
