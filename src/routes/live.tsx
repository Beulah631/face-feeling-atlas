import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { Camera, Upload } from "lucide-react";
import { cn } from "@/lib/utils";
import { CameraPanel } from "@/components/fer/CameraPanel";
import { ImageUploadPanel } from "@/components/fer/ImageUploadPanel";
import {
  EmotionResultCard,
  FaceDetectionCard,
  PreprocessingCard,
  ProbabilityPanel,
  SwinTransformerCard,
} from "@/components/fer/AnalysisPanels";
import { AlertCard, EmotionTimeline, HistoryTable } from "@/components/fer/HistoryPanels";
import { PageHeader } from "@/components/fer/ui";

export const Route = createFileRoute("/live")({
  head: () => ({
    meta: [
      { title: "Live Analysis — Facial Emotion Recognition Demo" },
      {
        name: "description",
        content:
          "Live webcam demo screen: face detection overlay, preprocessing pipeline states, Swin Transformer panel, emotion probabilities and monitoring alerts.",
      },
      { property: "og:title", content: "Live Analysis — Facial Emotion Recognition Demo" },
      {
        property: "og:description",
        content:
          "Run the webcam pipeline demo with clearly-labelled simulated predictions or a connected inference backend.",
      },
    ],
  }),
  component: LiveAnalysis,
});

function LiveAnalysis() {
  const [mode, setMode] = useState<"camera" | "upload">("camera");

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Main demo screen"
        title="Live Analysis"
        description="Analyze facial expressions in real-time via live webcam feed, or upload static face images for model evaluation."
      />

      <div className="grid gap-4 xl:grid-cols-3">
        <div className="space-y-4 xl:col-span-2">
          {/* Analysis Input Mode Switcher */}
          <div className="flex items-center gap-2 rounded-xl border border-border bg-surface-2/60 p-1.5 w-fit">
            <button
              type="button"
              onClick={() => setMode("camera")}
              className={cn(
                "inline-flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-semibold uppercase tracking-wider transition-all",
                mode === "camera"
                  ? "bg-primary text-primary-foreground shadow"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <Camera className="h-3.5 w-3.5" />
              Live Camera
            </button>
            <button
              type="button"
              onClick={() => setMode("upload")}
              className={cn(
                "inline-flex items-center gap-2 rounded-lg px-3.5 py-1.5 text-xs font-semibold uppercase tracking-wider transition-all",
                mode === "upload"
                  ? "bg-primary text-primary-foreground shadow"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <Upload className="h-3.5 w-3.5" />
              Upload Image
            </button>
          </div>

          {/* Mode 1: Live Webcam Pipeline */}
          <div className={mode === "camera" ? "space-y-4" : "hidden"}>
            <CameraPanel />
            <div className="grid gap-4 md:grid-cols-2">
              <FaceDetectionCard />
              <PreprocessingCard />
            </div>
            <SwinTransformerCard />
          </div>

          {/* Mode 2: Static Image Upload & Analysis */}
          <div className={mode === "upload" ? "space-y-4" : "hidden"}>
            <ImageUploadPanel />
          </div>
        </div>

        <div className="space-y-4">
          <EmotionResultCard />
          <ProbabilityPanel />
          <AlertCard />
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <HistoryTable />
        <EmotionTimeline />
      </div>
    </div>
  );
}
