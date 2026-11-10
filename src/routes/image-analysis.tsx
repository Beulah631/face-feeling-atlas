import { createFileRoute } from "@tanstack/react-router";
import { ImageUploadPanel } from "@/components/fer/ImageUploadPanel";
import { PageHeader } from "@/components/fer/ui";

export const Route = createFileRoute("/image-analysis")({ component: ImageAnalysisPage });

function ImageAnalysisPage() {
  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Single image inference"
        title="Image Analysis"
        description="Upload a still image for facial emotion analysis. The existing image analysis workflow remains available here and in Live Analysis."
      />
      <ImageUploadPanel />
    </div>
  );
}
