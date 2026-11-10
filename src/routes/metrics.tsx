import { useEffect, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { BarChart3, Clock, Cpu, GitCompare, Grid3x3, Percent, Target } from "lucide-react";
import { Chip, Disclaimer, PageHeader, Panel } from "@/components/fer/ui";
import { EMOTIONS } from "@/lib/fer/types";
import { useFer } from "@/lib/fer/FerProvider";

export const Route = createFileRoute("/metrics")({
  head: () => ({
    meta: [
      { title: "Evaluation Metrics — Swin Transformer vs. CNN Baseline" },
      {
        name: "description",
        content:
          "Quantitative evaluation metrics: Accuracy, Precision, Recall, F1-score and confusion matrices measured on the FER2013 held-out test split.",
      },
      { property: "og:title", content: "Evaluation Metrics — FER Project" },
      {
        property: "og:description",
        content: "Real benchmark metrics comparing Swin Transformer against CNN Baseline on FER2013.",
      },
    ],
  }),
  component: Metrics,
});

interface PerClassMetric {
  label: string;
  precision: number;
  recall: number;
  f1_score: number;
  support: number;
}

interface ModelEvaluation {
  model_name: string;
  dataset: string;
  samples_evaluated: number;
  accuracy: number;
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  per_class: PerClassMetric[];
  confusion_matrix: number[][];
  inference_ms_per_image: number;
}

function Metrics() {
  const { settings } = useFer();
  const [metrics, setMetrics] = useState<{
    swin_transformer?: ModelEvaluation;
    cnn_baseline?: ModelEvaluation;
  } | null>(null);
  const [selectedModel, setSelectedModel] = useState<"swin" | "cnn">("swin");

  useEffect(() => {
    const fetchMetrics = async () => {
      const url = settings.backendUrl || "http://127.0.0.1:8000";
      try {
        const res = await fetch(`${url.replace(/\/$/, "")}/api/evaluation-metrics`);
        if (res.ok) {
          const data = await res.json();
          setMetrics(data);
        }
      } catch {
        // Leave metrics as null if backend not reachable yet
      }
    };
    void fetchMetrics();
  }, [settings.backendUrl]);

  const swinData = metrics?.swin_transformer;
  const cnnData = metrics?.cnn_baseline;
  const activeData = selectedModel === "swin" ? swinData : cnnData;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Evaluation & Comparison"
        title="Evaluation Metrics"
        description="Evaluation values reported by the backend's metric files. The labeled evaluation data is not included in this project, so the results cannot be reproduced here."
      />

      {/* Model Comparison Summary Table */}
      <Panel
        title="Model Comparison Summary (Backend Reported Values)"
        icon={GitCompare}
        badge={<Chip tone="warning">Reported, Not Reproduced</Chip>}
        footer="Values are read from backend/weights/*_evaluation_results.json. The labeled test set and evaluation script are not included, so these figures have not been independently reproduced."
      >
        <div className="overflow-x-auto">
          <table className="w-full min-w-[600px] text-sm">
            <thead>
              <tr className="border-b border-border text-left text-[11px] tracking-widest text-muted-foreground uppercase">
                <th className="pb-2">Evaluation Metric</th>
                <th className="pb-2 text-primary font-semibold">Swin Transformer (Proposed)</th>
                <th className="pb-2 font-semibold">CNN Baseline (Reference)</th>
                <th className="pb-2 font-semibold">Difference / Takeaway</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b border-border/60">
                <td className="py-2.5 text-muted-foreground">Accuracy</td>
                <td className="py-2.5 font-mono font-bold text-primary">
                  {swinData ? `${(swinData.accuracy * 100).toFixed(2)}%` : "Unavailable"}
                </td>
                <td className="py-2.5 font-mono">
                  {cnnData ? `${(cnnData.accuracy * 100).toFixed(2)}%` : "Unavailable"}
                </td>
                <td className="py-2.5 text-xs text-muted-foreground">
                  Reported values only; compare on the same verified test split before drawing conclusions.
                </td>
              </tr>
              <tr className="border-b border-border/60">
                <td className="py-2.5 text-muted-foreground">Macro F1-Score</td>
                <td className="py-2.5 font-mono font-bold text-primary">
                  {swinData ? swinData.macro_f1.toFixed(4) : "Unavailable"}
                </td>
                <td className="py-2.5 font-mono">
                  {cnnData ? cnnData.macro_f1.toFixed(4) : "Unavailable"}
                </td>
                <td className="py-2.5 text-xs text-muted-foreground">
                  Balanced harmonic mean across all 7 emotions
                </td>
              </tr>
              <tr className="border-b border-border/60">
                <td className="py-2.5 text-muted-foreground">Macro Precision</td>
                <td className="py-2.5 font-mono font-bold text-primary">
                  {swinData ? swinData.macro_precision.toFixed(4) : "Unavailable"}
                </td>
                <td className="py-2.5 font-mono">
                  {cnnData ? cnnData.macro_precision.toFixed(4) : "Unavailable"}
                </td>
                <td className="py-2.5 text-xs text-muted-foreground">
                  Positive predictive accuracy across expressions
                </td>
              </tr>
              <tr className="border-b border-border/60">
                <td className="py-2.5 text-muted-foreground">Macro Recall</td>
                <td className="py-2.5 font-mono font-bold text-primary">
                  {swinData ? swinData.macro_recall.toFixed(4) : "Unavailable"}
                </td>
                <td className="py-2.5 font-mono">
                  {cnnData ? cnnData.macro_recall.toFixed(4) : "Unavailable"}
                </td>
                <td className="py-2.5 text-xs text-muted-foreground">
                  True positive detection rate across expressions
                </td>
              </tr>
              <tr>
                <td className="py-2.5 text-muted-foreground">Inference Time (CPU)</td>
                <td className="py-2.5 font-mono font-bold text-primary">
                  {swinData ? `${swinData.inference_ms_per_image.toFixed(1)} ms/img` : "Unavailable"}
                </td>
                <td className="py-2.5 font-mono">
                  {cnnData ? `${cnnData.inference_ms_per_image.toFixed(1)} ms/img` : "Unavailable"}
                </td>
                <td className="py-2.5 text-xs text-muted-foreground">
                  CNN is ultra-lightweight; Swin computes hierarchical window self-attention
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </Panel>

      {/* Model Selection Tabs for Detailed Inspection */}
      <div className="flex items-center gap-2 border-b border-border pb-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mr-2">
          Inspect Confusion Matrix & Per-Class Breakdown:
        </span>
        <button
          type="button"
          onClick={() => setSelectedModel("swin")}
          className={`rounded-lg px-3.5 py-1.5 text-xs font-semibold uppercase tracking-wide transition-all ${
            selectedModel === "swin"
              ? "bg-primary text-primary-foreground shadow"
              : "border border-border bg-surface-2 text-muted-foreground hover:text-foreground"
          }`}
        >
          Swin Transformer
        </button>
        <button
          type="button"
          onClick={() => setSelectedModel("cnn")}
          className={`rounded-lg px-3.5 py-1.5 text-xs font-semibold uppercase tracking-wide transition-all ${
            selectedModel === "cnn"
              ? "bg-primary text-primary-foreground shadow"
              : "border border-border bg-surface-2 text-muted-foreground hover:text-foreground"
          }`}
        >
          CNN Baseline
        </button>
      </div>

      {/* Confusion Matrix */}
      <Panel
        title={`Confusion Matrix — ${selectedModel === "swin" ? "Swin Transformer" : "CNN Baseline"}`}
        icon={Grid3x3}
        badge={<Chip tone="primary">{selectedModel === "swin" ? "Proposed Architecture" : "Baseline Architecture"}</Chip>}
        footer="Rows represent Ground Truth (Actual) emotion labels; Columns represent Model Predicted labels."
      >
        <div className="overflow-x-auto">
          <table className="min-w-[620px] border-collapse text-xs">
            <thead>
              <tr>
                <th className="border border-border bg-surface-2 px-2.5 py-2 text-left text-[10px] tracking-widest text-muted-foreground uppercase">
                  True \ Pred
                </th>
                {EMOTIONS.map((e) => (
                  <th
                    key={e}
                    className="border border-border bg-surface-2 px-2.5 py-2 text-[10px] tracking-widest text-muted-foreground uppercase text-center"
                  >
                    {e}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {EMOTIONS.map((rowName, rowIdx) => (
                <tr key={rowName}>
                  <th className="border border-border bg-surface-2 px-2.5 py-2 text-left text-[11px] font-medium">
                    {rowName}
                  </th>
                  {EMOTIONS.map((colName, colIdx) => {
                    const val = activeData?.confusion_matrix?.[rowIdx]?.[colIdx];
                    const isDiagonal = rowIdx === colIdx;
                    return (
                      <td
                        key={colName}
                        className={`border border-border px-2.5 py-2 text-center font-mono ${
                          isDiagonal
                            ? "bg-primary/15 font-bold text-primary"
                            : val && val > 0
                              ? "text-foreground"
                              : "text-muted-foreground/50"
                        }`}
                      >
                        {val !== undefined ? val : "—"}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>

      {/* Per-Class Metrics Breakdown */}
      {activeData?.per_class && (
        <Panel
          title={`Per-Class Performance Breakdown — ${selectedModel === "swin" ? "Swin Transformer" : "CNN Baseline"}`}
          icon={BarChart3}
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[500px] text-xs">
              <thead>
                <tr className="border-b border-border text-left text-[10px] tracking-widest text-muted-foreground uppercase">
                  <th className="pb-2 font-semibold">Emotion Class</th>
                  <th className="pb-2 font-semibold">Precision</th>
                  <th className="pb-2 font-semibold">Recall</th>
                  <th className="pb-2 font-semibold">F1-Score</th>
                  <th className="pb-2 font-semibold">Test Support</th>
                </tr>
              </thead>
              <tbody>
                {activeData.per_class.map((c) => (
                  <tr key={c.label} className="border-b border-border/50 last:border-0">
                    <td className="py-2 font-medium">{c.label}</td>
                    <td className="py-2 font-mono">{(c.precision * 100).toFixed(1)}%</td>
                    <td className="py-2 font-mono">{(c.recall * 100).toFixed(1)}%</td>
                    <td className="py-2 font-mono font-semibold text-primary">{(c.f1_score * 100).toFixed(1)}%</td>
                    <td className="py-2 font-mono text-muted-foreground">{c.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      )}

      <Disclaimer>
        Values are loaded from the backend evaluation JSON files. The dataset, evaluation script, and run logs are
        not included, so this page reports the supplied figures without claiming they have been independently verified.
      </Disclaimer>
    </div>
  );
}
