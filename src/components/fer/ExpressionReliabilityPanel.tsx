import { AlertTriangle, CheckCircle2, CircleHelp, ScanFace, ShieldAlert } from "lucide-react";
import { Panel, StatCard } from "@/components/fer/ui";
import type { ReliabilityReport } from "@/lib/fer/extendedAnalysisService";

function percent(value: number | null) {
  return value === null ? "Not available" : `${value.toFixed(1)}%`;
}

function timestamp(seconds: number) {
  const minutes = Math.floor(seconds / 60)
    .toString()
    .padStart(2, "0");
  const wholeSeconds = Math.floor(seconds % 60)
    .toString()
    .padStart(2, "0");
  const tenths = Math.floor((seconds % 1) * 10);
  return `${minutes}:${wholeSeconds}.${tenths}`;
}

export function ExpressionReliabilityPanel({ report }: { report: ReliabilityReport }) {
  const statusLabel = {
    reliable: "RELIABLE",
    uncertain: "UNCERTAIN",
    insufficient_visual_evidence: "INSUFFICIENT VISUAL EVIDENCE",
  }[report.status];
  const statusIcon =
    report.status === "reliable"
      ? CheckCircle2
      : report.status === "uncertain"
        ? CircleHelp
        : ShieldAlert;
  const StatusIcon = statusIcon;
  const agreement = report.model_agreement;
  const temporal = report.temporal_stability;
  const quality = report.face_quality;
  const insights: string[] = [];

  if (agreement.paired_prediction_frames > 0) {
    insights.push(
      `CNN and Swin agreed in ${agreement.agreeing_frames} of ${agreement.paired_prediction_frames} paired face frames${agreement.disagreeing_frames ? ` and disagreed in ${agreement.disagreeing_frames}` : ""}.`,
    );
  } else {
    insights.push(
      "CNN–Swin agreement could not be calculated because paired model predictions were unavailable.",
    );
  }
  if (temporal.consecutive_prediction_pairs > 0) {
    insights.push(
      `The ${temporal.primary_model === "swin" ? "Swin" : "CNN"} prediction changed ${temporal.expression_changes} times across ${temporal.consecutive_prediction_pairs} consecutive same-track comparisons.`,
    );
  } else {
    insights.push(
      "Temporal consistency could not be calculated because there were not consecutive predictions for a tracked face.",
    );
  }
  if (quality.insufficient_quality_frames > 0) {
    insights.push(
      `${quality.insufficient_quality_frames} of ${quality.frame_count} sampled frames had no face or failed the configured visual-quality screening rules.`,
    );
  }
  insights.push(
    "These findings describe model-output consistency and image measurements; they do not establish a person's actual emotional state or prediction accuracy.",
  );

  return (
    <section className="space-y-4" aria-labelledby="expression-reliability-title">
      <header>
        <p className="text-[11px] font-semibold tracking-widest text-primary uppercase">
          Research analysis
        </p>
        <h2 id="expression-reliability-title" className="mt-1 font-display text-xl font-bold">
          Expression Reliability Analysis
        </h2>
        <p className="mt-1 text-sm text-muted-foreground">
          Understand prediction consistency, model agreement and facial image quality across video
          frames.
        </p>
      </header>

      <Panel title="Overall reliability summary" icon={StatusIcon}>
        <div
          className={`inline-flex items-center gap-2 rounded-full border px-3 py-2 text-xs font-bold tracking-wide ${report.status === "reliable" ? "border-success/30 bg-success/10 text-success" : report.status === "uncertain" ? "border-warning/30 bg-warning/10 text-warning" : "border-destructive/30 bg-destructive/10 text-destructive"}`}
        >
          <StatusIcon className="h-4 w-4" />
          {statusLabel}
        </div>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
          <StatCard
            label="CNN–Swin agreement"
            value={percent(agreement.agreement_percent)}
            hint={`${agreement.agreeing_frames} agree · ${agreement.disagreeing_frames} disagree · ${agreement.unavailable_prediction_frames} unavailable`}
            icon={CheckCircle2}
          />
          <StatCard
            label="Temporal consistency"
            value={percent(temporal.temporal_consistency_percent)}
            hint={`${temporal.expression_changes} changes across ${temporal.consecutive_prediction_pairs} adjacent comparisons`}
            icon={CircleHelp}
            tone="accent"
          />
          <StatCard
            label="Usable prediction frames"
            value={String(quality.usable_frames)}
            hint={`Primary model: ${temporal.primary_model}`}
            icon={ScanFace}
            tone="success"
          />
          <StatCard
            label="Insufficient-quality frames"
            value={String(quality.insufficient_quality_frames)}
            hint={`of ${quality.frame_count} sampled frames`}
            icon={AlertTriangle}
            tone="warning"
          />
        </div>
        {report.reasons.length ? (
          <ul className="mt-4 list-disc space-y-1 pl-5 text-xs text-muted-foreground">
            {report.reasons.map((reason) => (
              <li key={reason}>{reason}</li>
            ))}
          </ul>
        ) : (
          <p className="mt-4 text-xs text-muted-foreground">
            The configured screening rules were met for the available paired predictions, temporal
            sequence and image-quality measurements. Agreement is not proof of correctness.
          </p>
        )}
      </Panel>

      <div className="grid gap-4 xl:grid-cols-2">
        <Panel title="Model disagreement timeline" icon={CircleHelp}>
          {agreement.timeline.length ? (
            <div className="max-h-96 space-y-2 overflow-auto">
              {agreement.timeline.map((frame) => (
                <article
                  key={`${frame.sample_index}-${frame.face_id}`}
                  className="flex flex-wrap items-center justify-between gap-2 rounded-lg border border-border p-3 text-xs"
                >
                  <div>
                    <p className="font-semibold">
                      {timestamp(frame.timestamp_seconds)} · {frame.face_id}
                    </p>
                    <p className="mt-1 text-muted-foreground">
                      CNN:{" "}
                      <span className="font-medium text-foreground">
                        {frame.cnn_emotion ?? "Unavailable"}
                      </span>{" "}
                      · Swin:{" "}
                      <span className="font-medium text-foreground">
                        {frame.swin_emotion ?? "Unavailable"}
                      </span>
                    </p>
                  </div>
                  <span
                    className={`rounded-full border px-2 py-1 text-[10px] font-semibold uppercase ${frame.agree === true ? "border-success/30 bg-success/10 text-success" : frame.agree === false ? "border-warning/30 bg-warning/10 text-warning" : "border-border bg-surface-2 text-muted-foreground"}`}
                  >
                    {frame.agree === true
                      ? "Agree"
                      : frame.agree === false
                        ? "Disagree"
                        : "Unavailable"}
                  </span>
                </article>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              No detected face frames were available for model comparison.
            </p>
          )}
          <p className="mt-3 text-[11px] text-muted-foreground">
            Agreement counts compare predicted labels for the same temporary face track and sampled
            frame. Missing outputs are excluded from the agreement percentage.
          </p>
        </Panel>

        <Panel title="Temporal stability" icon={CircleHelp}>
          <div className="grid gap-3 sm:grid-cols-2">
            <div className="rounded-lg bg-surface-2/50 p-3">
              <p className="text-[10px] font-semibold tracking-widest text-muted-foreground uppercase">
                Most frequent expression
              </p>
              <p className="mt-1 text-lg font-bold">
                {temporal.most_frequent_expression ?? "Not available"}
              </p>
              <p className="text-[11px] text-muted-foreground">
                Based on {temporal.primary_model} outputs
              </p>
            </div>
            <div className="rounded-lg bg-surface-2/50 p-3">
              <p className="text-[10px] font-semibold tracking-widest text-muted-foreground uppercase">
                Expression changes
              </p>
              <p className="mt-1 text-lg font-bold">{temporal.expression_changes}</p>
              <p className="text-[11px] text-muted-foreground">
                Across consecutive sampled predictions for each track
              </p>
            </div>
          </div>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <div>
              <h3 className="text-xs font-semibold">
                Stable sequences · {temporal.stable_sequences.length}
              </h3>
              <div className="mt-2 space-y-2">
                {temporal.stable_sequences.length ? (
                  temporal.stable_sequences.map((sequence, index) => (
                    <p
                      key={`${sequence.face_id}-${sequence.start_sample_index}-${index}`}
                      className="rounded-lg border border-border p-2 text-[11px]"
                    >
                      {sequence.face_id}: {sequence.emotion} ·{" "}
                      {timestamp(sequence.start_timestamp_seconds)}–
                      {timestamp(sequence.end_timestamp_seconds)} · {sequence.frames} frames
                    </p>
                  ))
                ) : (
                  <p className="text-xs text-muted-foreground">No repeated same-label runs.</p>
                )}
              </div>
            </div>
            <div>
              <h3 className="text-xs font-semibold">
                Unstable single-frame runs · {temporal.unstable_sequences.length}
              </h3>
              <div className="mt-2 space-y-2">
                {temporal.unstable_sequences.length ? (
                  temporal.unstable_sequences.map((sequence, index) => (
                    <p
                      key={`${sequence.face_id}-${sequence.start_sample_index}-${index}`}
                      className="rounded-lg border border-border p-2 text-[11px]"
                    >
                      {sequence.face_id}: {sequence.emotion} ·{" "}
                      {timestamp(sequence.start_timestamp_seconds)}
                    </p>
                  ))
                ) : (
                  <p className="text-xs text-muted-foreground">No isolated one-frame runs.</p>
                )}
              </div>
            </div>
          </div>
          <p className="mt-3 text-[11px] text-muted-foreground">{temporal.note}</p>
        </Panel>
      </div>

      <Panel title="Face quality report" icon={ScanFace}>
        <p className="mb-3 text-xs text-muted-foreground">
          Blur uses grayscale Laplacian variance; lighting uses mean grayscale brightness (0–255).
          The detection score is YOLOv8's returned confidence, not an emotion-model reliability
          score.
        </p>
        {quality.measurements.length ? (
          <div className="max-h-[30rem] overflow-auto">
            <table className="w-full min-w-[720px] border-collapse text-left text-xs">
              <thead>
                <tr className="border-b border-border text-muted-foreground">
                  <th className="p-2">Frame / face</th>
                  <th className="p-2">Visibility</th>
                  <th className="p-2">Crop size</th>
                  <th className="p-2">Blur variance</th>
                  <th className="p-2">Brightness</th>
                  <th className="p-2">YOLO score</th>
                  <th className="p-2">Screening</th>
                </tr>
              </thead>
              <tbody>
                {quality.measurements.map((item) => (
                  <tr
                    key={`${item.sample_index}-${item.face_id}`}
                    className="border-b border-border/70"
                  >
                    <td className="p-2">
                      {timestamp(item.timestamp_seconds)} · {item.face_id}
                    </td>
                    <td className="p-2">
                      {item.face_visibility === null
                        ? "Not available"
                        : `${(item.face_visibility * 100).toFixed(1)}%`}
                    </td>
                    <td className="p-2">
                      {item.face_crop_width_px ?? "N/A"} × {item.face_crop_height_px ?? "N/A"} px
                    </td>
                    <td className="p-2">
                      {item.blur_laplacian_variance?.toFixed(2) ?? "Not available"}
                    </td>
                    <td className="p-2">{item.brightness_mean?.toFixed(1) ?? "Not available"}</td>
                    <td className="p-2">
                      {item.detection_confidence === null
                        ? "Not available"
                        : `${(item.detection_confidence * 100).toFixed(1)}%`}
                    </td>
                    <td className="p-2">
                      {item.quality_flags.length
                        ? item.quality_flags.join(", ")
                        : "Within configured screen"}
                      {item.measurement_error ? ` · ${item.measurement_error}` : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-sm text-muted-foreground">No measurable face crops were available.</p>
        )}
        <p className="mt-3 text-[11px] text-muted-foreground">
          {quality.visibility_note} Actual occlusion or pose-based face visibility cannot be
          estimated by the current models.
        </p>
      </Panel>

      <Panel title="Research insights" icon={AlertTriangle}>
        <ul className="list-disc space-y-2 pl-5 text-sm">
          {insights.map((insight) => (
            <li key={insight}>{insight}</li>
          ))}
        </ul>
      </Panel>

      <Panel title="Reliability rules" icon={AlertTriangle}>
        <p className="mb-3 text-xs text-muted-foreground">
          Rules are configurable through backend environment variables. The current values are
          returned with this report. They are transparent screening heuristics and are not validated
          scientific cutoffs.
        </p>
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
          {Object.entries(report.rules)
            .filter(([key]) => key !== "threshold_note")
            .map(([key, value]) => (
              <div key={key} className="rounded-lg bg-surface-2/50 p-2 text-[11px]">
                <span className="text-muted-foreground">{key.replaceAll("_", " ")}</span>
                <span className="float-right font-semibold">{value}</span>
              </div>
            ))}
        </div>
        <p className="mt-3 text-[11px] text-warning">{String(report.rules["threshold_note"])}</p>
        <p className="mt-2 text-[11px] text-muted-foreground">{report.confidence_note}</p>
      </Panel>
    </section>
  );
}
