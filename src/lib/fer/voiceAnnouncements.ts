import type { Emotion } from "./types";

export interface VoiceAnnouncementTracker {
  lastProcessedSequence: number;
  candidate: Emotion | null;
  consecutiveCount: number;
  lastAnnouncedEmotion: Emotion | null;
  lastAnnouncedAt: number;
}

export interface VoicePredictionSample {
  sequence: number;
  emotion: Emotion | null;
  confidence: number | null;
  eligible: boolean;
  confidenceThreshold: number;
  requiredConsecutive: number;
  cooldownMs: number;
  voiceURI: string;
}

export function createVoiceAnnouncementTracker(sequence = 0): VoiceAnnouncementTracker {
  return {
    lastProcessedSequence: sequence,
    candidate: null,
    consecutiveCount: 0,
    lastAnnouncedEmotion: null,
    lastAnnouncedAt: 0,
  };
}

export function resetVoiceAnnouncementTracker(tracker: VoiceAnnouncementTracker): void {
  tracker.candidate = null;
  tracker.consecutiveCount = 0;
  tracker.lastAnnouncedEmotion = null;
}

export function processVoicePrediction(
  tracker: VoiceAnnouncementTracker,
  sample: VoicePredictionSample,
  now = Date.now(),
): boolean {
  if (sample.sequence <= tracker.lastProcessedSequence) return false;
  tracker.lastProcessedSequence = sample.sequence;

  if (
    !sample.eligible ||
    !sample.emotion ||
    sample.confidence === null ||
    sample.confidence < sample.confidenceThreshold
  ) {
    tracker.candidate = null;
    tracker.consecutiveCount = 0;
    return false;
  }

  if (tracker.candidate === sample.emotion) {
    tracker.consecutiveCount += 1;
  } else {
    tracker.candidate = sample.emotion;
    tracker.consecutiveCount = 1;
  }

  if (
    tracker.consecutiveCount < sample.requiredConsecutive ||
    tracker.lastAnnouncedEmotion === sample.emotion ||
    now - tracker.lastAnnouncedAt < sample.cooldownMs
  ) {
    return false;
  }

  if (!announceEmotion(sample.emotion, sample.voiceURI)) return false;
  tracker.lastAnnouncedEmotion = sample.emotion;
  tracker.lastAnnouncedAt = now;
  return true;
}

export function supportsVoiceAnnouncements(): boolean {
  return (
    typeof window !== "undefined" &&
    "speechSynthesis" in window &&
    typeof SpeechSynthesisUtterance !== "undefined"
  );
}

export function getEnglishVoices(): SpeechSynthesisVoice[] {
  if (!supportsVoiceAnnouncements()) return [];
  return window.speechSynthesis.getVoices().filter((voice) =>
    voice.lang.toLowerCase().startsWith("en"),
  );
}

export function announceEmotion(emotion: Emotion, voiceURI = ""): boolean {
  if (!supportsVoiceAnnouncements()) return false;

  const synthesis = window.speechSynthesis;
  // speechSynthesis queues new announcements instead of interrupting current speech.

  try {
    const utterance = new SpeechSynthesisUtterance(
      `The detected facial expression is ${emotion}.`,
    );
    const selectedVoice = getEnglishVoices().find((voice) => voice.voiceURI === voiceURI);
    utterance.lang = selectedVoice?.lang ?? "en-US";
    if (selectedVoice) utterance.voice = selectedVoice;
    synthesis.speak(utterance);
    return true;
  } catch {
    return false;
  }
}

export function cancelVoiceAnnouncements(): void {
  if (supportsVoiceAnnouncements()) window.speechSynthesis.cancel();
}
