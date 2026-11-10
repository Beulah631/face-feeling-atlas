import { cn } from "@/lib/utils";

interface FerLogoProps {
  className?: string;
  size?: number | string;
}

export function FerLogo({ className = "h-6 w-6", size }: FerLogoProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 48 48"
      fill="none"
      className={cn("shrink-0 block overflow-visible", className)}
      style={size ? { width: size, height: size } : undefined}
      aria-label="AI Facial Emotion Recognition Logo"
      role="img"
    >
      <defs>
        {/* Modern Cyan-to-Purple Gradient */}
        <linearGradient id="fer-grad-main" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#22D3EE" />
          <stop offset="50%" stopColor="#818CF8" />
          <stop offset="100%" stopColor="#C084FC" />
        </linearGradient>

        {/* AI Horizontal Scanline Gradient */}
        <linearGradient id="fer-grad-scan" x1="0%" y1="0%" x2="100%" y2="0%">
          <stop offset="0%" stopColor="#22D3EE" stopOpacity="0" />
          <stop offset="50%" stopColor="#22D3EE" stopOpacity="0.95" />
          <stop offset="100%" stopColor="#C084FC" stopOpacity="0" />
        </linearGradient>

        {/* Pure Vector Ambient Glow (No filter clipping) */}
        <radialGradient id="fer-ambient-glow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#22D3EE" stopOpacity="0.3" />
          <stop offset="60%" stopColor="#818CF8" stopOpacity="0.15" />
          <stop offset="100%" stopColor="#C084FC" stopOpacity="0" />
        </radialGradient>
      </defs>

      {/* Ambient Neural Glow Backdrop */}
      <circle cx="24" cy="24" r="22" fill="url(#fer-ambient-glow)" />

      {/* AI Face Detection Reticle Frame (YOLOv8 Bounding Indicators) */}
      <path
        d="M 7 15 V 9 C 7 7.9 7.9 7 9 7 H 15"
        stroke="#22D3EE"
        strokeWidth="2.2"
        strokeLinecap="round"
      />
      <path
        d="M 33 7 H 39 C 40.1 7 41 7.9 41 9 V 15"
        stroke="#818CF8"
        strokeWidth="2.2"
        strokeLinecap="round"
      />
      <path
        d="M 7 33 V 39 C 7 40.1 7.9 41 9 41 H 15"
        stroke="#818CF8"
        strokeWidth="2.2"
        strokeLinecap="round"
      />
      <path
        d="M 33 41 H 39 C 40.1 41 41 40.1 41 39 V 33"
        stroke="#C084FC"
        strokeWidth="2.2"
        strokeLinecap="round"
      />

      {/* Biometric Face Outline with Subtle Neural Fill */}
      <path
        d="M 16 19 C 16 13.5 19.5 10 24 10 C 28.5 10 32 13.5 32 19 C 32 25.5 27.5 32.5 24 35.5 C 20.5 32.5 16 25.5 16 19 Z"
        fill="rgba(34, 211, 238, 0.08)"
        stroke="url(#fer-grad-main)"
        strokeWidth="2.2"
        strokeLinejoin="round"
      />

      {/* Real-time AI Scanning Laser Beam */}
      <line
        x1="10"
        y1="23"
        x2="38"
        y2="23"
        stroke="url(#fer-grad-scan)"
        strokeWidth="1.5"
        strokeLinecap="round"
      />
      <circle cx="24" cy="23" r="1.25" fill="#22D3EE" />

      {/* Biometric Landmark Eyebrow Ticks */}
      <path
        d="M 18.5 15.5 Q 20.5 14.5 22 15"
        stroke="#22D3EE"
        strokeWidth="1.2"
        strokeLinecap="round"
        opacity="0.9"
      />
      <path
        d="M 26 15 Q 27.5 14.5 29.5 15.5"
        stroke="#C084FC"
        strokeWidth="1.2"
        strokeLinecap="round"
        opacity="0.9"
      />

      {/* Biometric Landmark Eyes */}
      <circle cx="20" cy="18.5" r="1.75" fill="#22D3EE" />
      <circle cx="28" cy="18.5" r="1.75" fill="#A855F7" />

      {/* Emotion Recognition Expression Arc (Affective Smile / Action Unit) */}
      <path
        d="M 19.5 28 C 21 31.5 27 31.5 28.5 28"
        stroke="url(#fer-grad-main)"
        strokeWidth="2.2"
        strokeLinecap="round"
      />

      {/* Neural Landmark Anchors */}
      <circle cx="24" cy="13.5" r="1.2" fill="#38BDF8" />
      <circle cx="24" cy="33.5" r="1.2" fill="#C084FC" />
    </svg>
  );
}
