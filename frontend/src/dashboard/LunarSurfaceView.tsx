import type { DescentTelemetry } from "../types/mission";

// Must match backend/simulation/physics.py — LANDING_RADIUS is the "counts as
// near target" tolerance used for scoring; we render it as two poles marking
// the edges of that corridor so a human pilot has the same target to aim for.
const LANDING_RADIUS = 60;
const X_RANGE = 500; // meters shown on either side of center
const GROUND_Y = 196;
const SKY_TOP_Y = 18;
const MAX_ALTITUDE = 1800;

function scaleX(xm: number): number {
  const clamped = Math.max(-X_RANGE, Math.min(X_RANGE, xm));
  return 200 + (clamped / X_RANGE) * 180;
}

function scaleY(yAlt: number): number {
  const frac = Math.max(0, Math.min(1, yAlt / MAX_ALTITUDE));
  return GROUND_Y - frac * (GROUND_Y - SKY_TOP_Y);
}

function Pole({ xm }: { xm: number }) {
  const px = scaleX(xm);
  return (
    <g>
      <line x1={px} y1={GROUND_Y} x2={px} y2={GROUND_Y - 34} stroke="var(--amber)" strokeWidth="2" />
      <circle cx={px} cy={GROUND_Y - 34} r="3.5" fill="var(--amber)" />
      <path d={`M ${px} ${GROUND_Y - 34} L ${px + 14} ${GROUND_Y - 29} L ${px} ${GROUND_Y - 24} Z`} fill="var(--amber)" opacity="0.85" />
    </g>
  );
}

export function LunarSurfaceView({
  descent,
  throttle,
  tiltDeg,
}: {
  descent: DescentTelemetry | null;
  throttle: number;
  tiltDeg: number;
}) {
  const x = descent?.x ?? 0;
  const y = descent?.y ?? MAX_ALTITUDE;
  const vx = descent?.vx ?? 0;
  const vy = descent?.vy ?? -2;
  const px = scaleX(x);
  const py = scaleY(y);
  const engineOn = throttle > 0.02;
  const flameLen = 6 + Math.max(0.1, throttle) * 16;
  const inCorridor = Math.abs(x) <= LANDING_RADIUS;

  // A lander's body stays roughly upright under thrust (only the engine
  // gimbal tilts), but a purely upright sprite reads as "falling straight
  // down" even while it's covering real ground sideways. Leaning the sprite
  // partway into its actual velocity heading — on top of the real engine
  // tilt — makes the diagonal approach visually read as diagonal, capped so
  // it never looks like it's nose-diving.
  const headingDeg = (Math.atan2(vx, Math.max(1, -vy)) * 180) / Math.PI;
  const visualTilt = Math.max(-40, Math.min(40, tiltDeg + headingDeg * 0.5));

  const leftPoleX = scaleX(-LANDING_RADIUS);
  const rightPoleX = scaleX(LANDING_RADIUS);

  return (
    <svg viewBox="0 0 400 220" preserveAspectRatio="xMidYMid meet">
      <defs>
        <linearGradient id="groundGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#3c3c46" />
          <stop offset="100%" stopColor="#1c1c22" />
        </linearGradient>
        <filter id="landerGlow" x="-100%" y="-100%" width="300%" height="300%">
          <feGaussianBlur stdDeviation="2" result="b" />
          <feMerge>
            <feMergeNode in="b" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <rect
        x={leftPoleX}
        y={SKY_TOP_Y}
        width={rightPoleX - leftPoleX}
        height={GROUND_Y - SKY_TOP_Y}
        fill={inCorridor ? "rgba(57,255,176,0.06)" : "rgba(75,199,242,0.04)"}
      />
      <line x1={leftPoleX} y1={SKY_TOP_Y} x2={leftPoleX} y2={GROUND_Y} stroke="rgba(255,201,74,0.15)" strokeDasharray="3 4" />
      <line x1={rightPoleX} y1={SKY_TOP_Y} x2={rightPoleX} y2={GROUND_Y} stroke="rgba(255,201,74,0.15)" strokeDasharray="3 4" />

      <rect x="0" y={GROUND_Y} width="400" height={220 - GROUND_Y} fill="url(#groundGrad)" />
      <line x1="0" y1={GROUND_Y} x2="400" y2={GROUND_Y} stroke="#55555f" strokeWidth="1.5" />
      {/* decorative craters — purely cosmetic, the real landing surface is near-flat */}
      {[
        [30, 8], [95, 5], [145, 10], [260, 6], [320, 9], [370, 5],
      ].map(([cx, r], i) => (
        <ellipse key={i} cx={cx} cy={GROUND_Y + 2} rx={r} ry={r * 0.35} fill="#15151a" stroke="#3a3a44" strokeWidth="0.5" />
      ))}

      <Pole xm={-LANDING_RADIUS} />
      <Pole xm={LANDING_RADIUS} />

      <text x="200" y={GROUND_Y + 15} textAnchor="middle" fontSize="9" fill="var(--text-dim)" letterSpacing="1">
        TARGET CORRIDOR — {LANDING_RADIUS * 2}m
      </text>

      <g className="lander-sprite" filter="url(#landerGlow)" transform={`translate(${px} ${py}) rotate(${visualTilt})`}>
        <path d="M0 -14 L7 6 L7 14 L-7 14 L-7 6 Z" fill="none" stroke={inCorridor ? "var(--green)" : "var(--cyan)"} strokeWidth="1.6" />
        <line x1="-7" y1="14" x2="7" y2="14" stroke={inCorridor ? "var(--green)" : "var(--cyan)"} strokeWidth="1.6" />
        {engineOn && (
          <path
            className="flame-flicker"
            d={`M-5 14 L5 14 L0 ${14 + flameLen} Z`}
            fill="var(--amber)"
            opacity="0.9"
          />
        )}
      </g>
    </svg>
  );
}
