import { useMissionStore } from "../store/missionStore";
import type { DecisionLogEntry } from "../types/mission";
import { useSmoothed } from "./useSmoothed";

const EMPTY_LOG: DecisionLogEntry[] = [];

const ICONS: { match: RegExp; icon: string }[] = [
  { match: /engine shutdown/i, icon: "▲" },
  { match: /ignited|descent engines/i, icon: "↓" },
  { match: /touchdown|landed/i, icon: "◉" },
  { match: /crash|fuel exhaust|abort/i, icon: "✕" },
  { match: /fuel|bonus|accuracy/i, icon: "♦" },
];
function iconFor(message: string): string {
  return ICONS.find((i) => i.match.test(message))?.icon ?? "•";
}

export function ActionFeedPanel() {
  const eventLog = useMissionStore((s) => s.eventLog);
  const throttle = useMissionStore((s) => s.state?.vehicle.throttle ?? 0);
  const smoothThrottle = useSmoothed(throttle);
  const phase = useMissionStore((s) => s.state?.phase);
  const firing = throttle > 0.04 && phase !== "PRELAUNCH";

  return (
    <div className="dash-panel">
      <div className="dash-panel-head">ACTION FEED</div>
      <div className="dash-panel-body">
        {firing && (
          <div className="dash-feed-live">
            <span className="dash-feed-dot" />
            Thrusters firing — {(smoothThrottle * 100).toFixed(0)}% throttle
          </div>
        )}
        <div className="dash-list">
          {eventLog.length === 0 && !firing && <div className="dash-list-empty">No actions yet.</div>}
          {eventLog
            .slice()
            .reverse()
            .map((e) => (
              <div key={e.id} className="dash-list-item">
                <span>{iconFor(e.message)}</span>
                <span className="dash-list-time">T+{e.sim_time.toFixed(1)}s</span>
                <span>{e.message}</span>
                {e.score_delta > 0 && <span className="dash-list-score">+{e.score_delta}</span>}
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

export function DecisionLogPanel() {
  const decisionLog = useMissionStore((s) => s.state?.decisionLog ?? EMPTY_LOG);
  const phase = useMissionStore((s) => s.state?.phase);
  const active = phase === "LUNAR_DESCENT" || phase === "TOUCHDOWN";

  return (
    <div className="dash-panel">
      <div className="dash-panel-head">AI PILOT DECISION LOG</div>
      <div className="dash-panel-body">
        <div className="dash-list">
          {(!active || decisionLog.length === 0) && <div className="dash-list-empty">No decisions yet.</div>}
          {active &&
            decisionLog
              .slice()
              .reverse()
              .map((d, i) => (
                <div key={i} className="dash-list-item">
                  <span className="dash-list-time">T+{d.simTime.toFixed(1)}s</span>
                  <span>{d.reason}</span>
                </div>
              ))}
        </div>
      </div>
    </div>
  );
}

const FAILURE_PHASES = new Set(["CRASH", "FUEL_EXHAUSTED", "ABORTED"]);
const MAX_SCORE = 700; // SAFE_TOUCHDOWN 300 + controlled 150 + fuel 100 + accuracy 100 + shutdown 50

export function ScorePanel() {
  const score = useMissionStore((s) => s.state?.score ?? 0);
  const phase = useMissionStore((s) => s.state?.phase);
  const isSuccess = phase === "MISSION_COMPLETE";
  const isFailure = phase ? FAILURE_PHASES.has(phase) : false;

  return (
    <div className="dash-panel">
      <div className="dash-panel-head">MISSION SCORE</div>
      <div className="dash-panel-body">
        <span className="dash-score-value">{score}</span>
        <span className="dash-score-max"> / {MAX_SCORE}</span>
        {isSuccess && <div className="dash-score-result success">MISSION SUCCESS</div>}
        {isFailure && <div className="dash-score-result failure">MISSION FAILURE</div>}
      </div>
    </div>
  );
}

export function ThrustMonitorPanel() {
  const throttle = useMissionStore((s) => s.state?.vehicle.throttle ?? 0);
  const phase = useMissionStore((s) => s.state?.phase ?? "PRELAUNCH");
  const activeController = useMissionStore((s) => s.state?.activeController ?? "baseline");
  const tiltDeg = useMissionStore((s) => ((s.state?.vehicle.attitude?.[2] ?? 0) * 180) / Math.PI);
  const smoothThrottle = useSmoothed(throttle);

  const engineOn = throttle > 0.02 && phase !== "PRELAUNCH";
  const flameScale = Math.max(0.15, throttle).toFixed(2);

  return (
    <div className="dash-panel">
      <div className="dash-panel-head">THRUST MONITOR</div>
      <div className="dash-panel-body">
        <div className="dash-thrust-row">
          <div className="dash-thrust-rocket">
            <svg width="56" height="118" viewBox="0 0 40 90">
              <defs>
                <filter id="rocketGlow" x="-60%" y="-60%" width="220%" height="220%">
                  <feGaussianBlur stdDeviation="1.6" result="b" />
                  <feMerge>
                    <feMergeNode in="b" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              <g filter="url(#rocketGlow)">
                <path
                  d="M20 2 L28 22 L28 60 L12 60 L12 22 Z"
                  fill="none"
                  stroke="var(--cyan)"
                  strokeWidth="1.3"
                />
                <path d="M12 46 L4 60 L12 56 Z" fill="none" stroke="var(--cyan)" strokeWidth="1" />
                <path d="M28 46 L36 60 L28 56 Z" fill="none" stroke="var(--cyan)" strokeWidth="1" />
                <line x1="12" y1="60" x2="28" y2="60" stroke="var(--cyan)" strokeWidth="1.3" />
              </g>
              {engineOn && (
                <path
                  className="flame-flicker"
                  d="M14 60 L26 60 L20 90 Z"
                  fill="var(--amber)"
                  opacity="0.9"
                  style={{ transform: `scaleY(${flameScale})` }}
                />
              )}
            </svg>
          </div>
          <div className="dash-thrust-readouts">
            <span className={`dash-engine-badge ${engineOn ? "firing" : "idle"}`}>
              {engineOn ? "FIRING" : "IDLE"}
            </span>
            <div className="dash-data-row" style={{ borderBottom: "none", padding: "5px 0" }}>
              <span className="dash-data-label">Pilot</span>
              <span className="dash-data-value" style={{ fontSize: 13 }}>{activeController.toUpperCase()}</span>
            </div>
            <div className="dash-data-row" style={{ borderBottom: "none", padding: "5px 0" }}>
              <span className="dash-data-label">Tilt</span>
              <span className="dash-data-value" style={{ fontSize: 13 }}>{tiltDeg.toFixed(1)}°</span>
            </div>
            <div className="dash-thrust-bar">
              <div className="dash-thrust-fill" style={{ width: `${Math.min(100, smoothThrottle * 100)}%` }} />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
