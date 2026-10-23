import { useEffect, useState } from "react";
import { useMissionStore } from "../store/missionStore";
import { deriveTelemetry, formatPhase } from "../utils/telemetry";
import { LunarSurfaceView } from "./LunarSurfaceView";
import { useSmoothed } from "./useSmoothed";
import { ActionFeedPanel, DecisionLogPanel, ScorePanel, ThrustMonitorPanel } from "./BottomPanels";
import "./dashboard.css";

const FAILURE_PHASES = new Set(["CRASH", "FUEL_EXHAUSTED", "ABORTED"]);

const FIDELITY_COLOR: Record<string, string> = {
  scripted: "#ffc94a",
  simulated: "#4bc7f2",
  learned: "#9b5cff",
};

function useUtcClock() {
  const [utc, setUtc] = useState(() => new Date().toUTCString().slice(17, 25));
  useEffect(() => {
    const id = setInterval(() => setUtc(new Date().toUTCString().slice(17, 25)), 1000);
    return () => clearInterval(id);
  }, []);
  return utc;
}

export function Dashboard() {
  const connect = useMissionStore((s) => s.connect);
  const connectionStatus = useMissionStore((s) => s.connectionStatus);
  const state = useMissionStore((s) => s.state);
  const ppoAvailable = useMissionStore((s) => s.ppoAvailable);
  const launch = useMissionStore((s) => s.launch);
  const pause = useMissionStore((s) => s.pause);
  const resume = useMissionStore((s) => s.resume);
  const restart = useMissionStore((s) => s.restart);
  const setController = useMissionStore((s) => s.setController);
  const setSimSpeed = useMissionStore((s) => s.setSimSpeed);
  const manualInput = useMissionStore((s) => s.manualInput);
  const toggleComparePanel = useMissionStore((s) => s.toggleComparePanel);
  const [manualThrottle, setManualThrottle] = useState(0.3);
  const [manualTilt, setManualTilt] = useState(0);
  const utc = useUtcClock();

  useEffect(() => {
    connect();
  }, [connect]);

  const phase = state?.phase ?? "PRELAUNCH";
  const isFailure = FAILURE_PHASES.has(phase);
  const isSuccess = phase === "MISSION_COMPLETE" || phase === "TOUCHDOWN";

  const canLaunch = phase === "PRELAUNCH";
  const canPause = !!state && !state.paused && phase !== "PRELAUNCH";
  const canResume = !!state && state.paused;
  const inDescent = phase === "LUNAR_DESCENT";
  const fidelity = state ? state.fidelity[state.phase] ?? "scripted" : "scripted";
  const telemetry = state ? deriveTelemetry(state) : null;
  const tiltDeg = ((state?.vehicle.attitude?.[2] ?? 0) * 180) / Math.PI;
  const smoothThrottle = useSmoothed(state?.vehicle.throttle ?? 0);

  return (
    <div className="dash-root">
      <header className="dash-header">
        <div className="dash-brand">
          <div className="dash-brand-mark" />
          <div>
            <div className="dash-brand-title">LUNAR PILOT</div>
            <div className="dash-brand-sub">LANDING CHALLENGE</div>
          </div>
        </div>
        <div className="dash-header-right">
          <span className={`dash-chip ${connectionStatus === "open" ? "live" : "offline"}`}>
            {connectionStatus === "open" ? "LIVE" : connectionStatus.toUpperCase()}
          </span>
          <span>PILOT: {state?.activeController.toUpperCase() ?? "—"}</span>
          <span>SIM T+{state?.simTime.toFixed(1) ?? "0.0"}s</span>
          <span>UTC {utc}</span>
        </div>
      </header>

      <div className="dash-main">
        <div className="dash-panel dash-col-controls">
          <div className="dash-panel-head">MISSION CONTROLS</div>
          <div className="dash-panel-body">
            <div className="dash-btn-row">
              {canLaunch && (
                <button className="dash-btn primary" onClick={launch}>
                  LAUNCH
                </button>
              )}
              {canPause && <button className="dash-btn" onClick={pause}>PAUSE</button>}
              {canResume && <button className="dash-btn" onClick={resume}>RESUME</button>}
              <button className="dash-btn" onClick={restart}>RESTART</button>
              <button className="dash-btn" onClick={toggleComparePanel}>COMPARE PILOTS</button>
            </div>

            <div className="dash-divider" />

            <label className="dash-field-label">SIM SPEED</label>
            <div className="dash-speed-row">
              {[1, 2, 4, 8].map((speed) => (
                <button
                  key={speed}
                  className={`dash-speed-btn ${(state?.simSpeed ?? 1) === speed ? "active" : ""}`}
                  onClick={() => setSimSpeed(speed)}
                >
                  {speed}x
                </button>
              ))}
            </div>

            <div className="dash-divider" />

            {inDescent && (
              <>
                <label className="dash-field-label">PILOT</label>
                <select
                  className="dash-select"
                  value={state?.activeController ?? "baseline"}
                  onChange={(e) => setController(e.target.value as "baseline" | "ppo" | "manual")}
                >
                  <option value="baseline">Baseline guidance</option>
                  <option value="manual">Manual override</option>
                  <option value="ppo" disabled={!ppoAvailable}>
                    Trained PPO {ppoAvailable ? "" : "(not trained)"}
                  </option>
                </select>
              </>
            )}

            {inDescent && state?.activeController === "manual" && (
              <>
                <label className="dash-field-label">THROTTLE {Math.round(manualThrottle * 100)}%</label>
                <input
                  className="dash-slider"
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={manualThrottle}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    setManualThrottle(v);
                    manualInput(v, manualTilt);
                  }}
                />
                <label className="dash-field-label">TILT {Math.round((manualTilt * 180) / Math.PI)}°</label>
                <input
                  className="dash-slider"
                  type="range"
                  min={-0.49}
                  max={0.49}
                  step={0.01}
                  value={manualTilt}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    setManualTilt(v);
                    manualInput(manualThrottle, v);
                  }}
                />
              </>
            )}
          </div>
        </div>

        <div className="dash-panel dash-col-center">
          <div className="dash-panel-head">LUNAR SURFACE</div>
          <div className="dash-phase-card">
            <span className="dash-phase-name">{formatPhase(phase)}</span>
            <span className="dash-fidelity-badge" style={{ color: FIDELITY_COLOR[fidelity] }}>
              {fidelity.toUpperCase()}
            </span>
            {isFailure && <span className="dash-fidelity-badge" style={{ color: "var(--red)" }}>FAILED</span>}
            {isSuccess && <span className="dash-fidelity-badge" style={{ color: "var(--green)" }}>LANDED</span>}
          </div>
          <div className="dash-orbit-wrap">
            <LunarSurfaceView descent={state?.descent ?? null} throttle={state?.vehicle.throttle ?? 0} tiltDeg={tiltDeg} />
          </div>
        </div>

        <div className="dash-panel dash-col-state">
          <div className="dash-panel-head">VEHICLE STATE</div>
          <div className="dash-panel-body">
            {telemetry && (
              <>
                <DataRow label={telemetry.primaryLabel} value={telemetry.primaryValue} />
                <DataRow label={telemetry.verticalLabel} value={telemetry.verticalValue} />
                <DataRow label={telemetry.horizontalLabel} value={telemetry.horizontalValue} />
              </>
            )}
            <DataRow
              label="Distance to target"
              value={state?.descent ? `${Math.abs(state.descent.x).toFixed(0)} m` : "—"}
            />
            <DataRow label="Mass" value={state?.vehicle.mass ? `${state.vehicle.mass.toFixed(0)} kg` : "—"} />
            <DataRow label="Throttle" value={`${(smoothThrottle * 100).toFixed(0)}%`} />
            <DataRow label="Tilt" value={`${tiltDeg.toFixed(1)}°`} />
            <div className="dash-field-label" style={{ marginTop: 10 }}>FUEL</div>
            <div className="dash-fuel-bar">
              <div
                className="dash-fuel-fill"
                style={{
                  width: `${Math.max(0, Math.min(100, (state?.vehicle.fuelFraction ?? 1) * 100))}%`,
                  background: (state?.vehicle.fuelFraction ?? 1) < 0.15 ? "var(--red)" : "var(--green)",
                }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="dash-bottom">
        <ActionFeedPanel />
        <DecisionLogPanel />
        <ScorePanel />
        <ThrustMonitorPanel />
      </div>

      {connectionStatus !== "open" && (
        <div className="dash-connection-banner">
          {connectionStatus === "connecting" ? "Connecting to mission server…" : "Disconnected — retrying…"}
        </div>
      )}
    </div>
  );
}

function DataRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="dash-data-row">
      <span className="dash-data-label">{label}</span>
      <span className="dash-data-value">{value}</span>
    </div>
  );
}
