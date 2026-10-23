import { useEffect, useState } from "react";
import { useMissionStore } from "../store/missionStore";
import type { CompareResult } from "../types/mission";

const API_URL = import.meta.env.VITE_MISSION_API_URL ?? "http://localhost:8000";

function fmt(value: number | null, suffix: string, digits = 1): string {
  return value === null ? "—" : `${value.toFixed(digits)}${suffix}`;
}

export function ComparePanel() {
  const open = useMissionStore((s) => s.comparePanelOpen);
  const toggle = useMissionStore((s) => s.toggleComparePanel);
  const ppoAvailable = useMissionStore((s) => s.ppoAvailable);
  const [data, setData] = useState<CompareResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!open || data || loading) return;
    setLoading(true);
    setError(null);
    fetch(`${API_URL}/api/compare?episodes=80`)
      .then((r) => {
        if (!r.ok) throw new Error(`Server returned ${r.status}`);
        return r.json();
      })
      .then((json: CompareResult) => setData(json))
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [open, data, loading]);

  if (!open) return null;

  return (
    <div className="compare-overlay" onClick={toggle}>
      <div className="compare-panel" onClick={(e) => e.stopPropagation()}>
        <div className="compare-header">
          <div className="hud-title">PILOT COMPARISON</div>
          <button className="btn" onClick={toggle}>
            Close
          </button>
        </div>

        {loading && <div className="compare-status">Running {80} seeded descent episodes per pilot…</div>}
        {error && <div className="compare-status compare-error">Failed to load: {error}</div>}

        {data && (
          <>
            <table className="compare-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Baseline (PD guidance)</th>
                  <th>{ppoAvailable ? "Trained PPO" : "Trained PPO (not available)"}</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td>Episodes</td>
                  <td>{data.baseline.episodes}</td>
                  <td>{data.ppo?.episodes ?? "—"}</td>
                </tr>
                <tr>
                  <td>Success rate</td>
                  <td>{fmt(data.baseline.successRate * 100, "%", 0)}</td>
                  <td>{data.ppo ? fmt(data.ppo.successRate * 100, "%", 0) : "—"}</td>
                </tr>
                <tr>
                  <td>Avg. landing speed</td>
                  <td>{fmt(data.baseline.avgLandingSpeed, " m/s", 2)}</td>
                  <td>{data.ppo ? fmt(data.ppo.avgLandingSpeed, " m/s", 2) : "—"}</td>
                </tr>
                <tr>
                  <td>Avg. fuel used</td>
                  <td>{fmt(data.baseline.avgFuelUsedKg, " kg", 0)}</td>
                  <td>{data.ppo ? fmt(data.ppo.avgFuelUsedKg, " kg", 0) : "—"}</td>
                </tr>
                <tr>
                  <td>Avg. distance to target</td>
                  <td>{fmt(data.baseline.avgDistanceToTarget, " m", 1)}</td>
                  <td>{data.ppo ? fmt(data.ppo.avgDistanceToTarget, " m", 1) : "—"}</td>
                </tr>
                <tr>
                  <td>Failure rate</td>
                  <td>{fmt(data.baseline.failureRate * 100, "%", 0)}</td>
                  <td>{data.ppo ? fmt(data.ppo.failureRate * 100, "%", 0) : "—"}</td>
                </tr>
              </tbody>
            </table>
            {!data.ppo && (
              <div className="compare-status">
                No trained PPO checkpoint found. Run{" "}
                <code>python -m agents.train</code> in <code>backend/</code>, then restart the
                backend server.
              </div>
            )}
            <div className="compare-footnote">
              Each row is averaged over {data.episodes} seeded episodes per pilot — not a single
              run — sampling the same randomized altitude/offset range for both controllers.
            </div>
          </>
        )}
      </div>
    </div>
  );
}
