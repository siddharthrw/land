import { useEffect, useRef } from "react";
import { useMissionStore } from "../store/missionStore";

export interface FlightLogEntry {
  simTime: number;
  phase: string;
  pilot: string;
  altitude: number | null;
  x: number | null;
  vy: number | null;
  vx: number | null;
  throttle: number;
  tiltDeg: number;
  fuelPct: number;
  reason: string;
}

const SAMPLE_INTERVAL = 0.5; // sim-seconds between routine samples

/** Accumulates a detailed, full-flight log client-side (nothing extra sent
 * over the wire — just samples the state that already arrives every tick),
 * so a crash can be handed over as exact numbers instead of a screenshot
 * caught at one random moment. Resets whenever a new mission starts. */
export function useFlightLogger() {
  const logRef = useRef<FlightLogEntry[]>([]);
  const lastSampledT = useRef(-999);
  const lastPhase = useRef<string | null>(null);

  const state = useMissionStore((s) => s.state);

  useEffect(() => {
    if (!state) return;

    if (state.phase === "PRELAUNCH") {
      if (lastPhase.current !== "PRELAUNCH") {
        logRef.current = [];
      }
      lastPhase.current = state.phase;
      lastSampledT.current = -999;
      return;
    }

    const phaseChanged = lastPhase.current !== state.phase;
    const timeDue = state.simTime - lastSampledT.current >= SAMPLE_INTERVAL;
    if (!phaseChanged && !timeDue) return;

    const latestDecision = state.decisionLog[state.decisionLog.length - 1];
    const reason = latestDecision?.reason ?? "";
    const pilot = reason.startsWith("[Safety fallback")
      ? "baseline (safety fallback)"
      : state.activeController;

    logRef.current.push({
      simTime: state.simTime,
      phase: state.phase,
      pilot,
      altitude: state.descent?.y ?? null,
      x: state.descent?.x ?? null,
      vy: state.descent?.vy ?? null,
      vx: state.descent?.vx ?? null,
      throttle: state.vehicle.throttle,
      tiltDeg: (state.vehicle.attitude[2] * 180) / Math.PI,
      fuelPct: state.vehicle.fuelFraction * 100,
      reason,
    });
    lastSampledT.current = state.simTime;
    lastPhase.current = state.phase;
  }, [state]);

  return logRef;
}

export function formatFlightLog(log: FlightLogEntry[]): string {
  const lines = [
    "LUNAR PILOT — FLIGHT LOG",
    `Generated: ${new Date().toISOString()}`,
    "",
    "T+sec   PHASE             PILOT                      ALT(m)    X(m)     VY      VX    THR%  TILT°  FUEL%  NOTE",
  ];
  for (const e of log) {
    lines.push(
      `${e.simTime.toFixed(1).padStart(6)}  ${e.phase.padEnd(17)} ${e.pilot.padEnd(26)} ` +
        `${(e.altitude ?? 0).toFixed(0).padStart(7)}  ${(e.x ?? 0).toFixed(0).padStart(6)}  ` +
        `${(e.vy ?? 0).toFixed(1).padStart(6)}  ${(e.vx ?? 0).toFixed(1).padStart(6)}  ` +
        `${(e.throttle * 100).toFixed(0).padStart(4)}  ${e.tiltDeg.toFixed(1).padStart(5)}  ${e.fuelPct.toFixed(0).padStart(5)}  ${e.reason}`
    );
  }
  if (log.length === 0) lines.push("(no flight data yet — launch first)");
  return lines.join("\n");
}
