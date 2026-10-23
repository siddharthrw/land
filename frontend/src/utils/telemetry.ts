import type { MissionState } from "../types/mission";

export function formatPhase(phase: string): string {
  return phase.replace(/_/g, " ");
}

export interface DisplayTelemetry {
  primaryLabel: string;
  primaryValue: string;
  verticalLabel: string;
  verticalValue: string;
  horizontalLabel: string;
  horizontalValue: string;
}

export function deriveTelemetry(state: MissionState): DisplayTelemetry {
  if (!state.descent) {
    return {
      primaryLabel: "Altitude",
      primaryValue: "—",
      verticalLabel: "Vertical speed",
      verticalValue: "—",
      horizontalLabel: "Horizontal speed",
      horizontalValue: "—",
    };
  }
  return {
    primaryLabel: "Altitude",
    primaryValue: `${state.descent.y.toFixed(0)} m`,
    verticalLabel: "Vertical speed",
    verticalValue: `${state.descent.vy.toFixed(1)} m/s`,
    horizontalLabel: "Horizontal speed",
    horizontalValue: `${state.descent.vx.toFixed(1)} m/s`,
  };
}
