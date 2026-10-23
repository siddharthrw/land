export type Vec3 = [number, number, number];

export type FidelityLevel = "scripted" | "simulated" | "learned";

export interface Vehicle {
  attitude: Vec3;
  fuelFraction: number;
  mass: number | null;
  throttle: number;
}

export interface DecisionLogEntry {
  simTime: number;
  controller: string;
  throttle: number;
  tilt: number;
  reason: string;
}

export interface DescentTelemetry {
  x: number;
  y: number;
  vx: number;
  vy: number;
  fuelKg: number;
}

export interface MissionState {
  phase: string;
  simTime: number;
  paused: boolean;
  simSpeed: number;
  vehicle: Vehicle;
  activeController: "baseline" | "ppo" | "manual";
  score: number;
  fidelity: Record<string, FidelityLevel>;
  decisionLog: DecisionLogEntry[];
  descent: DescentTelemetry | null;
}

export interface MissionEvent {
  id: string;
  type: string;
  message: string;
  phase: string;
  sim_time: number;
  score_delta: number;
}

export interface MissionMessage {
  state: MissionState;
  events: MissionEvent[];
  ppoAvailable: boolean;
}

export type CommandType =
  | "launch"
  | "pause"
  | "resume"
  | "restart"
  | "abort"
  | "set_controller"
  | "manual_input"
  | "set_sim_speed";

export interface Command {
  type: CommandType;
  payload?: unknown;
}

export interface ControllerMetrics {
  controller: string;
  episodes: number;
  successRate: number;
  avgLandingSpeed: number | null;
  avgFuelUsedKg: number | null;
  avgDistanceToTarget: number | null;
  failureRate: number;
}

export interface CompareResult {
  baseline: ControllerMetrics;
  ppo: ControllerMetrics | null;
  episodes: number;
}
