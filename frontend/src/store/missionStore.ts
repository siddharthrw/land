import { create } from "zustand";
import type { Command, MissionEvent, MissionState } from "../types/mission";
import { missionSocket } from "../services/missionSocket";

interface MissionStore {
  connectionStatus: "connecting" | "open" | "closed";
  state: MissionState | null;
  prevState: MissionState | null;
  eventLog: MissionEvent[];
  ppoAvailable: boolean;
  comparePanelOpen: boolean;
  toggleComparePanel: () => void;
  connect: () => void;
  send: (cmd: Command) => void;
  launch: () => void;
  pause: () => void;
  resume: () => void;
  restart: () => void;
  abort: () => void;
  setController: (controller: "baseline" | "ppo" | "manual") => void;
  setSimSpeed: (speed: number) => void;
  manualInput: (throttle: number, tilt: number) => void;
}

const MAX_LOG = 60;
let hasConnected = false;

export const useMissionStore = create<MissionStore>((set, get) => ({
  connectionStatus: "closed",
  state: null,
  prevState: null,
  eventLog: [],
  ppoAvailable: false,
  comparePanelOpen: false,
  toggleComparePanel: () => set((s) => ({ comparePanelOpen: !s.comparePanelOpen })),

  connect: () => {
    if (hasConnected) return;
    hasConnected = true;
    missionSocket.onStatus((status) => set({ connectionStatus: status }));
    missionSocket.onMessage((msg) => {
      set((prev) => ({
        prevState: prev.state,
        state: msg.state,
        ppoAvailable: msg.ppoAvailable,
        eventLog:
          msg.events.length > 0
            ? [...prev.eventLog, ...msg.events].slice(-MAX_LOG)
            : prev.eventLog,
      }));
    });
    missionSocket.connect();
  },

  send: (cmd) => missionSocket.send(cmd),
  launch: () => get().send({ type: "launch" }),
  pause: () => get().send({ type: "pause" }),
  resume: () => get().send({ type: "resume" }),
  restart: () => {
    set({ eventLog: [] });
    get().send({ type: "restart" });
  },
  abort: () => get().send({ type: "abort" }),
  setController: (controller) => get().send({ type: "set_controller", payload: controller }),
  setSimSpeed: (speed) => get().send({ type: "set_sim_speed", payload: speed }),
  manualInput: (throttle, tilt) =>
    get().send({ type: "manual_input", payload: { throttle, tilt } }),
}));
