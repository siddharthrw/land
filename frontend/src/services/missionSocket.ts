import type { Command, MissionMessage } from "../types/mission";

const WS_URL = import.meta.env.VITE_MISSION_WS_URL ?? "ws://localhost:8000/ws";

type MessageHandler = (msg: MissionMessage) => void;
type StatusHandler = (status: "connecting" | "open" | "closed") => void;

class MissionSocket {
  private ws: WebSocket | null = null;
  private messageHandlers: MessageHandler[] = [];
  private statusHandlers: StatusHandler[] = [];
  private reconnectTimer: number | null = null;

  connect() {
    this.notifyStatus("connecting");
    const ws = new WebSocket(WS_URL);
    this.ws = ws;

    ws.onopen = () => this.notifyStatus("open");
    ws.onclose = () => {
      this.notifyStatus("closed");
      this.scheduleReconnect();
    };
    ws.onerror = () => {
      ws.close();
    };
    ws.onmessage = (ev) => {
      try {
        const parsed = JSON.parse(ev.data) as MissionMessage;
        this.messageHandlers.forEach((h) => h(parsed));
      } catch {
        // ignore malformed frames
      }
    };
  }

  private scheduleReconnect() {
    if (this.reconnectTimer) return;
    this.reconnectTimer = window.setTimeout(() => {
      this.reconnectTimer = null;
      this.connect();
    }, 1500);
  }

  send(command: Command) {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(command));
    }
  }

  onMessage(handler: MessageHandler) {
    this.messageHandlers.push(handler);
    return () => {
      this.messageHandlers = this.messageHandlers.filter((h) => h !== handler);
    };
  }

  onStatus(handler: StatusHandler) {
    this.statusHandlers.push(handler);
    return () => {
      this.statusHandlers = this.statusHandlers.filter((h) => h !== handler);
    };
  }

  private notifyStatus(status: "connecting" | "open" | "closed") {
    this.statusHandlers.forEach((h) => h(status));
  }
}

export const missionSocket = new MissionSocket();
