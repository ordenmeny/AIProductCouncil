import type { MeetingState, UserAnswer } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {})
      },
      ...init
    });
  } catch (error) {
    throw new Error(
      "Backend API недоступен. Запустите FastAPI на 127.0.0.1:8000: uv run uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000"
    );
  }
  if (!response.ok) {
    const text = await response.text();
    throw new Error(extractError(text, response.status));
  }
  return response.json() as Promise<T>;
}

function extractError(text: string, status: number): string {
  try {
    const parsed = JSON.parse(text);
    if (typeof parsed.detail === "string") return parsed.detail;
  } catch {
    return text || `HTTP ${status}`;
  }
  return text || `HTTP ${status}`;
}

export async function createMeeting(idea: string): Promise<MeetingState> {
  const result = await request<{ meeting: MeetingState }>("/api/meetings", {
    method: "POST",
    body: JSON.stringify({ idea })
  });
  return result.meeting;
}

export async function submitAnswers(meetingId: string, answers: UserAnswer[]): Promise<MeetingState> {
  return request<MeetingState>(`/api/meetings/${meetingId}/answers`, {
    method: "POST",
    body: JSON.stringify({ answers })
  });
}

export async function advanceMeeting(meetingId: string): Promise<MeetingState> {
  const result = await request<{ meeting: MeetingState }>(`/api/meetings/${meetingId}/advance`, {
    method: "POST"
  });
  return result.meeting;
}

export function exportUrl(meetingId: string, kind: "protocol" | "final-plan"): string {
  return `${API_BASE}/api/meetings/${meetingId}/export/${kind}.md`;
}
