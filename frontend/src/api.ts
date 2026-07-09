import type { MeetingState, UserAnswer } from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    ...init
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `HTTP ${response.status}`);
  }
  return response.json() as Promise<T>;
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
