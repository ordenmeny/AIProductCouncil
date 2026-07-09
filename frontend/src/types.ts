export type AgentId =
  | "product_manager"
  | "tech_lead"
  | "ux_researcher"
  | "security_data_expert"
  | "skeptic_risk_officer";

export type MeetingPhase =
  | "intake"
  | "clarifying_questions"
  | "waiting_user_answers"
  | "individual_analysis"
  | "debate"
  | "mvp_proposals"
  | "vote"
  | "final_report"
  | "completed";

export type AgentPhase =
  | "clarifying_question"
  | "individual_analysis"
  | "debate"
  | "mvp_proposal"
  | "vote";

export type VoteDecision =
  | "go"
  | "go_after_clarification"
  | "no_go"
  | "pivot_or_narrow_mvp";

export interface ClarifyingQuestion {
  id: string;
  agent_id: AgentId;
  agent_name: string;
  question: string;
  reason: string;
}

export interface UserAnswer {
  question_id: string;
  answer: string;
}

export interface AgentStructuredResponse {
  agent: string;
  agent_id: AgentId;
  phase: AgentPhase;
  summary: string;
  mvp_priority: string[];
  roadmap_items: string[];
  open_questions: string[];
  insights: string[];
  risks: string[];
  main_risk: string;
  decision: VoteDecision | null;
  next_step: string;
  reason: string;
  target_audience: string[];
  user_problem: string[];
  core_mvp_features: string[];
  tech_stack: string[];
  tech_stack_reasoning: string[];
  user_scenario: string[];
  user_screens: string[];
  processed_data: string[];
  data_sensitivity: string[];
  security_measures: string[];
  risk_mitigations: string[];
}

export interface AgentMessage {
  id: string;
  created_at: string;
  agent_id: AgentId;
  agent_name: string;
  phase: AgentPhase;
  content: string;
  structured: AgentStructuredResponse | null;
  raw_response: string | null;
  validation_error: string | null;
}

export interface VoteSummary {
  decisions: Partial<Record<VoteDecision, number>>;
  final_decision: VoteDecision | null;
  key_mvp_features: string[];
  key_risks: string[];
  open_questions: string[];
  insights: string[];
  main_next_step: string;
}

export interface FinalDocuments {
  protocol_md: string;
  final_plan_md: string;
}

export interface MeetingState {
  id: string;
  created_at: string;
  updated_at: string;
  idea: string;
  phase: MeetingPhase;
  questions: ClarifyingQuestion[];
  user_answers: UserAnswer[];
  messages: AgentMessage[];
  vote_summary: VoteSummary | null;
  final_documents: FinalDocuments | null;
  errors: string[];
}
