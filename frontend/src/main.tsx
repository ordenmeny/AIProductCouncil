import React from "react";
import ReactDOM from "react-dom/client";
import { ArrowRight, Download, Loader2, MessageSquare, Play, Send } from "lucide-react";
import { advanceMeeting, createMeeting, exportUrl, submitAnswers } from "./api";
import type { ClarifyingQuestion, MeetingPhase, MeetingState, UserAnswer } from "./types";
import "./styles.css";

const phaseLabels: Record<MeetingPhase, string> = {
  intake: "Ввод идеи",
  clarifying_questions: "Вопросы",
  waiting_user_answers: "Ожидаем ответы",
  individual_analysis: "Индивидуальный анализ",
  debate: "Обсуждение",
  mvp_proposals: "MVP",
  vote: "Голосование",
  final_report: "Финальный отчет",
  completed: "Готово"
};

function App() {
  const [idea, setIdea] = React.useState("");
  const [meeting, setMeeting] = React.useState<MeetingState | null>(null);
  const [answers, setAnswers] = React.useState<Record<string, string>>({});
  const [activeDoc, setActiveDoc] = React.useState<"plan" | "protocol" | "json">("plan");
  const [loading, setLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  async function run<T>(fn: () => Promise<T>, after?: (result: T) => void) {
    setLoading(true);
    setError(null);
    try {
      const result = await fn();
      after?.(result);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Неизвестная ошибка");
    } finally {
      setLoading(false);
    }
  }

  function startMeeting(event: React.FormEvent) {
    event.preventDefault();
    run(() => createMeeting(idea), (created) => {
      setMeeting(created);
      setAnswers(Object.fromEntries(created.questions.map((question) => [question.id, ""])));
    });
  }

  function saveAnswers(event: React.FormEvent) {
    event.preventDefault();
    if (!meeting) return;
    const payload: UserAnswer[] = meeting.questions.map((question) => ({
      question_id: question.id,
      answer: answers[question.id]?.trim() || "Пока нет точного ответа; агент может зафиксировать разумное допущение."
    }));
    run(() => submitAnswers(meeting.id, payload), setMeeting);
  }

  function advance() {
    if (!meeting) return;
    run(() => advanceMeeting(meeting.id), setMeeting);
  }

  const canAdvance = meeting && meeting.phase !== "waiting_user_answers" && meeting.phase !== "completed";

  return (
    <main className="app-shell">
      <section className="workspace">
        <header className="topbar">
          <div>
            <h1>AI Product Council</h1>
            <p>Мультиагентный созвон для проектирования MVP, рисков и плана реализации.</p>
          </div>
          {meeting && <span className="phase-pill">{phaseLabels[meeting.phase]}</span>}
        </header>

        {!meeting ? (
          <form className="idea-form" onSubmit={startMeeting}>
            <label htmlFor="idea">Идея сервиса, стартапа или фичи</label>
            <textarea
              id="idea"
              value={idea}
              onChange={(event) => setIdea(event.target.value)}
              minLength={10}
              placeholder="Например: сервис для автоматического согласования внутренних заявок между отделами..."
              required
            />
            <button type="submit" disabled={loading || idea.trim().length < 10}>
              {loading ? <Loader2 className="spin" size={18} /> : <Play size={18} />}
              Запустить созвон
            </button>
          </form>
        ) : (
          <div className="meeting-grid">
            <section className="panel main-panel">
              <div className="panel-heading">
                <div>
                  <span className="eyebrow">Исходная идея</span>
                  <p>{meeting.idea}</p>
                </div>
                {canAdvance && (
                  <button onClick={advance} disabled={loading}>
                    {loading ? <Loader2 className="spin" size={18} /> : <ArrowRight size={18} />}
                    Следующая фаза
                  </button>
                )}
              </div>

              {meeting.phase === "waiting_user_answers" && (
                <QuestionForm
                  questions={meeting.questions}
                  answers={answers}
                  loading={loading}
                  onChange={(questionId, answer) => setAnswers((current) => ({ ...current, [questionId]: answer }))}
                  onSubmit={saveAnswers}
                />
              )}

              <MessageFeed meeting={meeting} />
            </section>

            <aside className="panel side-panel">
              <h2>Итог</h2>
              {meeting.vote_summary ? (
                <div className="summary-block">
                  <strong>{decisionLabel(meeting.vote_summary.final_decision)}</strong>
                  <p>{meeting.vote_summary.main_next_step}</p>
                </div>
              ) : (
                <p className="muted">Итог появится после голосования агентов.</p>
              )}
              {meeting.final_documents && (
                <div className="downloads">
                  <a href={exportUrl(meeting.id, "protocol")}>
                    <Download size={16} />
                    Протокол
                  </a>
                  <a href={exportUrl(meeting.id, "final-plan")}>
                    <Download size={16} />
                    План
                  </a>
                </div>
              )}
              {meeting.errors.length > 0 && (
                <div className="error-list">
                  <h3>Ошибки модели</h3>
                  {meeting.errors.map((item) => (
                    <p key={item}>{item}</p>
                  ))}
                </div>
              )}
            </aside>
          </div>
        )}

        {meeting?.final_documents && (
          <section className="panel docs-panel">
            <div className="tabs">
              <button className={activeDoc === "plan" ? "active" : ""} onClick={() => setActiveDoc("plan")}>
                Итоговый план
              </button>
              <button className={activeDoc === "protocol" ? "active" : ""} onClick={() => setActiveDoc("protocol")}>
                Протокол
              </button>
              <button className={activeDoc === "json" ? "active" : ""} onClick={() => setActiveDoc("json")}>
                JSON
              </button>
            </div>
            <pre>
              {activeDoc === "plan"
                ? meeting.final_documents.final_plan_md
                : activeDoc === "protocol"
                  ? meeting.final_documents.protocol_md
                  : JSON.stringify(meeting, null, 2)}
            </pre>
          </section>
        )}

        {error && <div className="toast">{error}</div>}
      </section>
    </main>
  );
}

function QuestionForm({
  questions,
  answers,
  loading,
  onChange,
  onSubmit
}: {
  questions: ClarifyingQuestion[];
  answers: Record<string, string>;
  loading: boolean;
  onChange: (questionId: string, answer: string) => void;
  onSubmit: (event: React.FormEvent) => void;
}) {
  return (
    <form className="question-form" onSubmit={onSubmit}>
      <div className="section-title">
        <MessageSquare size={18} />
        <h2>Вопросы агентов</h2>
      </div>
      {questions.map((question) => (
        <label key={question.id} className="question-item">
          <span>{question.agent_name}</span>
          <strong>{question.question}</strong>
          {question.reason && <em>{question.reason}</em>}
          <textarea
            value={answers[question.id] ?? ""}
            onChange={(event) => onChange(question.id, event.target.value)}
            placeholder="Ответьте или оставьте пустым, чтобы система зафиксировала допущение."
          />
        </label>
      ))}
      <button type="submit" disabled={loading}>
        {loading ? <Loader2 className="spin" size={18} /> : <Send size={18} />}
        Сохранить ответы
      </button>
    </form>
  );
}

function MessageFeed({ meeting }: { meeting: MeetingState }) {
  const visible = meeting.messages.filter((message) => message.phase !== "clarifying_question");
  if (visible.length === 0) {
    return <p className="muted">После ответов пользователя здесь появятся реплики агентов по фазам созвона.</p>;
  }
  return (
    <div className="message-feed">
      {visible.map((message) => (
        <article key={message.id} className="message">
          <header>
            <span>{message.agent_name}</span>
            <small>{phaseLabelsFromAgent(message.phase)}</small>
          </header>
          <p>{message.content}</p>
          {message.structured?.insights?.length ? (
            <ul>
              {message.structured.insights.slice(0, 3).map((insight) => (
                <li key={insight}>{insight}</li>
              ))}
            </ul>
          ) : null}
        </article>
      ))}
    </div>
  );
}

function phaseLabelsFromAgent(phase: string) {
  const labels: Record<string, string> = {
    individual_analysis: "Анализ",
    debate: "Спор",
    mvp_proposal: "MVP",
    vote: "Голос"
  };
  return labels[phase] ?? phase;
}

function decisionLabel(decision: string | null) {
  const labels: Record<string, string> = {
    go: "Запускать",
    go_after_clarification: "Запускать после уточнений",
    no_go: "Не запускать",
    pivot_or_narrow_mvp: "Сузить или изменить MVP"
  };
  return decision ? labels[decision] ?? decision : "Решение не определено";
}

ReactDOM.createRoot(document.getElementById("root")!).render(<App />);
