import { useState } from "react";
import { num } from "../format";
import type { ContentBlock, StrandsMessage } from "../record/transcript";
import { Long } from "./common";

function asText(v: unknown): string {
  if (typeof v === "string") {
    // tool output often arrives as a JSON string: show it parsed, readably
    const s = v.trim();
    if ((s.startsWith("{") && s.endsWith("}")) || (s.startsWith("[") && s.endsWith("]"))) {
      try {
        const o = JSON.parse(s);
        // a lone {"output": "..."} is the tool's text; anything more is shown whole
        if (o && typeof o === "object" && !Array.isArray(o) && Object.keys(o).length === 1 && typeof (o as { output?: unknown }).output === "string") return (o as { output: string }).output;
        return JSON.stringify(o, null, 1);
      } catch {
        return v;
      }
    }
    return v;
  }
  return JSON.stringify(v, null, 1);
}

function usageText(u: Record<string, number> | undefined): string | null {
  if (!u) return null;
  const parts = [`${num(u.inputTokens ?? 0)} in`, `${num(u.outputTokens ?? 0)} out`];
  if (u.cacheReadInputTokens) parts.push(`${num(u.cacheReadInputTokens)} cache read`);
  return `${parts.join(" · ")} tokens`;
}

function Block({ b, reasoning }: { b: ContentBlock; reasoning: boolean }) {
  if (b.text !== undefined) return <Long text={b.text} />;
  if (b.reasoningContent) return reasoning ? <div className="text muted small">{b.reasoningContent.reasoningText?.text}</div> : null;
  if (b.toolUse)
    return (
      <div className="call">
        <span className="small">toolUse <b className="mono">{b.toolUse.name}</b></span>
        <pre>{JSON.stringify(b.toolUse.input, null, 1)}</pre>
      </div>
    );
  if (b.toolResult) {
    const text = (b.toolResult.content ?? []).map((c) => (c.text !== undefined ? c.text : asText(c.json ?? c))).join("\n");
    return (
      <div className="call">
        <span className="small muted">toolResult{b.toolResult.status && b.toolResult.status !== "success" ? <b> · {b.toolResult.status}</b> : ""}</span>
        <Long text={text} limit={700} mono />
      </div>
    );
  }
  return <pre className="muted small">{JSON.stringify(b).slice(0, 600)}</pre>;
}

function StrandsTranscript({ messages }: { messages: StrandsMessage[] }) {
  const [reasoning, setReasoning] = useState(false);
  const hasReasoning = messages.some((m) => (m.content ?? []).some((b) => b.reasoningContent));
  return (
    <>
      <div className="controls">
        <span>{messages.length} messages</span>
        {hasReasoning && (
          <label>
            <input type="checkbox" checked={reasoning} onChange={(e) => setReasoning(e.target.checked)} /> show reasoning
          </label>
        )}
        <span className="muted"></span>
      </div>
      {messages.map((m, i) => {
        const content = Array.isArray(m.content) ? m.content : [{ text: String(m.content) }];
        const isTool = content.length > 0 && content.every((b) => b.toolResult);
        const kind = isTool ? "tool" : m.role;
        const u = usageText(m.metadata?.usage);
        return (
          <div key={i} className={`msg ${kind}`}>
            <div className="who">
              <span>#{i}</span>
              <span>{m.role}</span>
              {u && <span className="u">{u}</span>}
            </div>
            {content.map((b, j) => <Block key={j} b={b} reasoning={reasoning} />)}
          </div>
        );
      })}
    </>
  );
}

/** A transcript as the record stores it (any JSON: read as Strands messages). */
export function TranscriptView({ transcript }: { transcript: unknown }) {
  const [raw, setRaw] = useState(false);
  if (!Array.isArray(transcript) || transcript.length === 0) return <div className="empty">empty</div>;
  return (
    <>
      <StrandsTranscript messages={transcript as StrandsMessage[]} />
      <details style={{ marginTop: 10 }} onToggle={(e) => setRaw((e.target as HTMLDetailsElement).open)}>
        <summary className="small">raw JSON</summary>
        {raw && <pre className="textbox mono small">{JSON.stringify(transcript, null, 1).slice(0, 400_000)}</pre>}
      </details>
    </>
  );
}
