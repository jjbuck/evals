// Transcripts as the record stores them: each trial's Strands messages.
export type ContentBlock = {
  text?: string;
  toolUse?: { toolUseId: string; name: string; input: unknown };
  toolResult?: { toolUseId: string; status?: string; content: { text?: string; json?: unknown }[] };
  reasoningContent?: { reasoningText?: { text: string } };
  [k: string]: unknown;
};
export type StrandsMessage = { role: string; content: ContentBlock[]; metadata?: { usage?: Record<string, number> } };
export type Transcript = StrandsMessage[];
