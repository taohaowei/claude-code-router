// Shared in-memory store for DeepSeek-style reasoning_content replay across
// multi-turn conversations. Used by both deepseek and openrouter transformers.
//
// DeepSeek V4 thinking mode requires that the reasoning_content emitted on a
// previous assistant turn (the one that performed tool calls) be sent back on
// subsequent requests. Otherwise the API returns 400:
//
//   The `reasoning_content` in the thinking mode must be passed back to the API.
//
// Anthropic-style clients (e.g. Claude Code CLI) do not retain or replay the
// thinking content across turns, so the transformer captures it on response
// and reinjects it on the next request, keyed by sorted tool_call IDs joined
// by "|". Tool-call IDs round-trip cleanly because tool_result messages
// reference them.

const STORE = new Map<string, string>();
const STORE_LIMIT = 1000;

export function storeReasoning(key: string, value: string): void {
  if (STORE.size >= STORE_LIMIT) {
    const firstKey = STORE.keys().next().value;
    if (firstKey !== undefined) STORE.delete(firstKey);
  }
  STORE.set(key, value);
}

export function getReasoning(key: string): string | undefined {
  return STORE.get(key);
}

export function keyFromToolCalls(toolCalls: any): string | null {
  if (!Array.isArray(toolCalls) || toolCalls.length === 0) return null;
  const ids: string[] = toolCalls
    .map((tc: any) => tc && tc.id)
    .filter((id: any): id is string => typeof id === "string" && id.length > 0);
  if (ids.length === 0) return null;
  return ids.slice().sort().join("|");
}
