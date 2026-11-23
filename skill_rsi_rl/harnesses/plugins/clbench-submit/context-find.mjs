const MAX_MATCHES = 3;
const MAX_LINE = 320;

// Bind public substrings to the original DSH user message. Older entries
// remain searchable only while that message and its original text are present.
export function visibleSources(messages, query, known) {
  const feedback = Array.isArray(query.publicFeedback) ? query.publicFeedback : [];
  const parts = [...feedback.map((text, i) => ({ label: `feedback-${i + 1}`, text })),
    { label: 'prompt', text: query.publicPrompt }];
  const complete = parts.every(part => typeof part.text === 'string');
  const prefix = complete ? parts.map(part => part.text).join('\n\n') : '';
  const sources = [];
  const seen = new Set();
  for (let i = messages.length - 1; i >= 0; --i) {
    const message = messages[i];
    if (message.role !== 'user' || message.source?.kind !== 'user' || !message.id) continue;
    const body = message.content[0];
    if (body?.type !== 'text') continue;
    if (!known.has(message.id) && prefix && (body.text === prefix || body.text.startsWith(prefix + '\n\n'))) {
      let line = 1;
      const entries = parts.map(({ label, text }) => {
        const entry = { source: `${query.requestId}:${label}`, text, line };
        line += text.split('\n').length + 1;
        return entry;
      });
      known.set(message.id, { prefix, entries });
    }
    const record = known.get(message.id);
    if (!record || !(body.text === record.prefix || body.text.startsWith(record.prefix + '\n\n'))) continue;
    for (const entry of record.entries.toReversed()) {
      if (!entry.text || seen.has(entry.source)) continue;
      seen.add(entry.source);
      sources.push({ ...entry, message_id: message.id, message_number: i + 1 });
    }
  }
  return sources;
}

function clippedLine(lines, index, firstLine, column = 0) {
  const text = lines[index];
  const start = Math.min(Math.max(0, column - 50), Math.max(0, text.length - MAX_LINE));
  return { line: firstLine + index, text: text.slice(start, start + MAX_LINE),
    ...(text.length > MAX_LINE && { start_column: start + 1, truncated: true }) };
}

export function findText(sources, text) {
  if (typeof text !== 'string' || !text.trim() || text.length > 160 || /[\r\n]/.test(text)) {
    throw new Error('Use a literal substring of one line, from 1 to 160 characters.');
  }
  const needle = text.toLowerCase();
  const matches = [];
  let truncated = false;
  for (const { source, message_id, message_number, text: original, line, start_line = 1 } of sources) {
    const lines = original.split(/\r?\n/);
    let header = -1;
    for (const [i, value] of lines.entries()) {
      // CL-Bench uses -+- separators. Markdown tables use pipes and colons.
      if (i > 0 && /^[-+:|\s]+$/.test(value) && value.includes('---')) header = i - 1;
      else if (!value.trim() || (header >= 0 && lines[header].includes('|') && !value.includes('|'))) header = -1;
      const column = value.toLowerCase().indexOf(needle);
      if (column < 0) continue;
      if (matches.length === MAX_MATCHES) return { matches, truncated: true };
      const indices = new Set([header, i - 1, i, i + 1].filter(n => n >= 0 && n < lines.length));
      const context = [...indices].sort((a, b) => a - b)
        .map(n => clippedLine(lines, n, line, n === i ? column : 0));
      truncated ||= context.some(item => item.truncated);
      matches.push({ source, message_id, message_number, line: line + i, source_line: start_line + i, context });
    }
  }
  return { matches, truncated };
}

export function registerContextFind(agent, state, capture, maxSteps) {
  return agent.ctx.tools.register({
    name: 'ctx_find',
    description: 'Find a literal substring in the public task prompts and feedback visible in this model request. Returns nearby source lines and table headers. Use when rereading could change your action.',
    parameters: { type: 'object', properties: {
      text: { type: 'string', minLength: 1, maxLength: 160 },
      scope: { type: 'string', enum: ['current'] },
    }, required: ['text'], additionalProperties: false },
    output: { schema: { type: 'string' }, render: (_args, value) => [{ type: 'text', text: value }] },
    async execute({ text, scope = 'current' }, exec) {
      exec.signal.throwIfAborted();
      const snapshot = state.findCalls.get(exec.callId);
      state.findCalls.delete(exec.callId);
      if (!snapshot || scope !== 'current') throw new Error('The originating model request is unavailable for this search.');
      const result = findText(snapshot.sources, text);
      if (snapshot.version !== undefined) result.version = snapshot.version;
      capture.event('context_find', 'context_find', agent.session.id, {
        query_request_id: state.requestId, llm_call_id: snapshot.callId, tool_call_id: exec.callId,
        text, scope, ...result,
      });
      return JSON.stringify(result);
    },
  // DSH resolves tools just before the pre-step hook increments state.steps.
  }, { when: () => state.steps + 1 < maxSteps });
}
