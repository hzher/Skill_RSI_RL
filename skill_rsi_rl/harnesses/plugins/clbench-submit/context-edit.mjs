import { probeEvidence, probeSpec } from './context-probe.mjs';
import { focusSource } from './context-view.mjs';

const PAGE_SIZE = 20;
const READ_LINES = 12;
const READ_CHARS = 1800;
const sourceId = { type: 'string', minLength: 1 };
const versionNumber = { type: 'integer', minimum: 0 };
const positiveInteger = { type: 'integer', minimum: 1 };
const evidenceQuestion = {
  type: 'string', maxLength: 160, description: 'Optional single factual question this lookup should settle.',
};

const operations = {
  ctx_probe: { ...probeSpec, properties: { ...probeSpec.properties, question: evidenceQuestion } },
  ctx_list: {
    description: 'List public evidence in this request, newest first. Returns stable source IDs, sizes, folded state, editability and a view version. Page with next_offset.',
    properties: { offset: versionNumber, question: evidenceQuestion }, required: [],
  },
  ctx_read: {
    description: 'Read exact text from a visible source by ID, starting at an original line and column. Continue with the returned cursor when a short excerpt is insufficient.',
    properties: { source: sourceId, start_line: positiveInteger, start_column: positiveInteger, question: evidenceQuestion },
    required: ['source'],
  },
  ctx_fold: {
    description: 'Replace one past public source in subsequent requests with a recoverable marker. Supply a short factual summary of what may still matter. Your summary is a model note. The current question and its feedback stay visible.',
    properties: { source: sourceId, version: versionNumber, summary: { type: 'string', minLength: 1, maxLength: 320 } },
    required: ['source', 'summary'],
  },
  ctx_focus: {
    description: 'Keep an exact inclusive range of original lines from one old source in subsequent model requests and reversibly hide both sides. Select a table or passage using a hit from ctx_probe/ctx_find or the catalog. Use when the rest of the source distracts from the current decision.',
    properties: { source: sourceId, version: versionNumber, start_line: positiveInteger, end_line: positiveInteger },
    required: ['source', 'start_line', 'end_line'],
  },
  ctx_restore: {
    description: 'Recover the full original text of one folded or focused source still in the native session. Use when evidence outside the visible excerpt could change your action.',
    properties: { source: sourceId, version: versionNumber }, required: ['source'],
  },
};

export const isContextEditTool = (name, editing = true) => Object.hasOwn(operations, name) &&
  (editing || name === 'ctx_list' || name === 'ctx_read' || name === 'ctx_probe');

function sourcePage(text, line, column, firstLine = 1) {
  const lines = text.split('\n');
  const lastLine = firstLine + lines.length - 1;
  if (!Number.isSafeInteger(line) || !Number.isSafeInteger(column) || line < firstLine || line > lastLine ||
      column < 1 || column > (lines[line - firstLine].length || 1)) throw new Error('Cursor outside source.');
  const rows = [];
  let remaining = READ_CHARS;
  while (line <= lastLine && rows.length < READ_LINES && remaining > 0) {
    const row = lines[line - firstLine];
    const value = row.slice(column - 1, column - 1 + remaining);
    rows.push({ line, column, text: value });
    remaining -= value.length;
    column += value.length;
    if (column <= row.length) break;
    line++;
    column = 1;
  }
  return { rows, next: line <= lastLine ? { start_line: line, start_column: column } : null };
}

function operate(name, args, view, state, editing) {
  if (name === 'ctx_probe') {
    const { question, ...selection } = args;
    return { version: view.version, ...probeEvidence(view.sources, selection, state.requestId) };
  }
  if (name === 'ctx_list') {
    const offset = args.offset ?? 0;
    if (!Number.isSafeInteger(offset) || offset < 0) throw new Error('Invalid catalog offset.');
    const items = view.catalog.slice(offset, offset + PAGE_SIZE);
    return { version: view.version, items: editing ? items : items.map(item => ({ ...item, editable: false })),
      next_offset: offset + items.length < view.catalog.length ? offset + items.length : null };
  }
  const item = view.catalog.find(entry => entry.source === args.source);
  if (!item) throw new Error('Source is unavailable in the originating request.');
  if (name === 'ctx_read') {
    const source = view.sources.find(entry => entry.source === item.source);
    if (!source) throw new Error('Restore the folded source before reading it.');
    const firstLine = source.start_line ?? 1;
    return { source: item.source, version: view.version,
      ...sourcePage(source.text, args.start_line ?? firstLine, args.start_column ?? 1, firstLine) };
  }
  if ((args.version !== undefined && args.version !== view.version) || view.version !== state.edit.version) {
    throw new Error('Stale view version. Use ctx_list for the latest version.');
  }
  if (!item.editable) throw new Error('Current, ambiguous, or unverified source is protected.');
  if (name === 'ctx_fold') {
    if (typeof args.summary !== 'string' || !args.summary.trim() || args.summary.length > 320) {
      throw new Error('Provide a model summary of 1–320 characters.');
    }
    if (item.folded || item.focused_range) throw new Error('Restore this source before changing its view.');
    state.edit.folded.set(item.source, args.summary.trim());
  } else if (name === 'ctx_focus') {
    const source = view.sources.find(entry => entry.source === item.source);
    if (!source || item.focused_range) throw new Error('Restore this source before focusing it.');
    if (!focusSource(source.text, args.start_line, args.end_line)) {
      throw new Error('Select an in-bounds line range that shortens the source.');
    }
    state.edit.folded.set(item.source, { start_line: args.start_line, end_line: args.end_line });
  } else {
    if (!item.folded && !item.focused_range) throw new Error('Source is already visible.');
    state.edit.folded.delete(item.source);
  }
  state.edit.version++;
  return { source: item.source, version: state.edit.version,
    next_view: { ctx_fold: 'folded', ctx_focus: 'focused', ctx_restore: 'restored' }[name] };
}

export function registerContextEdits(agent, state, capture, maxSteps, editing = true) {
  const enabled = Object.entries(operations).filter(([name]) => isContextEditTool(name, editing));
  const disposers = enabled.map(([name, spec]) => agent.ctx.tools.register({
    name, description: spec.description,
    parameters: { type: 'object', properties: spec.properties, required: spec.required, additionalProperties: false },
    output: { schema: { type: 'string' }, render: (_args, value) => [{ type: 'text', text: value }] },
    async execute(args, exec) {
      exec.signal.throwIfAborted();
      const view = state.editCalls.get(exec.callId);
      state.editCalls.delete(exec.callId);
      if (!view || state.submitted || view.name !== name) throw new Error('The originating context call is no longer active.');
      const event = { query_request_id: state.requestId, llm_call_id: view.callId, tool_call_id: exec.callId,
        tool: name, source: args.source, input_version: view.version,
        ...(args.question && { evidence_question: args.question }) };
      try {
        const result = operate(name, args, view, state, editing);
        capture.event('context_edit', 'context_operation', agent.session.id, { ...event, status: 'ok', version: state.edit.version,
          ...(name === 'ctx_probe' && { terms: args.terms, scope: args.scope ?? 'all', match: args.match ?? 'all',
            kind_filter: args.kind ?? 'all', matching_sources: result.total, returned_sources: result.hits.map(hit => hit.source) }),
          ...(name === 'ctx_focus' && { start_line: args.start_line, end_line: args.end_line }) });
        return JSON.stringify(result);
      } catch (error) {
        capture.event('context_edit', 'context_operation', agent.session.id, { ...event, status: 'rejected', error: error.message });
        throw error;
      }
    },
  }, { when: () => state.steps + 1 < maxSteps && !state.submitted }));
  return () => { for (const dispose of disposers) dispose(); };
}
