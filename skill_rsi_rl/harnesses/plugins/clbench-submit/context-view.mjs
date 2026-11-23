import { visibleSources } from './context-find.mjs';

export function focusSource(text, start_line, end_line) {
  const lines = text.split('\n');
  if (!Number.isSafeInteger(start_line) || !Number.isSafeInteger(end_line) ||
      start_line < 1 || end_line < start_line || end_line > lines.length ||
      (start_line === 1 && end_line === lines.length)) return null;
  const kept = lines.slice(start_line - 1, end_line).join('\n');
  const before = start_line > 1 ? '[…]\n' : '';
  const after = end_line < lines.length ? '\n[…]' : '';
  const rendered = before + kept + after;
  if (rendered.length >= text.length || Buffer.byteLength(rendered) >= Buffer.byteLength(text)) return null;
  return { text: kept, rendered, leading_marker_lines: before ? 1 : 0 };
}

function setPreviews(catalog, sources) {
  const frequency = new Map();
  const candidates = new Map();
  for (const entry of sources) {
    const samples = new Map();
    const lines = entry.text.split('\n');
    for (let index = 0; index < lines.length; index++) {
      const text = lines[index].slice(0, 96).replace(/[\uD800-\uDBFF]$/, '');
      if (text.trim() && !samples.has(text)) samples.set(text, (entry.start_line ?? 1) + index);
    }
    for (const text of samples.keys()) frequency.set(text, (frequency.get(text) ?? 0) + 1);
    candidates.set(entry.source, samples);
  }
  for (const item of catalog) {
    let count = Infinity;
    for (const [text, line] of candidates.get(item.source) ?? []) {
      if (frequency.get(text) < count) {
        item.preview = text;
        item.preview_line = line;
        count = frequency.get(text);
      }
    }
  }
}

export function projectContextView(options, query, known, folded) {
  const { messages } = options;
  const original = visibleSources(messages, query, known);
  const currentPrefix = query.requestId + ':';
  const latestUser = messages.findLastIndex(message => message.role === 'user' && message.source?.kind === 'user');
  const currentSeen = original.some(entry =>
    entry.source === currentPrefix + 'prompt' && entry.message_number === latestUser + 1);
  const occurrences = new Map();
  for (const message of messages) {
    if (message.role !== 'user' || message.source?.kind !== 'user') continue;
    const record = known.get(message.id);
    const block = message.content[0];
    if (!record || block?.type !== 'text' ||
        !(block.text === record.prefix || block.text.startsWith(record.prefix + '\n\n'))) continue;
    for (const entry of record.entries) {
      occurrences.set(entry.source, (occurrences.get(entry.source) ?? 0) + 1);
    }
  }

  const catalog = original.map(entry => ({
    source: entry.source,
    message_number: entry.message_number,
    message_id: entry.message_id,
    line: entry.line,
    chars: entry.text.length,
    lines: entry.text.split('\n').length,
    kind: entry.source.endsWith(':prompt') ? 'prompt' : 'feedback',
    preview: '',
    editable: currentSeen && !entry.source.startsWith(currentPrefix) && occurrences.get(entry.source) === 1,
    folded: false,
  }));
  const bySource = new Map(catalog.map(item => [item.source, item]));
  const edits = new Map();
  const visible = new Map(original.map(entry => [entry.source, { ...entry }]));
  for (const entry of original) {
    const item = bySource.get(entry.source);
    if (!item.editable || !folded.has(item.source)) continue;
    const value = folded.get?.(item.source);
    let rendered;
    if (value && typeof value === 'object') {
      const focus = focusSource(entry.text, value.start_line, value.end_line);
      if (!focus) continue;
      rendered = focus.rendered;
      item.focused_range = { start_line: value.start_line, end_line: value.end_line };
      visible.set(item.source, { ...entry, text: focus.text, start_line: value.start_line });
    } else {
      rendered = `[DSH context view: folded ${entry.source}]`;
      item.folded = true;
      if (typeof value === 'string') item.model_summary = value;
      visible.delete(item.source);
    }
    const index = item.message_number - 1;
    if (!edits.has(index)) edits.set(index, new Map());
    edits.get(index).set(item.source, rendered);
  }
  setPreviews(catalog, visible.values());
  if (!edits.size) return { request: options, sources: original, catalog, applied: [] };

  const projected = messages.slice();
  const applied = [];
  for (const [index, targets] of edits) {
    const message = messages[index];
    const [block, ...rest] = message.content;
    const record = known.get(message.id);
    let line = 1;
    const parts = record.entries.map(entry => {
      let text = entry.text;
      const item = bySource.get(entry.source);
      if (targets.has(entry.source)) {
        text = targets.get(entry.source);
        applied.push(entry.source);
      }
      if (item?.message_number === index + 1) {
        item.line = line + (item.focused_range?.start_line > 1 ? 1 : 0);
        if (visible.has(entry.source)) visible.get(entry.source).line = item.line;
      }
      line += text.split('\n').length + 1;
      return text;
    });
    const text = parts.join('\n\n') + block.text.slice(record.prefix.length);
    const content = Object.freeze([Object.freeze({ ...block, text }), ...rest]);
    projected[index] = Object.freeze({ ...message, content });
  }
  const sources = [...visible.values()];
  const request = Object.freeze({ ...options, messages: Object.freeze(projected) });
  return { request, sources, catalog, applied };
}
