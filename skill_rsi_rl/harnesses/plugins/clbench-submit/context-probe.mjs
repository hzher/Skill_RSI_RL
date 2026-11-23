const MAX_TERMS = 4;
const MAX_TERM_LENGTH = 96;
const MAX_EXAMPLES = 4;
const EXCERPT_LENGTH = 320;

export const probeSpec = {
  description: 'Check how broadly literal evidence occurs across public sources in this request. Match terms on the same line, count distinct source IDs, and return one cited line per sampled source. Current/history refer to the originating benchmark query versus earlier queries still visible. A source ID denotes a prompt or a feedback segment, not an independent real-world observation.',
  properties: {
    terms: { type: 'array', minItems: 1, maxItems: MAX_TERMS, items: { type: 'string', minLength: 1, maxLength: MAX_TERM_LENGTH } },
    match: { type: 'string', enum: ['all', 'any'], description: 'Default all. Literal case-insensitive matching within one line.' },
    scope: { type: 'string', enum: ['current', 'history', 'all'], description: 'Default all visible public sources.' },
    kind: { type: 'string', enum: ['prompt', 'feedback', 'all'], description: 'Default all.' },
  },
  required: ['terms'],
};

function excerpt(line, matches) {
  const first = Math.min(...matches.map(match => match.index));
  const last = Math.max(...matches.map(match => match.index + match[0].length));
  const start = Math.min(Math.max(0, line.length - EXCERPT_LENGTH), first,
    Math.max(0, first - 40, last - EXCERPT_LENGTH));
  return { text: line.slice(start, start + EXCERPT_LENGTH), start_column: start + 1, clipped: line.length > EXCERPT_LENGTH };
}

export function probeEvidence(sources, args, currentRequestId) {
  if (!args || typeof args !== 'object' || Object.keys(args).some(key => !Object.hasOwn(probeSpec.properties, key))) {
    throw new Error('Unknown probe argument.');
  }
  const { terms, match = 'all', scope = 'all', kind = 'all' } = args;
  if (!Array.isArray(terms) || !terms.length || terms.length > MAX_TERMS ||
      terms.some(term => typeof term !== 'string' || !term.trim() || term.length > MAX_TERM_LENGTH || /[\r\n]/.test(term))) {
    throw new Error('Supply one to four nonempty single-line literals, each at most 96 characters.');
  }
  if (!['all', 'any'].includes(match) || !['current', 'history', 'all'].includes(scope) ||
      !['prompt', 'feedback', 'all'].includes(kind)) throw new Error('Invalid probe selection.');
  if (typeof currentRequestId !== 'string' || !currentRequestId) throw new Error('Originating request is unavailable.');
  const patterns = terms.map(term => new RegExp(RegExp.escape(term), 'iu'));
  const linesBySource = new Map();
  const hits = [];
  let matchingLines = 0;
  for (const entry of sources) {
    const id = /^(.+):(prompt|feedback-[1-9]\d*)$/.exec(entry.source);
    if (!id) continue;
    const sourceKind = id[2] === 'prompt' ? 'prompt' : 'feedback';
    if ((scope === 'current' && id[1] !== currentRequestId) || (scope === 'history' && id[1] === currentRequestId) ||
        (kind !== 'all' && kind !== sourceKind)) continue;
    const matchedLines = linesBySource.get(entry.source) ?? new Set();
    linesBySource.set(entry.source, matchedLines);
    for (const [index, line] of entry.text.split(/\r?\n/).entries()) {
      const sourceLine = (entry.start_line ?? 1) + index;
      if (matchedLines.has(sourceLine)) continue;
      const termHits = patterns.map(pattern => pattern.exec(line));
      const matches = termHits.filter(Boolean);
      if (!matches.length || (match === 'all' && matches.length < patterns.length)) continue;
      matchingLines++;
      if (!matchedLines.size && hits.length < MAX_EXAMPLES) {
        hits.push({ source: entry.source, kind: sourceKind, message_id: entry.message_id,
          line: entry.line + index, source_line: sourceLine, ...excerpt(line, matches),
          matched_terms: terms.filter((term, position) => termHits[position]) });
      }
      matchedLines.add(sourceLine);
    }
  }
  const total = [...linesBySource.values()].filter(lines => lines.size > 0).length;
  return { total, matching_lines: matchingLines, sources_examined: linesBySource.size, hits,
    truncated: matchingLines > hits.length || hits.some(row => row.clipped),
    count_unit: 'distinct matching public source IDs; separate IDs may repeat the same observation' };
}
